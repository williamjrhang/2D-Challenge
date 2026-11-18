#!/usr/bin/env python3
"""Local obstacle detector (YOLO-World, no training): stairs, curtain, rubble, survivors.

Runs on the CPU (~8 frames/s at imgsz 480). For every detection it measures where the
thing is relative to the robot (box + depth + TF, see locator.py). When the same kind
is IN THE ROBOT'S PATH and CLOSE for `confirm_frames` frames in a row, it sends:

    stairs  -> stairs_ahead     curtain -> curtain_ahead     rubble -> rubble_ahead
    person  -> person   (mission_controller stops and asks the scene VLM)

and mission_controller + special_obstacles do the crossing (with the gait change).

publish_events: false (default) = only LOG what it would send. Safe for testing.
When a trained model exists, set `model` to its .pt file; class names must then be the
kinds themselves (stairs, curtain, rubble, person) and `prompts` is ignored.

Topics
  in   <image_topic>                    sensor_msgs/CompressedImage
  in   /mission/state                   std_msgs/String  (events only in EXPLORE / GO_TO_EXIT)
  out  /perception/event                std_msgs/String
  out  /perception/obstacles            std_msgs/String  JSON of every frame's detections
  out  /perception/obstacle_image/compressed  sensor_msgs/CompressedImage  boxes drawn (RViz)
"""

import json
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '2')   # keep numpy / torch to 2 cores
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')   # idle worker threads sleep, not spin
os.environ.setdefault('KMP_BLOCKTIME', '0')
import threading
import time

import cv2
cv2.setNumThreads(1)
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from lite3_explore.locator import Locator
from lite3_explore.terrain import classify, estimate_floor
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformException

EVENTS = {'stairs': 'stairs_ahead', 'curtain': 'curtain_ahead',
          'rubble': 'rubble_ahead', 'potholes': 'potholes_ahead', 'person': 'person'}
COLOURS = {'stairs': (255, 160, 0), 'curtain': (255, 255, 255),
           'rubble': (0, 140, 255), 'potholes': (180, 0, 180), 'person': (0, 0, 255)}


class ObstacleDetector(Node):

    def __init__(self):
        super().__init__('obstacle_detector')
        names = []

        def p(name, default):
            names.append(name)
            self.declare_parameter(name, default)

        p('image_topic', '/camera/color/image_raw/compressed')
        p('model', 'yolov8s-worldv2.pt')
        p('model_dir', os.path.expanduser('~/challenge_ws/models'))
        # YOLO-World text prompts and which kind each one means (same length lists).
        # No 'person' by default: YOLO-World also finds the real people around the arena
        # (tested: it boxed lab students behind the walls). Survivors are left to the VLM.
        p('prompts', ['metal staircase', 'stairs with handrails', 'white fabric curtain',
                      'cloth curtain in a doorway', 'uneven wooden floor', 'pile of rubble'])
        p('kinds', ['stairs', 'stairs', 'curtain', 'curtain', 'rubble', 'rubble'])
        # CPU budget (live 17/11: default torch used all cores -> load 14/10, the robot data
        # bridge, twist_mux and Fast-LIO starved and the robot stopped responding).
        p('threads', 2)                  # CPU cores YOLO may use
        p('nice', 10)                    # lower priority than the robot's own programs
        p('imgsz', 320)
        p('conf', 0.25)
        # Close-up curtain: when the camera is right at the curtain the whole image is
        # fabric and YOLO-World only gives ~0.10 (tested live). A curtain box covering most
        # of the image is accepted at this lower confidence.
        p('closeup_conf', 0.08)
        p('closeup_area', 0.70)
        # Close-up curtain by colour + depth (live test at the curtain: 87% low-saturation
        # pixels, blocking points at 0.36 m). Wood is brown (high saturation) -> no trigger.
        p('white_curtain_check', True)
        p('white_frac', 0.70)            # share of low-saturation pixels in the image centre
        p('white_block_dist_m', 0.8)     # something blocking within this distance
        p('white_block_points', 2000)
        p('rate_hz', 2.0)
        p('trigger_distance_m', 1.2)     # obstacle closer than this ...
        p('path_half_width_m', 0.5)      # ... and within this of the robot's centre line
        p('person_distance_m', 3.0)
        p('confirm_frames', 3)
        p('cooldown_s', 15.0)
        p('publish_events', False)
        # Depth floor check (rubble = bumps above floor, potholes = holes below it).
        # Floor height calibrates itself whenever the floor ahead looks flat.
        p('use_floor_check', True)
        p('floor_half_width_m', 0.25)    # robot is 0.37 m wide
        p('wall_min_m', 0.20)            # anything taller than this is a wall, not rubble
        p('floor_z_start', -0.356)       # floor height in base_link, measured live 17/11 (standing)
        self.P = {n: self.get_parameter(n).value for n in names}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Everything is measured relative to the robot (base_link), NOT the map:
        # the detector must keep working even when SLAM / the map frame is down.
        self.locator = Locator(self, self.tf_buffer, global_frame='base_link')

        self.create_subscription(CompressedImage, self.P['image_topic'], self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/mission/state', self.on_state, 10)
        self.event_pub = self.create_publisher(String, '/perception/event', 10)
        self.dets_pub = self.create_publisher(String, '/perception/obstacles', 10)
        self.img_pub = self.create_publisher(CompressedImage,
                                             '/perception/obstacle_image/compressed', 2)

        self.image = None
        self.mission_state = None
        self.streak = {k: 0 for k in EVENTS}
        self.last_event = {}
        self.model = None
        self.kind_of = {}
        self.running = True
        self.floor_z = self.P['floor_z_start']
        threading.Thread(target=self.loop, daemon=True).start()

    def on_image(self, msg):
        self.image = msg

    def on_state(self, msg):
        self.mission_state = msg.data

    # ================= model =================

    def load_model(self):
        import torch
        torch.set_num_threads(int(self.P['threads']))
        torch.set_num_interop_threads(1)
        try:
            os.nice(int(self.P['nice']))
        except OSError:
            pass
        from ultralytics import YOLO, YOLOWorld
        path = self.P['model']
        if not os.path.isabs(path):
            path = os.path.join(self.P['model_dir'], path)
        os.makedirs(self.P['model_dir'], exist_ok=True)
        t = time.monotonic()
        if 'world' in os.path.basename(path):
            prompts, kinds = self.P['prompts'], self.P['kinds']
            if len(prompts) != len(kinds):
                raise ValueError('prompts and kinds must have the same length')
            cwd = os.getcwd()
            os.chdir(self.P['model_dir'])          # ultralytics downloads into cwd
            try:
                self.model = YOLOWorld(path)
                self.model.set_classes(list(prompts))
            finally:
                os.chdir(cwd)
            self.kind_of = {i: k for i, k in enumerate(kinds)}
        else:                                      # trained model: class names = kinds
            self.model = YOLO(path)
            self.kind_of = {i: n for i, n in self.model.names.items()}
        self.get_logger().info(
            f'Model {os.path.basename(path)} ready in {time.monotonic() - t:.0f} s '
            f'({"PUBLISHING events" if self.P["publish_events"] else "LOG ONLY"})')

    # ================= main loop =================

    def loop(self):
        try:
            self.load_model()
        except Exception as e:
            self.get_logger().error(f'Could not load YOLO model: {e!r}')
            return
        period = 1.0 / self.P['rate_hz']
        while self.running and rclpy.ok():
            start = time.monotonic()
            msg, self.image = self.image, None
            if msg is not None:
                try:
                    self.process(msg)
                except Exception as e:
                    self.get_logger().warn(f'Frame failed: {e!r}')
            time.sleep(max(0.0, period - (time.monotonic() - start)))

    def process(self, msg):
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]
        snap = self.locator.snapshot()
        pose = snap['pose'] if snap else self.locator.robot_pose()
        low = min(self.P['conf'], self.P['closeup_conf'])
        r = self.model.predict(img, imgsz=self.P['imgsz'], conf=low, verbose=False)[0]

        dets, seen = [], set()
        for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            kind = self.kind_of.get(int(cls))
            if kind not in EVENTS:
                continue
            area = (box[2] - box[0]) * (box[3] - box[1]) / float(w * h)
            closeup = kind == 'curtain' and area >= self.P['closeup_area']
            if conf < self.P['conf'] and not closeup:
                continue
            d = {'kind': kind, 'conf': round(conf, 2), 'box': [round(v) for v in box]}
            xy = self.locator.locate_box(box, w, h, snap) if snap else None
            if xy is not None and pose is not None:
                fwd, left = self.locator.to_robot(xy, pose)
                d.update(forward_m=round(fwd, 2), left_m=round(left, 2), located_by='depth')
                close = 0.0 < fwd <= (self.P['person_distance_m'] if kind == 'person'
                                      else self.P['trigger_distance_m'])
                in_path = kind == 'person' or abs(left) <= self.P['path_half_width_m']
            else:
                # No depth: near the bottom-centre of the image means close and ahead.
                cx = (box[0] + box[2]) / 2
                d['located_by'] = 'image'
                close = box[3] > 0.85 * h
                in_path = w / 3 < cx < 2 * w / 3
            d['trigger'] = bool(close and in_path)
            dets.append(d)
            if d['trigger']:
                seen.add(kind)
            self.draw(img, d)

        if self.P['white_curtain_check'] and snap is not None and 'curtain' not in seen:
            dist, frac = self.white_curtain(img, snap)
            if dist is not None:
                seen.add('curtain')
                dets.append({'kind': 'curtain', 'conf': round(frac, 2), 'forward_m': round(dist, 2),
                             'located_by': 'colour+depth', 'trigger': True})
                cv2.putText(img, f'curtain {dist:.1f}m (white+depth)', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS['curtain'], 2)

        if self.P['use_floor_check'] and snap is not None:
            kind, dist, stats = self.floor_check(snap)
            if kind and dist <= self.P['trigger_distance_m']:
                seen.add(kind)
                d = {'kind': kind, 'conf': 1.0, 'forward_m': round(dist, 2),
                     'located_by': 'floor', 'trigger': True, **stats}
                dets.append(d)
                cv2.putText(img, f'{kind} {dist:.1f}m (depth)', (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[kind], 2)

        for kind in self.streak:
            self.streak[kind] = self.streak[kind] + 1 if kind in seen else 0
            if self.streak[kind] >= self.P['confirm_frames']:
                self.streak[kind] = 0
                self.send(kind, dets)

        self.dets_pub.publish(String(data=json.dumps(dets)))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            out = CompressedImage(format='jpeg', data=buf.tobytes())
            out.header = msg.header
            self.img_pub.publish(out)

    def white_curtain(self, img, snap):
        h, w = img.shape[:2]
        band = cv2.cvtColor(img[int(.2 * h):int(.7 * h), int(.25 * w):int(.75 * w)], cv2.COLOR_BGR2HSV)
        frac = float(((band[..., 1] < 35) & (band[..., 2] > 60)).mean())
        if frac < self.P['white_frac']:
            return None, frac
        pts = self.base_points(snap)
        if pts is None:
            return None, frac
        floor = self.floor_z if self.floor_z is not None else -0.32
        block = pts[(pts[:, 0] > 0.1) & (pts[:, 0] < self.P['white_block_dist_m'])
                    & (np.abs(pts[:, 1]) < 0.3) & (pts[:, 2] > floor + 0.12)]
        if len(block) < self.P['white_block_points']:
            return None, frac
        return float(np.percentile(block[:, 0], 5)), frac

    def base_points(self, snap):
        cloud = snap['cloud']
        try:
            R, t = self.locator.transform(self.locator.robot_frame, cloud.header.frame_id)
        except TransformException:
            return None
        pts = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'),
                                             skip_nans=True).astype(float)
        if len(pts) > 30000:
            pts = pts[::len(pts) // 30000 + 1]
        return pts @ R.T + t

    def floor_check(self, snap):
        cloud = snap['cloud']
        try:
            R, t = self.locator.transform(self.locator.robot_frame, cloud.header.frame_id)
        except TransformException:
            return None, None, {}
        pts = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'),
                                             skip_nans=True).astype(float)
        if len(pts) > 30000:
            pts = pts[::len(pts) // 30000 + 1]
        pts = pts @ R.T + t
        est = estimate_floor(pts)
        if est is not None:            # flat ahead: (re)calibrate floor height
            self.floor_z = est if self.floor_z is None else 0.8 * self.floor_z + 0.2 * est
        return classify(pts, self.floor_z, half_width=self.P['floor_half_width_m'],
                        wall_min=self.P['wall_min_m'])

    def send(self, kind, dets):
        now = time.monotonic()
        if now - self.last_event.get(kind, -1e9) < self.P['cooldown_s']:
            return
        self.last_event[kind] = now
        best = max((d for d in dets if d['kind'] == kind), key=lambda d: d['conf'])
        active = self.mission_state in (None, 'EXPLORE', 'GO_TO_EXIT')
        will = self.P['publish_events'] and active
        self.get_logger().warn(f'{"EVENT" if will else "would send"}: {EVENTS[kind]} {best}')
        if will:
            self.event_pub.publish(String(data=EVENTS[kind]))

    @staticmethod
    def draw(img, d):
        if 'box' not in d:
            return
        x1, y1, x2, y2 = d['box']
        c = COLOURS[d['kind']]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 3 if d['trigger'] else 1)
        dist = f' {d["forward_m"]:.1f}m' if 'forward_m' in d else ''
        cv2.putText(img, f'{d["kind"]} {d["conf"]:.2f}{dist}', (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
