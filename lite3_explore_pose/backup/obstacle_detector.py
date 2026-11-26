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
import math
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
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String
from tf2_ros import Buffer, TransformListener

from lite3_explore_pose.locator import Locator
from lite3_explore_pose.stairs import stairs_profile
from lite3_explore_pose.terrain import (classify, estimate_floor, shadow_check, surface_wrinkle_mm,
                                   update_floor, wall_ahead_m)
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
        # No YOLO rubble prompts: the whole arena is OSB wood and they fired everywhere;
        # rubble comes from the depth floor check. 'wall' prompts are DECOYS: with
        # agnostic_nms a wall box wins over a weaker 'curtain' box on the same wall, and
        # 'wall' itself never triggers anything.
        p('prompts', ['metal staircase', 'stairs with handrails', 'white fabric curtain',
                      'cloth curtain in a doorway', 'plain wooden wall', 'white wall panel'])
        p('kinds', ['stairs', 'stairs', 'curtain', 'curtain', 'wall', 'wall'])
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
        p('white_block_dist_m', 0.7)     # something blocking within this distance
        p('white_block_points', 2000)
        # Wall vs curtain (live 18/11: pale wood walls triggered 'curtain' 3x and the robot
        # walked into them). A curtain is fabric with folds; a wall is flat. Every curtain
        # trigger needs the surface ahead to be at least this wrinkled (mm). Flat or no
        # depth -> NOT a curtain. Tune: face a wall and the curtain at ~0.4 m and read
        # 'wrinkle_mm' in the log / on /perception/obstacles.
        p('curtain_flat_check', True)
        p('curtain_min_wrinkle_mm', 2.5)
        p('rate_hz', 2.0)
        p('trigger_distance_m', 1.2)     # obstacle closer than this ...
        p('curtain_trigger_m', 0.7)      # curtain: only this close, and only with depth
        p('sighting_range_m', 3.0)       # report obstacles seen this far away (for APPROACH)
        # Issue 7: curtains are reported further away (mission_controller drives to them on
        # purpose; the lidar maps the curtain as a wall). Within curtain_wrinkle_range_m the
        # surface AT the curtain must look like fabric (wrinkle) or the sighting is dropped;
        # further away depth is too noisy to judge, repeated sightings decide.
        p('curtain_sighting_range_m', 2.5)
        p('curtain_wrinkle_range_m', 1.0)
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
        p('floor_tolerance_m', 0.05)     # floor recalibration only within this of floor_z_start
        # Stairs from 3D SHAPE (stairs.py): >= 2 regular steps in the depth profile ahead.
        # 25/11: on photos of the arena stairs YOLO-World saw nothing and terrain_v1.pt said
        # 'rubble' 0.70-0.85, so the shape decides. A staircase is never sent as rubble/potholes.
        p('use_stairs_profile', True)
        p('stairs_trigger_m', 1.0)       # AI mode only at most 1 cell before the stairs (rule)
        p('stairs_max_yaw_deg', 20.0)    # AI mode climbs straight: must face the steps
        p('stairs_require_profile', True)  # YOLO 'stairs' alone never starts AI mode
        # Shadow fallback for potholes: when the depth check finds no potholes (or there is
        # no depth), look for shadow patches in the bottom third of the image (uneven ground
        # casts shadows). Skipped when depth sees rubble (slats cast shadows too) or a wall.
        # Tune: read 'shadow: dark ..' in the log on flat floor and in front of the potholes.
        p('use_shadow_check', True)
        p('shadow_dark_ratio', 0.6)      # cell darker than this x median floor brightness = shadow
        p('shadow_dark_frac', 0.15)      # share of shadow cells for 'potholes'
        p('shadow_min_brightness', 40.0) # floor darker than this: lighting check not trusted
        p('shadow_distance_m', 0.8)      # rough distance reported (bottom third = right ahead)
        # Issue 9 (desktop): depth from the robot's link_relay (thinned cloud, ~2 Hz) instead
        # of the raw 640x480 cloud. Defaults = today (raw topic, one cloud on demand per frame).
        p('cloud_topic', '/camera/depth/points')
        p('cloud_on_demand', True)
        self.P = {n: self.get_parameter(n).value for n in names}

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Everything is measured relative to the robot (base_link), NOT the map:
        # the detector must keep working even when SLAM / the map frame is down.
        # Depth on demand: ONE cloud per processed frame (rate_hz), not the permanent 30 Hz
        # stream. 25/11: at mission start the 640x480 cloud (~5 MB each) choked the robot
        # link, /livox/imu fell from 200 Hz to 5 Hz and FAST-LIO odometry went 24 s stale.
        self.locator = Locator(self, self.tf_buffer, global_frame='base_link',
                               cloud_topic=self.P['cloud_topic'],
                               cloud_on_demand=self.P['cloud_on_demand'])

        self.create_subscription(CompressedImage, self.P['image_topic'], self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/mission/state', self.on_state, 10)
        self.event_pub = self.create_publisher(String, '/perception/event', 10)
        self.dets_pub = self.create_publisher(String, '/perception/obstacles', 10)
        # Nearest wall (>= 20 cm tall) in the robot's path from depth, every frame, also while
        # crossing: special_obstacles stops rubble / pothole crossings before a wall. -1 = none.
        self.wall_pub = self.create_publisher(Float32, '/perception/wall_ahead', 10)
        # Stairs / curtain / rubble seen up to sighting_range_m away, with map position, so
        # mission_controller can drive there on purpose (APPROACH mode).
        self.seen_pub = self.create_publisher(String, '/perception/obstacle_seen', 10)
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
        snap = self.frame_snapshot()
        pose = snap['pose'] if snap else self.locator.robot_pose()
        low = min(self.P['conf'], self.P['closeup_conf'])
        r = self.model.predict(img, imgsz=self.P['imgsz'], conf=low, verbose=False,
                               agnostic_nms=True)[0]

        dets, seen = [], set()
        curtain_sightings = []
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
                if kind == 'curtain':
                    curtain_sightings.append((fwd, left, conf))   # judged below (issue 7)
                else:
                    self.report_sighting(kind, fwd, left)
                limit = (self.P['person_distance_m'] if kind == 'person'
                         else self.P['curtain_trigger_m'] if kind == 'curtain'
                         else self.P['trigger_distance_m'])
                close = 0.0 < fwd <= limit
                in_path = kind == 'person' or abs(left) <= self.P['path_half_width_m']
            else:
                # No depth: near the bottom-centre of the image means close and ahead.
                cx = (box[0] + box[2]) / 2
                d['located_by'] = 'image'
                close = box[3] > 0.85 * h and kind != 'curtain'   # curtain needs depth
                in_path = w / 3 < cx < 2 * w / 3
            d['trigger'] = bool(close and in_path)
            dets.append(d)
            if d['trigger']:
                seen.add(kind)
            self.draw(img, d)

        # Lighting check (venue light differs from the lab): brightness and how "white"
        # the centre of the view looks, against the curtain colour rule's limits.
        centre = cv2.cvtColor(img[int(.2 * h):int(.7 * h), int(.25 * w):int(.75 * w)],
                              cv2.COLOR_BGR2HSV)
        self.get_logger().info(
            f'lighting: brightness {float(centre[..., 2].mean()):.0f}/255 '
            f'(dark < 60)  saturation {float(centre[..., 1].mean()):.0f}/255  '
            f'white share {float(((centre[..., 1] < 35) & (centre[..., 2] > 60)).mean()):.2f} '
            f'(curtain rule > {self.P["white_frac"]})  best YOLO: '
            + (', '.join(f'{d["kind"]} {d["conf"]:.2f}' for d in dets if 'box' in d) or 'none'),
            throttle_duration_sec=3.0)

        if self.P['white_curtain_check'] and snap is not None and 'curtain' not in seen:
            dist, frac = self.white_curtain(img, snap)
            if dist is not None:
                seen.add('curtain')
                dets.append({'kind': 'curtain', 'conf': round(frac, 2), 'forward_m': round(dist, 2),
                             'located_by': 'colour+depth', 'trigger': True})
                cv2.putText(img, f'curtain {dist:.1f}m (white+depth)', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS['curtain'], 2)

        stairs_shape = False
        if self.P['use_stairs_profile'] and snap is not None:
            pts = self.base_points(snap)
            if pts is not None:
                st = stairs_profile(pts, self.floor_z)
                stairs_shape = st['stairs']
                self.get_logger().info(f'stairs shape: {st["reason"]}'
                                       + (f', first step {st["first_step_m"]} m, yaw '
                                          f'{st["yaw_deg"]} deg' if stairs_shape else ''),
                                       throttle_duration_sec=3.0)
                if stairs_shape:
                    fwd = st['first_step_m']
                    self.report_sighting('stairs', fwd + 0.3, 0.0)
                    yaw_ok = bool(st['yaw_deg'] is None or abs(st['yaw_deg']) <= self.P['stairs_max_yaw_deg'])
                    trig = bool(fwd <= self.P['stairs_trigger_m'] and yaw_ok)
                    dets.append({'kind': 'stairs', 'conf': 1.0, 'forward_m': fwd, 'left_m': 0.0,
                                 'located_by': 'depth_profile', 'trigger': trig,
                                 'steps': st['steps'], 'rises': st['rises'],
                                 'yaw_deg': st['yaw_deg']})
                    if trig:
                        seen.add('stairs')
                    elif fwd <= self.P['stairs_trigger_m']:
                        self.get_logger().warn(f'stairs {fwd} m ahead but not square '
                                               f'(yaw {st["yaw_deg"]} deg): no AI mode yet',
                                               throttle_duration_sec=2.0)
                    cv2.putText(img, f'stairs {fwd:.1f}m (shape, {st["steps"]} steps)',
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS['stairs'], 2)
        if self.P['stairs_require_profile'] and 'stairs' in seen and not any(
                d.get('located_by') == 'depth_profile' and d['trigger'] for d in dets):
            seen.discard('stairs')                     # YOLO alone: log only, no AI mode
            for d in dets:
                if d['kind'] == 'stairs':
                    d['trigger'] = False
                    d['note'] = 'no stairs shape in depth'

        depth_kind, depth_stats = None, {}
        if self.P['use_floor_check'] and snap is not None:
            kind, dist, stats = self.floor_check(snap)
            if stairs_shape and kind in ('rubble', 'potholes'):
                stats = dict(stats, suppressed=f'{kind}: stairs shape')
                # Issue 4: say it loudly, a false stairs shape would hide real rubble
                self.get_logger().warn(
                    f'floor check said {kind} at {dist} m (bump {stats.get("bump")}, hole '
                    f'{stats.get("hole")}, wall {stats.get("wall")}) but the stairs shape '
                    f'wins: NOT sent as {kind}', throttle_duration_sec=2.0)
                kind = None                            # the steps are not rubble
            if kind == 'rubble' and dist is not None:
                self.report_sighting('rubble', dist + 0.3, 0.0)   # +0.3: middle of the rubble
            depth_kind, depth_stats = kind, stats
            if 'bump' in stats:
                self.get_logger().info(
                    f'floor ahead: bump {stats["bump"]:.2f} (rubble > 0.25)  hole '
                    f'{stats["hole"]:.2f} (potholes > 0.08)  wall {stats["wall"]:.2f} '
                    f'wall_at {stats.get("wall_at", "-")}  floor_z '
                    f'{self.floor_z if self.floor_z is None else round(self.floor_z, 3)}  '
                    f'-> {kind or "flat"}', throttle_duration_sec=3.0)
            if kind and dist <= self.P['trigger_distance_m']:
                seen.add(kind)
                d = {'kind': kind, 'conf': 1.0, 'forward_m': round(dist, 2),
                     'located_by': 'floor', 'trigger': True, **stats}
                dets.append(d)
                cv2.putText(img, f'{kind} {dist:.1f}m (depth)', (10, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[kind], 2)

        # Depth found no potholes -> try the lighting in the bottom third of the frame.
        wall_ahead = depth_stats.get('wall', 0.0) >= 0.10
        if (self.P['use_shadow_check'] and 'potholes' not in seen
                and depth_kind is None and not wall_ahead and not stairs_shape):
            v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
            found, st = shadow_check(v, dark_ratio=self.P['shadow_dark_ratio'],
                                     dark_cells_frac=self.P['shadow_dark_frac'],
                                     min_median=self.P['shadow_min_brightness'])
            if st:
                self.get_logger().info(
                    f'shadow: dark {st["dark"]:.2f} (potholes > {self.P["shadow_dark_frac"]})  '
                    f'floor brightness {st["median"]:.0f}  darkest '
                    f'{st.get("darkest", "-")} -> {"potholes" if found else "none"}',
                    throttle_duration_sec=3.0)
            cv2.rectangle(img, (w // 4, 2 * h // 3), (3 * w // 4, h - 1),
                          COLOURS['potholes'], 2 if found else 1)
            if found:
                seen.add('potholes')
                dets.append({'kind': 'potholes', 'conf': st['dark'],
                             'forward_m': self.P['shadow_distance_m'],
                             'located_by': 'shadow', 'trigger': True, **st})
                cv2.putText(img, f'potholes (shadow {st["dark"]:.2f})', (10, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS['potholes'], 2)

        if 'curtain' in seen and self.P['curtain_flat_check']:
            wr = None
            pts = self.base_points(snap) if snap is not None else None
            if pts is not None:
                wr = surface_wrinkle_mm(pts, self.floor_z)
            fabric = wr is not None and wr >= self.P['curtain_min_wrinkle_mm']
            for d in dets:
                if d['kind'] == 'curtain':
                    d['wrinkle_mm'] = None if wr is None else round(wr, 1)
                    if not fabric:
                        d['trigger'] = False
            if not fabric:
                seen.discard('curtain')
                self.get_logger().info(
                    f'curtain candidate REJECTED: surface ahead is flat '
                    f'(wrinkle {"no depth" if wr is None else f"{wr:.1f} mm"} < '
                    f'{self.P["curtain_min_wrinkle_mm"]} mm) -> wall, not curtain',
                    throttle_duration_sec=2.0)
            else:
                self.get_logger().info(f'curtain candidate: wrinkle {wr:.1f} mm (fabric)',
                                       throttle_duration_sec=2.0)

        for fwd, left, conf in curtain_sightings:        # issue 7: sightings for APPROACH
            self.curtain_sighting(snap, fwd, left, conf)

        if stairs_shape:                               # camera models called the steps rubble
            for k in ('rubble', 'potholes'):
                seen.discard(k)
            for d in dets:
                if d['kind'] in ('rubble', 'potholes'):
                    d['trigger'] = False
                    d['note'] = 'stairs shape'
        for kind in self.streak:
            self.streak[kind] = self.streak[kind] + 1 if kind in seen else 0
            if self.streak[kind] >= self.P['confirm_frames']:
                self.streak[kind] = 0
                self.send(kind, dets)

        self.dets_pub.publish(String(data=json.dumps(dets)))
        if self.img_pub.get_subscription_count() == 0:
            return                       # nobody watching: skip the JPEG encode (CPU)
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

    def frame_snapshot(self):
        """Depth cloud + transforms for this frame, or None (same contract as
        Locator.snapshot()). Runs in the worker thread, so wait_cloud() may block."""
        if not self.locator.on_demand:
            return self.locator.snapshot()
        snap = self.locator.camera_snapshot()
        if snap is None:
            return None
        snap = self.locator.attach_cloud(snap, self.locator.wait_cloud(timeout_s=1.0))
        return snap if snap is not None and snap.get('cloud') is not None else None

    def base_points(self, snap):
        """Depth points in base_link, thinned to ~30k. Computed once per frame and shared
        by the floor check, the white-curtain check and the curtain wrinkle check."""
        if 'base_pts' not in snap:
            try:
                R, t = self.locator.transform(self.locator.robot_frame,
                                              snap['cloud'].header.frame_id)
            except TransformException:
                return None
            pts = self.locator.cloud_xyz(snap)
            if len(pts) > 30000:
                pts = pts[::len(pts) // 30000 + 1]
            snap['base_pts'] = pts @ R.T + t
        return snap['base_pts']

    def floor_check(self, snap):
        pts = self.base_points(snap)
        if pts is None:
            return None, None, {}
        est = estimate_floor(pts)
        # Flat ahead: (re)calibrate floor height, but only near the robot's standing height
        # (live 18/11 a raised flat board became 'the floor' -> real floor = false potholes).
        if est is not None and abs(est - self.P['floor_z_start']) > self.P['floor_tolerance_m']:
            self.get_logger().info(f'floor estimate {est:.3f} ignored (standing floor '
                                   f'{self.P["floor_z_start"]}): raised surface ahead',
                                   throttle_duration_sec=3.0)
        self.floor_z = update_floor(self.floor_z, est, self.P['floor_z_start'],
                                    self.P['floor_tolerance_m'])
        wall = wall_ahead_m(pts, self.floor_z, wall_min=self.P['wall_min_m'])
        self.wall_pub.publish(Float32(data=-1.0 if wall is None else wall))
        return classify(pts, self.floor_z, half_width=self.P['floor_half_width_m'],
                        wall_min=self.P['wall_min_m'])

    def to_map(self, fwd, left):
        """base_link point -> (map x, map y, robot x, robot y), or None without SLAM."""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        rx, ry = t.transform.translation.x, t.transform.translation.y
        c, s = math.cos(yaw), math.sin(yaw)
        return rx + c * fwd - s * left, ry + s * fwd + c * left, rx, ry

    def curtain_wrinkle_at(self, snap, fwd, left):
        """Wrinkle (mm) of the surface AT a curtain sighting, or None (no / too few points)."""
        pts = self.base_points(snap) if snap is not None else None
        if pts is None:
            return None
        sel = (np.abs(pts[:, 0] - fwd) < 0.35) & (np.abs(pts[:, 1] - left) < 0.45)
        sub = pts[sel] - np.array([0.0, left, 0.0])
        return surface_wrinkle_mm(sub, self.floor_z, x_max=fwd + 0.35)

    def curtain_sighting(self, snap, fwd, left, conf):
        """Every curtain decision is logged (issue 7)."""
        if not 0.0 < fwd <= self.P['curtain_sighting_range_m']:
            self.get_logger().info(f'CURTAIN sighting at {fwd:.2f} m: beyond '
                                   f'{self.P["curtain_sighting_range_m"]} m, ignored',
                                   throttle_duration_sec=3.0)
            return
        wr = None
        if fwd <= self.P['curtain_wrinkle_range_m'] and self.P['curtain_flat_check']:
            wr = self.curtain_wrinkle_at(snap, fwd, left)
            if wr is not None and wr < self.P['curtain_min_wrinkle_mm']:
                self.get_logger().info(
                    f'CURTAIN sighting at {fwd:.2f} m REJECTED: flat ({wr:.1f} mm < '
                    f'{self.P["curtain_min_wrinkle_mm"]} mm) -> wall', throttle_duration_sec=2.0)
                return
        self.get_logger().info(
            f'CURTAIN sighting at {fwd:.2f} m, {left:+.2f} m left, conf {conf:.2f}, wrinkle '
            f'{"n/a (far / no depth)" if wr is None else f"{wr:.1f} mm"} -> reported',
            throttle_duration_sec=2.0)
        self.report_sighting('curtain', fwd, left,
                             {'conf': round(float(conf), 2),
                              'wrinkle_mm': None if wr is None else round(wr, 1)})

    def report_sighting(self, kind, fwd, left, extra=None):
        if not self.P['publish_events'] or kind not in ('stairs', 'curtain', 'rubble'):
            return
        limit = (self.P['curtain_sighting_range_m'] if kind == 'curtain'
                 else self.P['sighting_range_m'])
        if not 0.0 < fwd <= limit:
            return
        m = self.to_map(fwd, left)
        if m is None:
            return
        self.seen_pub.publish(String(data=json.dumps(
            {'kind': kind, 'x': round(m[0], 2), 'y': round(m[1], 2),
             'rx': round(m[2], 2), 'ry': round(m[3], 2), **(extra or {})})))

    def send(self, kind, dets):
        now = time.monotonic()
        # A curtain can be detected while turning. Keep confirming it during the approach
        # so the controller can require a new close-range detection after alignment.
        confirming_curtain = kind == 'curtain' and self.mission_state == 'APPROACH'
        if not confirming_curtain and now - self.last_event.get(kind, -1e9) < self.P['cooldown_s']:
            return
        best = max((d for d in dets if d['kind'] == kind), key=lambda d: d['conf'])
        active = self.mission_state in (None, 'EXPLORE', 'GO_TO_EXIT', 'APPROACH')
        will = self.P['publish_events'] and active
        if will:
            # Start the cooldown only when the event is really sent: an event dropped while
            # the robot is busy (crossing / looking) must not hide the next real obstacle.
            self.last_event[kind] = now
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
