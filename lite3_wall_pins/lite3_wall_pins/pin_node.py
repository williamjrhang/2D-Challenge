#!/usr/bin/env python3
"""wall_pins: OBSERVE-ONLY node that pins wall QR codes / printed pictures on the map.

Safety: this node only subscribes to sensors + TF and publishes visualisation / log
topics under /wall_pins/*. It has no cmd_vel publisher, no Nav2 action client, no gait or
control-mode publisher, and does not publish on /perception/* (which the exploration
mission_controller reacts to). test/test_observe_only.py checks this.

Topics
  in   image_topic        sensor_msgs/CompressedImage   front RGB (JPEG)
  in   camera_info_topic  sensor_msgs/CameraInfo        K, D, frame of the RGB image
  in   cloud_topic        sensor_msgs/PointCloud2       depth points (any frame, TF known)
  in   /tf, /tf_static
  out  /wall_pins/markers       visualization_msgs/MarkerArray  (map frame, latched)
  out  /wall_pins/events        std_msgs/String  JSON, one per NEW pin
  out  /wall_pins/debug_image   sensor_msgs/Image  detections drawn (only if subscribed)

Per frame with a detection:
  image stamp t_img -> depth cloud with |t_depth - t_img| <= max_sync_dt_s
  -> TF depth@t_depth -> colour@t_img (through the map frame, so robot motion between the
     two stamps is compensated) -> project into the image -> points inside the region
  -> robust surface point -> TF colour@t_img -> map. Any missing / stale input: no pin,
  and the reason is logged. There is no "point in front of the robot" fallback.
"""

import json
import math
import os
import signal
import threading
import time
from collections import deque

# CPU budget: must run BEFORE numpy / cv2 are imported. Measured on the real robot bag with
# a QR in view: OpenBLAS' default thread pool spin-waits on every core (~100% of a core in
# total); with 1 thread the node uses ~12%. The point clouds are small enough for 1 thread.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')       # torch (pictures) also capped per param
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import cv2  # noqa: E402
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from lite3_wall_pins import geometry as geo
from lite3_wall_pins import markers
from lite3_wall_pins.pin_store import PinStore, ViewPose, default_pins_path
from lite3_wall_pins.qr_detect import QrDetector

cv2.setNumThreads(1)


def stamp_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def tf_to_matrix(t):
    tr, q = t.transform.translation, t.transform.rotation
    return geo.make_transform((tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w))


class Skip(Exception):
    """A frame or detection that cannot be located; message = reason for the log."""


class WallPins(Node):

    def __init__(self):
        super().__init__('wall_pins')
        P = {}

        def p(name, default):
            self.declare_parameter(name, default)
            P[name] = self.get_parameter(name).value

        # topics / frames
        p('image_topic', '/camera/color/image_raw/compressed')
        p('camera_info_topic', '/camera/color/camera_info')
        p('cloud_topic', '/camera/depth/points')
        p('global_frame', 'map')
        p('robot_frame', 'base_link')
        # timing
        p('rate_hz', 3.0)                  # frames per second checked for QR codes
        p('max_sync_dt_s', 0.05)           # RGB <-> depth stamp mismatch allowed
        p('sync_wait_s', 0.6)              # wait this long for the matching depth frame
        p('cloud_buffer_s', 3.0)
        p('tf_timeout_s', 0.3)
        p('max_image_age_s', 2.0)          # older images (node clock - stamp) are skipped
        p('max_future_s', 0.5)             # images stamped further in the future: skipped
        # depth estimate
        p('min_range_m', 0.2)
        p('max_range_m', 4.0)
        p('qr_region_shrink', 0.8)         # use inner 80% of the QR quad
        p('picture_region_shrink', 0.6)
        p('min_points', 10)
        p('min_inlier_frac', 0.6)
        p('depth_tol_abs_m', 0.06)
        p('depth_tol_rel', 0.03)
        p('min_coverage', 0.10)            # depth points per region pixel
        # pins
        p('map_id', 'default')
        p('pins_file', '')                 # '' -> ~/.ros/lite3_wall_pins/<map_id>.json
        p('load_saved', True)
        p('merge_radius_m', 0.6)
        p('picture_merge_radius_m', 0.8)
        p('save_period_s', 5.0)
        p('marker_period_s', 1.0)
        p('show_view_pose', True)
        # QR
        p('enable_qr', True)
        p('qr_try_upscale', True)
        p('qr_min_size_px', 12.0)
        # pictures (EXPERIMENTAL, off by default)
        p('enable_pictures', False)
        p('picture_rate_hz', 1.0)
        p('picture_model', 'yolov8s-worldv2.pt')
        p('model_dir', '~/challenge_ws/models')
        p('picture_prompts', ['poster of a person on a wall', 'printed photo of a person'])
        p('person_prompt', 'person')
        p('picture_conf', 0.30)
        p('picture_imgsz', 480)
        p('picture_threads', 2)
        p('picture_nice', 10)
        p('picture_require_flat', True)
        p('picture_flat_tol_m', 0.03)
        p('picture_flat_tol_per_m2', 0.005)   # + this * range^2 (depth noise)
        p('picture_flat_frac', 0.85)
        # debug
        p('debug_image', True)
        self.P = P

        P['model_dir'] = os.path.expanduser(P['model_dir'])
        path = P['pins_file'] or default_pins_path(P['map_id'])
        self.store = PinStore(path, P['map_id'], P['global_frame'], P['merge_radius_m'],
                              P['picture_merge_radius_m'])
        if P['load_saved']:
            try:
                self.get_logger().info(self.store.load())
            except (ValueError, OSError, KeyError) as e:   # other map / unreadable file
                self.get_logger().fatal(f'cannot use pins file: {e}. Use the matching map_id, '
                                        'another pins_file, or move the file away.')
                raise SystemExit(1)
        else:
            self.get_logger().warn(f'load_saved=false: ignoring {self.store.path}; it WILL '
                                   'be overwritten when the first pin is saved')

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.qr = QrDetector(P['qr_try_upscale'], min_size_px=P['qr_min_size_px'])
        self.pictures = None

        self.lock = threading.Lock()
        self.image = None
        self.info = None
        self.clouds = deque()
        self.last_log = {}

        self.create_subscription(CompressedImage, P['image_topic'], self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, P['camera_info_topic'], self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(PointCloud2, P['cloud_topic'], self.on_cloud,
                                 qos_profile_sensor_data)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(markers.MarkerArray, '/wall_pins/markers',
                                                latched)
        self.event_pub = self.create_publisher(String, '/wall_pins/events', 10)
        self.debug_pub = self.create_publisher(Image, '/wall_pins/debug_image', 2)

        self.create_timer(P['marker_period_s'], self.publish_markers)
        self.create_timer(P['save_period_s'], self.save_if_dirty)
        self.publish_markers()

        self.running = True
        self.worker = threading.Thread(target=self.loop, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'wall_pins ready (observe only). QR={"on" if P["enable_qr"] else "off"}, '
            f'pictures={"on (EXPERIMENTAL)" if P["enable_pictures"] else "off"}, '
            f'map_id="{P["map_id"]}", file={self.store.path}')

    # ============================================================ callbacks

    def on_image(self, msg):
        with self.lock:
            self.image = msg

    def on_info(self, msg):
        with self.lock:
            self.info = msg

    def on_cloud(self, msg):
        t = stamp_sec(msg.header.stamp)
        with self.lock:
            self.clouds.append((t, msg))
            while self.clouds and self.clouds[-1][0] - self.clouds[0][0] > self.P['cloud_buffer_s']:
                self.clouds.popleft()

    # ============================================================ helpers

    def log(self, key, text, level='info', period=3.0):
        """Throttled per reason, so a code in view for a minute does not flood the log."""
        now = time.monotonic()
        if now - self.last_log.get(key, -1e9) < period:
            return
        self.last_log[key] = now
        # rclpy refuses different severities from ONE call site ("Logger severity cannot be
        # changed between calls"), so every level needs its own line.
        lg = self.get_logger()
        if level == 'error':
            lg.error(text)
        elif level == 'warn':
            lg.warn(text)
        elif level == 'debug':
            lg.debug(text)
        else:
            lg.info(text)

    def publish_markers(self):
        self.marker_pub.publish(markers.build(self.store, self.get_clock().now().to_msg(),
                                              self.P['show_view_pose']))

    def save_if_dirty(self):
        if self.store.dirty:
            try:
                self.store.save()
            except OSError as e:
                self.log('save', f'could not save pins to {self.store.path}: {e}', 'error')

    def match_cloud(self, t_img):
        """Depth cloud nearest t_img within max_sync_dt_s; waits for it to arrive."""
        dt_max = self.P['max_sync_dt_s']
        deadline = time.monotonic() + self.P['sync_wait_s']
        while True:
            with self.lock:
                clouds = list(self.clouds)
            best = min(clouds, key=lambda c: abs(c[0] - t_img), default=None)
            if best is not None and abs(best[0] - t_img) <= dt_max:
                return best[1]
            newest = clouds[-1][0] if clouds else None
            # a newer cloud already exists -> the matching one will never come
            if newest is not None and newest > t_img + dt_max:
                break
            if time.monotonic() > deadline or not self.running:
                break
            time.sleep(0.02)
        if best is None:
            raise Skip(f'no depth cloud received on {self.P["cloud_topic"]}')
        raise Skip(f'no depth frame within {dt_max * 1000:.0f} ms of the image '
                   f'(nearest {abs(best[0] - t_img) * 1000:.0f} ms)')

    def lookup(self, target, source, t, what):
        try:
            return tf_to_matrix(self.tf_buffer.lookup_transform(
                target, source, t, timeout=Duration(seconds=self.P['tf_timeout_s'])))
        except TransformException as e:
            raise Skip(f'TF {target} <- {source} not available at image time ({what}): '
                       f'{str(e).splitlines()[0][:160]}')

    # ============================================================ main loop

    def load_picture_model(self):
        from lite3_wall_pins.picture_detect import PictureDetector
        t = time.monotonic()
        try:
            self.pictures = PictureDetector(
                self.P['picture_model'], self.P['model_dir'], self.P['picture_prompts'],
                self.P['person_prompt'], self.P['picture_conf'], self.P['picture_imgsz'],
                self.P['picture_threads'], self.P['picture_nice'])
            self.get_logger().warn(
                f'EXPERIMENTAL picture detector loaded in {time.monotonic() - t:.0f} s '
                f'(prompts {self.P["picture_prompts"]}, flat check '
                f'{"on" if self.P["picture_require_flat"] else "OFF"})')
        except Exception as e:                       # QR keeps working without it
            self.pictures = None
            self.get_logger().error(f'picture detection disabled, model failed: {e!r}')

    def loop(self):
        if self.P['enable_pictures']:
            self.load_picture_model()
        period = 1.0 / max(self.P['rate_hz'], 0.1)
        pic_period = 1.0 / max(self.P['picture_rate_hz'], 0.01)
        last_pic = 0.0
        while self.running and rclpy.ok():
            start = time.monotonic()
            with self.lock:
                msg, self.image = self.image, None
            if msg is not None:
                run_pic = self.pictures is not None and start - last_pic >= pic_period
                if run_pic:
                    last_pic = start
                try:
                    self.process(msg, run_pic)
                except Exception as e:                  # never let the worker die
                    try:
                        self.log('crash', f'frame failed: {e!r}', 'error')
                    except Exception:
                        print(f'[wall_pins] frame failed: {e!r}', flush=True)
            time.sleep(max(0.0, period - (time.monotonic() - start)))

    # ============================================================ one frame

    def process(self, msg, run_pic):
        t_img = stamp_sec(msg.header.stamp)
        age = self.get_clock().now().nanoseconds * 1e-9 - t_img
        if age > self.P['max_image_age_s']:
            self.log('stale', f'image is {age:.1f} s old (> max_image_age_s) - skipped. '
                     'Clock offset between robot and this PC, or a lagging stream?', 'warn')
            return
        if age < -self.P['max_future_s']:
            self.log('future', f'image stamp is {-age:.1f} s in the future - skipped. Clocks of '
                     'robot and this PC are not synchronised (check chrony/NTP).', 'warn')
            return
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            self.log('decode', 'could not decode compressed image', 'warn')
            return

        dets = []                                # (kind, payload, polygon, extra)
        if self.P['enable_qr']:
            for q in self.qr.detect(img):
                dets.append(('qr', q.text, q.corners, {'size_px': round(q.size_px, 1)}))
        if run_pic:
            for d in self.pictures.detect(img):
                dets.append(('picture', '', geo.box_to_polygon(*d.box),
                             {'picture_score': round(d.picture_score, 2),
                              'person_score': round(d.person_score, 2),
                              'prompt': d.prompt}))
        status = [''] * len(dets)
        if dets:
            try:
                ctx = self.frame_context(msg, t_img, img.shape)
            except Skip as e:
                ctx = None
                for i, d in enumerate(dets):
                    status[i] = 'SKIP'
                    self.log(f'skip:{d[0]}:{d[1]}:{str(e)[:20]}',
                             f'{self.describe(d)} seen but NOT pinned: {e}', 'warn')
            if ctx is not None:
                for i, d in enumerate(dets):
                    status[i] = self.locate_and_pin(d, ctx)
        self.publish_debug(img, dets, status)

    def describe(self, d):
        return f'QR "{d[1]}"' if d[0] == 'qr' else 'picture'

    def frame_context(self, msg, t_img, shape):
        with self.lock:
            info = self.info
        if info is None:
            raise Skip(f'no camera_info on {self.P["camera_info_topic"]} yet')
        cloud = self.match_cloud(t_img)
        gf = self.P['global_frame']
        cam = info.header.frame_id or msg.header.frame_id
        img_time = Time.from_msg(msg.header.stamp)
        # depth@t_depth -> map -> colour@t_img
        T_map_depth = self.lookup(gf, cloud.header.frame_id, Time.from_msg(cloud.header.stamp),
                                  'depth frame')
        T_map_cam = self.lookup(gf, cam, img_time, 'camera frame')
        T_cam_depth = np.linalg.inv(T_map_cam) @ T_map_depth
        try:
            T_map_robot = self.lookup(gf, self.P['robot_frame'], img_time, 'robot pose')
        except Skip as e:
            T_map_robot = None
            self.log('viewpose', f'viewing pose unavailable, pins saved without it: {e}')

        pts = point_cloud2.read_points_numpy(cloud, field_names=('x', 'y', 'z'),
                                             skip_nans=True).astype(np.float64)
        pts = geo.apply_transform(T_cam_depth, pts)
        K = np.array(info.k, dtype=float).reshape(3, 3)
        return {'K': K, 'D': np.array(info.d, dtype=float), 'info_size': (info.width, info.height),
                'img_size': (shape[1], shape[0]), 'pts': pts, 'T_map_cam': T_map_cam,
                'T_map_robot': T_map_robot, 'stamp': t_img,
                'sync_ms': abs(stamp_sec(cloud.header.stamp) - t_img) * 1000}

    def locate_and_pin(self, det, ctx):
        kind, payload, poly_img, extra = det
        P = self.P
        name = self.describe(det)
        K = ctx['K']
        poly = geo.scale_polygon(poly_img, ctx['img_size'], ctx['info_size'])
        poly = geo.undistort_polygon(poly, K, ctx['D'])
        shrink = P['qr_region_shrink'] if kind == 'qr' else P['picture_region_shrink']
        region = geo.shrink_polygon(poly, shrink)
        pts = geo.region_points(ctx['pts'], K, region, P['min_range_m'], P['max_range_m'])

        area = max(geo.polygon_area(region), 1.0)
        reason = None
        if len(pts) / area < P['min_coverage'] and len(pts) < 4 * P['min_points']:
            reason = (f'too little valid depth in region ({len(pts)} points for '
                      f'{area:.0f} px; range {P["min_range_m"]}-{P["max_range_m"]} m)')
        else:
            p_cam, info = geo.robust_surface_point(pts, P['min_points'], P['min_inlier_frac'],
                                                   P['depth_tol_abs_m'], P['depth_tol_rel'])
            if p_cam is None:
                reason = info
        if reason is None and kind == 'picture' and P['picture_require_flat']:
            reason, extra = self.check_flat_picture(ctx, poly, info['range_m'], extra)
        if reason is not None:
            self.log(f'skip:{kind}:{payload}:{reason[:20]}',
                     f'{name} seen but NOT pinned: {reason}')
            return 'SKIP'

        p_map = geo.apply_transform(ctx['T_map_cam'], p_cam[None, :])[0]
        rng = info['range_m']
        if kind == 'qr':
            side_px = float(np.mean([np.linalg.norm(poly[i] - poly[(i + 1) % 4])
                                     for i in range(4)]))
            extra = dict(extra, size_m=round(side_px * rng / K[0, 0], 3))
        extra = dict(extra, range_m=round(rng, 2), depth_points=info['inliers'],
                     sync_ms=round(ctx['sync_ms'], 1))
        view = None
        if ctx['T_map_robot'] is not None:
            T = ctx['T_map_robot']
            rx, ry = float(T[0, 3]), float(T[1, 3])
            yaw = math.atan2(T[1, 0], T[0, 0])
            view = ViewPose(x=round(rx, 3), y=round(ry, 3), yaw=round(yaw, 4),
                            yaw_to_target=round(math.atan2(p_map[1] - ry, p_map[0] - rx), 4),
                            range_m=round(rng, 3))
        far = self.store.same_payload_elsewhere(payload, p_map[0], p_map[1]) if kind == 'qr' else []
        pin, created = self.store.observe(kind, payload, p_map, view,
                                          experimental=(kind == 'picture'), extra=extra,
                                          stamp=ctx['stamp'])
        if created:
            where = f'({pin.x:.2f}, {pin.y:.2f}, z {pin.z:.2f})'
            vp = (f'robot at ({view.x:.2f}, {view.y:.2f}) yaw {math.degrees(view.yaw):.0f} deg'
                  if view else 'no viewing pose')
            self.get_logger().warn(f'NEW PIN {pin.id} {name} at {where} m, {rng:.2f} m away, '
                                   f'{vp}, {extra}')
            if far:
                self.get_logger().warn(
                    f'{name} was already pinned as {[f.id for f in far]} more than '
                    f'{self.store.merge_radius_m} m away - kept as a separate pin '
                    '(second copy of the code, or map drift?)')
            self.event_pub.publish(String(data=json.dumps(pin.to_json())))
            self.save_if_dirty()                  # new pins are written immediately
            self.publish_markers()
            return 'NEW ' + pin.id
        self.log(f'seen:{pin.id}', f'{name} matched existing {pin.id} '
                 f'(x{pin.observations}) - no new pin', 'debug', 10.0)
        return pin.id

    def check_flat_picture(self, ctx, poly, rng, extra):
        """Printed picture = a flat sheet ON a wall. Fit the dominant (wall) plane in the
        box, then require the box CORE to lie on it and the plane to be vertical. A real
        person's torso fills the core and stands in front of the wall -> rejected."""
        P, K = self.P, ctx['K']
        tol = P['picture_flat_tol_m'] + P['picture_flat_tol_per_m2'] * rng ** 2
        full = geo.region_points(ctx['pts'], K, geo.shrink_polygon(poly, 0.9),
                                 P['min_range_m'], P['max_range_m'])
        core = geo.region_points(ctx['pts'], K, geo.shrink_polygon(poly, 0.5),
                                 P['min_range_m'], P['max_range_m'])
        plane = geo.dominant_plane(full, tol)
        if plane is None or len(core) < P['min_points']:
            return 'not enough depth to check that the picture is flat', extra
        core_frac = geo.on_plane_fraction(core, plane, tol)
        nz = abs(float((ctx['T_map_cam'][:3, :3] @ plane[0])[2]))   # normal, map z part
        extra = dict(extra, core_on_plane=round(core_frac, 2), plane_normal_z=round(nz, 2))
        if core_frac < P['picture_flat_frac']:
            return (f'centre of the box is not on the wall plane ({100 * core_frac:.0f}% '
                    f'within {tol * 100:.1f} cm) - probably a real person / 3D object'), extra
        if nz > 0.35:
            return (f'surface is not a wall (plane tilted {math.degrees(math.asin(min(nz, 1))):.0f}'
                    ' deg from vertical)'), extra
        return None, extra

    def publish_debug(self, img, dets, status):
        if not self.P['debug_image'] or self.debug_pub.get_subscription_count() == 0:
            return
        for (kind, payload, poly, _), st in zip(dets, status):
            c = (255, 190, 0) if kind == 'qr' else (0, 140, 255)
            if st == 'SKIP':
                c = (0, 0, 255)
            cv2.polylines(img, [np.int32(poly).reshape(-1, 1, 2)], True, c, 2)
            label = (f'{payload[:24]} ' if kind == 'qr' else 'picture? ') + st
            x, y = np.int32(poly).min(axis=0)
            cv2.putText(img, label, (int(x), max(12, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, c, 2)
        m = Image()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'camera_color_optical_frame'
        m.height, m.width = img.shape[:2]
        m.encoding, m.step = 'bgr8', img.shape[1] * 3
        m.data = img.tobytes()
        self.debug_pub.publish(m)

    def shutdown(self):
        self.running = False
        self.worker.join(timeout=2.0)
        if self.store.dirty:
            self.store.save()
            self.get_logger().info(f'saved {len(self.store.pins)} pins to {self.store.path}')


def _sigterm(signum, frame):
    raise KeyboardInterrupt         # same clean path as Ctrl+C: pins are saved on exit


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGTERM, _sigterm)
    try:
        node = WallPins()
    except SystemExit:
        rclpy.try_shutdown()
        raise
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
