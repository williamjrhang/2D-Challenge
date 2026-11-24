"""Find where something seen in the colour image is on the map (Lab 7, Exercise III-B).

    box in colour image  +  depth point cloud  ->  3D point  ->  TF  ->  map (x, y)

The depth camera and the colour camera are separate frames on this robot, so the
depth points are moved into the colour camera frame with TF and projected into the
colour image; the points that land inside the box give the object's position.
"""

import math
import time

import numpy as np
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformException


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Locator:

    def __init__(self, node, tf_buffer, global_frame='map', robot_frame='base_link',
                 info_topic='/camera/color/camera_info', cloud_topic='/camera/depth/points',
                 map_topic=None, floor_z=-0.356, cloud_on_demand=False):
        self.tf = tf_buffer
        self.grid = None              # OccupancyGrid, only if map_topic is given (ray fallback)
        self.floor_z = floor_z        # floor height in the map frame (map origin = base_link at start)
        self.global_frame = global_frame
        self.robot_frame = robot_frame
        self.K = None
        self.color_frame = None
        self.cloud = None
        node.create_subscription(CameraInfo, info_topic, self.on_info, 10)
        # cloud_on_demand: no permanent depth subscription (the 640x480 cloud at 30 Hz crosses
        # the robot link once PER SUBSCRIBER and choked it on 24/11). wait_cloud() /
        # request_cloud() subscribe until ONE cloud arrives, then unsubscribe.
        self.node = node
        self.cloud_topic = cloud_topic
        self.on_demand = cloud_on_demand
        self._req = None
        if not cloud_on_demand:
            node.create_subscription(PointCloud2, cloud_topic, self.on_cloud,
                                     qos_profile_sensor_data)
        if map_topic:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
            node.create_subscription(OccupancyGrid, map_topic, self.on_map, qos)

    def on_map(self, msg):
        info = msg.info
        self.grid = {'res': info.resolution, 'ox': info.origin.position.x,
                     'oy': info.origin.position.y, 'w': info.width, 'h': info.height,
                     'data': np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)}

    def on_info(self, msg):
        self.K = np.array(msg.k, dtype=float).reshape(3, 3)
        self.color_frame = msg.header.frame_id

    def on_cloud(self, msg):
        self.cloud = msg

    def transform(self, target, source):
        t = self.tf.lookup_transform(target, source, Time())
        tr = t.transform.translation
        return quat_to_matrix(t.transform.rotation), np.array([tr.x, tr.y, tr.z])

    def robot_pose(self):
        """(x, y, yaw) of the robot on the map, or None."""
        try:
            t = self.tf.lookup_transform(self.global_frame, self.robot_frame, Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def snapshot(self):
        """Freeze the current depth cloud, transforms and robot pose, so a box found in
        an image taken NOW can be located later (e.g. after a slow VLM answer)."""
        if self.K is None or self.cloud is None:
            return None
        try:
            R_cd, t_cd = self.transform(self.color_frame, self.cloud.header.frame_id)
            R_mc, t_mc = self.transform(self.global_frame, self.color_frame)
        except TransformException:
            return None
        return {'cloud': self.cloud, 'R_cd': R_cd, 't_cd': t_cd, 'R_mc': R_mc, 't_mc': t_mc,
                'pose': self.robot_pose()}

    @staticmethod
    def cloud_xyz(snap):
        """x, y, z of the snapshot's cloud (NaNs dropped). Parsed once per snapshot and
        reused by every caller (parsing a full depth cloud is the expensive part)."""
        if 'xyz' not in snap:
            snap['xyz'] = point_cloud2.read_points_numpy(
                snap['cloud'], field_names=('x', 'y', 'z'), skip_nans=True).astype(float)
        return snap['xyz']

    def locate_box(self, box, image_w, image_h, snap=None):
        """box = (x1, y1, x2, y2) in pixels of an image of size image_w x image_h.
        Returns map (x, y) or None if there is no usable depth inside the box."""
        snap = snap or self.snapshot()
        if snap is None:
            return None
        R_cd, t_cd, R_mc, t_mc = snap['R_cd'], snap['t_cd'], snap['R_mc'], snap['t_mc']

        pts = self.cloud_xyz(snap)
        if len(pts) > 60000:                      # thin out for speed
            pts = pts[::len(pts) // 60000 + 1]
        pts = pts @ R_cd.T + t_cd                 # into colour camera frame
        pts = pts[(pts[:, 2] > 0.15) & (pts[:, 2] < 6.0)]
        if len(pts) == 0:
            return None

        # Scale the box to the camera_info resolution, shrink it to avoid the background.
        sx = (self.K[0, 2] * 2) / image_w
        sy = (self.K[1, 2] * 2) / image_h
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        x1, x2, y1, y2 = x1 + mx, x2 - mx, y1 + my, y2 - my

        u = self.K[0, 0] * pts[:, 0] / pts[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * pts[:, 1] / pts[:, 2] + self.K[1, 2]
        inside = pts[(u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)]
        if len(inside) < 15:
            return None
        p = np.median(inside, axis=0) @ R_mc.T + t_mc
        return float(p[0]), float(p[1])

    @staticmethod
    def to_robot(xy, pose):
        """Map point -> (forward, left) metres relative to the robot pose (x, y, yaw)."""
        dx, dy = xy[0] - pose[0], xy[1] - pose[1]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return c * dx + s * dy, -s * dx + c * dy

    def ahead(self, distance=1.0, pose=None):
        """Fallback: the point `distance` metres in front of the robot."""
        pose = pose or self.robot_pose()
        if pose is None:
            return None
        x, y, yaw = pose
        return x + distance * math.cos(yaw), y + distance * math.sin(yaw)

    # ============ placing things for the scene reports (vlm_watcher / qr_reader) ============

    def camera_snapshot(self):
        """Like snapshot() but works without a depth cloud (only the camera pose), so the
        ray fallbacks below still work. snapshot() itself is unchanged (obstacle_detector)."""
        if self.K is None:
            return None
        try:
            R_mc, t_mc = self.transform(self.global_frame, self.color_frame)
        except TransformException:
            return None
        return {'cloud': None, 'R_mc': R_mc, 't_mc': t_mc, 'pose': self.robot_pose()}

    def pixel_ray(self, u, v, image_w, image_h, snap):
        """Camera origin and unit direction (map frame) through pixel (u, v)."""
        u = u * (self.K[0, 2] * 2) / image_w
        v = v * (self.K[1, 2] * 2) / image_h
        d = np.array([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], 1.0])
        d = snap['R_mc'] @ d
        return snap['t_mc'], d / np.linalg.norm(d)

    def floor_hit(self, origin, d, max_range=4.0):
        """Where the ray meets the floor plane, or None."""
        if d[2] > -0.05:
            return None
        t = (self.floor_z - origin[2]) / d[2]
        if t <= 0 or t > max_range:
            return None
        p = origin + t * d
        return float(p[0]), float(p[1])

    def wall_hit(self, origin, d, max_range=4.0, step=0.04):
        """First occupied map cell along the ray's horizontal direction, or None."""
        g = self.grid
        h = np.array([d[0], d[1]])
        if g is None or np.linalg.norm(h) < 1e-3:
            return None
        h = h / np.linalg.norm(h)
        for r in np.arange(0.2, max_range, step):
            x, y = origin[0] + r * h[0], origin[1] + r * h[1]
            i, j = int((y - g['oy']) / g['res']), int((x - g['ox']) / g['res'])
            if not (0 <= i < g['h'] and 0 <= j < g['w']):
                return None
            if g['data'][i, j] >= 65:
                return float(x), float(y)
        return None

    def place(self, box, image_w, image_h, snap, on='wall', pull_m=0.25):
        """Map position of a box for the scene reports.
        Order: depth inside the box -> ray to the floor (on='floor') or to the first wall
        on the SLAM map (on='wall') -> None (never a guessed 'ahead' point).
        Returns {'xy', 'cell_xy', 'cam_xy', 'range_m', 'how'} or None.
        cell_xy = xy pulled pull_m back towards the camera: a sign or picture on a wall sits
        exactly on a cell border, the pull puts it in the room on the camera's side."""
        if snap is None or self.K is None:
            return None
        xy, how = None, None
        if snap.get('cloud') is not None:
            xy, how = self.locate_box(box, image_w, image_h, snap), 'depth'
        if xy is None:
            u = (box[0] + box[2]) / 2
            v = box[3] if on == 'floor' else (box[1] + box[3]) / 2
            origin, d = self.pixel_ray(u, v, image_w, image_h, snap)
            first, second = ((self.floor_hit, self.wall_hit) if on == 'floor'
                             else (self.wall_hit, self.floor_hit))
            xy, how = first(origin, d), 'ray_' + first.__name__.split('_')[0]
            if xy is None:
                xy, how = second(origin, d), 'ray_' + second.__name__.split('_')[0]
        if xy is None:
            return None
        cam = snap['t_mc']
        dx, dy = xy[0] - cam[0], xy[1] - cam[1]
        rng = math.hypot(dx, dy)
        k = max(0.0, rng - pull_m) / rng if rng > 1e-6 else 1.0
        return {'xy': [round(xy[0], 3), round(xy[1], 3)],
                'cell_xy': [round(cam[0] + k * dx, 3), round(cam[1] + k * dy, 3)],
                'cam_xy': [round(float(cam[0]), 3), round(float(cam[1]), 3)],
                'range_m': round(rng, 2), 'how': how}

    # ============ on-demand depth (cloud_on_demand=True) ============

    def attach_cloud(self, snap, cloud):
        """Add a cloud to a camera_snapshot() so locate_box() can use depth."""
        if snap is None or cloud is None:
            return snap
        try:
            R_cd, t_cd = self.transform(self.color_frame, cloud.header.frame_id)
        except TransformException:
            return snap
        snap.update(cloud=cloud, R_cd=R_cd, t_cd=t_cd)
        snap.pop('xyz', None)
        return snap

    def request_cloud(self, callback, timeout_s=1.5):
        """Subscribe until one cloud arrives (or timeout), then unsubscribe and call
        callback(cloud or None). One request at a time; returns False if one is running.
        Safe to call from a ROS callback (does not block)."""
        if self._req is not None:
            return False
        state = {'done': False}

        def finish(cloud):
            if state['done']:
                return
            state['done'] = True
            req, self._req = self._req, None
            if req is not None:
                try:
                    self.node.destroy_subscription(req['sub'])
                except Exception:  # noqa: BLE001
                    pass
                req['timer'].cancel()
                try:
                    self.node.destroy_timer(req['timer'])
                except Exception:  # noqa: BLE001
                    pass
            callback(cloud)

        sub = self.node.create_subscription(PointCloud2, self.cloud_topic,
                                            lambda m: finish(m), qos_profile_sensor_data)
        timer = self.node.create_timer(timeout_s, lambda: finish(None))
        self._req = {'sub': sub, 'timer': timer}
        return True

    def wait_cloud(self, timeout_s=1.5):
        """Blocking version for worker threads (NOT for ROS callbacks)."""
        import threading
        ev, box = threading.Event(), {}

        def cb(cloud):
            box['cloud'] = cloud
            ev.set()
        t0 = time.monotonic()
        while not self.request_cloud(cb, timeout_s):     # another request is running
            if time.monotonic() - t0 > timeout_s:
                return None
            time.sleep(0.05)
        ev.wait(timeout_s + 1.0)
        return box.get('cloud')
