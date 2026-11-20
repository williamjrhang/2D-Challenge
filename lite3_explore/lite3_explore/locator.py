"""Find where something seen in the colour image is on the map (Lab 7, Exercise III-B).

    box in colour image  +  depth point cloud  ->  3D point  ->  TF  ->  map (x, y)

The depth camera and the colour camera are separate frames on this robot, so the
depth points are moved into the colour camera frame with TF and projected into the
colour image; the points that land inside the box give the object's position.
"""

import math

import numpy as np
from rclpy.qos import qos_profile_sensor_data
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
                 info_topic='/camera/color/camera_info', cloud_topic='/camera/depth/points'):
        self.tf = tf_buffer
        self.global_frame = global_frame
        self.robot_frame = robot_frame
        self.K = None
        self.color_frame = None
        self.cloud = None
        node.create_subscription(CameraInfo, info_topic, self.on_info, 10)
        node.create_subscription(PointCloud2, cloud_topic, self.on_cloud, qos_profile_sensor_data)

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
