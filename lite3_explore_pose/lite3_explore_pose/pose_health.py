"""Check localization timestamps independently of navigation goals."""

import math

from rclpy.time import Time
from tf2_ros import TransformException


class PoseHealth:
    def __init__(self, buffer, map_frame, odom_frame, robot_frame,
                 odom_max_age, slam_max_age, settle_time):
        self.buffer = buffer
        self.map_frame = map_frame
        self.odom_frame = odom_frame
        self.robot_frame = robot_frame
        self.odom_max_age = odom_max_age
        self.slam_max_age = slam_max_age
        self.settle_time = settle_time
        self.positions = {}
        self.jump = None
        self.reset()

    def reset(self):
        self.healthy_since = None
        self.first_odom_stamp = None
        self.odom_stamp = None
        self.pose_stamp = None

    def sample(self, now):
        if self.jump is not None:
            return None, self.jump
        # Check the two edges separately so a fresh SLAM transform cannot mask old odometry.
        # Use the coherent map-frame transform for decisions, as Nav2 does.
        for parent, child, max_age, allow_static in (
            (self.odom_frame, self.robot_frame, self.odom_max_age, False),
            (self.map_frame, self.odom_frame, self.slam_max_age, True),
            (self.map_frame, self.robot_frame, self.slam_max_age, False),
        ):
            edge = f'{parent} -> {child}'
            try:
                transform = self.buffer.lookup_transform(parent, child, Time())
            except TransformException:
                self.reset()
                return None, f'{edge} unavailable'
            stamp = transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec
            age = now - stamp / 1e9
            if stamp == 0 and allow_static:
                continue
            if stamp == 0 or age > max_age or age < -max_age:
                self.reset()
                return None, f'{edge} timestamp age {age:.2f} s (limit {max_age:.2f} s)'
            if parent == self.odom_frame:
                odom_stamp = stamp
                self.check_position(edge, transform, stamp, slack=0.5)
            elif child == self.robot_frame:
                self.check_position(edge, transform, stamp, slack=3.0)
            if self.jump is not None:
                self.reset()
                return None, self.jump

        if self.odom_stamp is not None and odom_stamp < self.odom_stamp:
            self.reset()
        self.odom_stamp = odom_stamp
        if self.healthy_since is None or now < self.healthy_since:
            self.healthy_since = now
            self.first_odom_stamp = odom_stamp
        if now - self.healthy_since < self.settle_time or odom_stamp == self.first_odom_stamp:
            return None, 'waiting for stable TF and new odometry'
        self.pose_stamp = stamp
        return transform, None

    def check_position(self, edge, transform, stamp, slack):
        xyz = transform.transform.translation
        xy = (xyz.x, xyz.y)
        if not all(math.isfinite(value) for value in xy):
            self.jump = f'{edge} returned a non-finite position; restart the robot bringup and explorer'
            return
        last = self.positions.get(edge)
        if last is not None:
            old_stamp, old_xy = last
            if stamp <= old_stamp:
                return
            elapsed = (stamp - old_stamp) / 1e9
            distance = math.dist(xy, old_xy)
            # Allow sampling gaps and up to 1 m/s, plus xy correction in the map frame.
            if distance > slack + elapsed:
                self.jump = (f'{edge} jumped {distance:.2f} m in {elapsed:.2f} s; '
                             'restart the robot bringup and explorer from the start')
        self.positions[edge] = (stamp, xy)
