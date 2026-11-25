"""Check localization timestamps independently of navigation goals."""

import math

from rclpy.time import Time
from tf2_ros import TransformException


class PoseHealth:
    """TF health for the explorer.

    Position jumps: a jump latches (robot_pose() -> None). 25/11 the latch was permanent,
    so one FAST-LIO hiccup ('odom -> base_link jumped 1.30 m in 0.70 s') left the explorer
    in WAIT_POSE for the rest of the run. Now (jump_recover_s > 0) the latch clears once
    every edge has been fresh and jump-free for jump_recover_s. Never cleared:
    non-finite positions and jumps larger than hard_jump_m (real breakage, restart).
    jump_recover_s = 0 keeps the old permanent latch.
    """

    def __init__(self, buffer, map_frame, odom_frame, robot_frame,
                 odom_max_age, slam_max_age, settle_time,
                 jump_recover_s=0.0, hard_jump_m=math.inf):
        self.buffer = buffer
        self.map_frame = map_frame
        self.odom_frame = odom_frame
        self.robot_frame = robot_frame
        self.odom_max_age = odom_max_age
        self.slam_max_age = slam_max_age
        self.settle_time = settle_time
        self.jump_recover_s = float(jump_recover_s)
        self.hard_jump_m = float(hard_jump_m)
        self.positions = {}
        self.jump = None            # message while latched
        self.jump_hard = False      # never recovers
        self.jump_since = None      # time of the last jump (recovery clock)
        self.recovered = None       # message once after a recovery (for the node's log)
        self.reset()

    def reset(self):
        self.healthy_since = None
        self.first_odom_stamp = None
        self.odom_stamp = None
        self.pose_stamp = None

    def recoverable(self):
        return self.jump is not None and not self.jump_hard and self.jump_recover_s > 0

    def sample(self, now):
        if self.jump is not None and not self.recoverable():
            return None, self.jump
        latched = self.jump is not None
        new_jump = False
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
                if latched:
                    self.jump_since = now          # need a full stable period again
                    return None, self.latched_reason(now)
                return None, f'{edge} unavailable'
            stamp = transform.header.stamp.sec * 1_000_000_000 + transform.header.stamp.nanosec
            age = now - stamp / 1e9
            if stamp == 0 and allow_static:
                continue
            if stamp == 0 or age > max_age or age < -max_age:
                self.reset()
                if latched:
                    self.jump_since = now
                    return None, self.latched_reason(now)
                return None, f'{edge} timestamp age {age:.2f} s (limit {max_age:.2f} s)'
            if parent == self.odom_frame:
                odom_stamp = stamp
                new_jump |= self.check_position(edge, transform, stamp, slack=0.5, now=now)
            elif child == self.robot_frame:
                new_jump |= self.check_position(edge, transform, stamp, slack=3.0, now=now)
            if self.jump is not None and not self.recoverable():
                self.reset()
                return None, self.jump

        if self.jump is not None:
            self.reset()
            if new_jump or not latched:
                return None, self.jump
            if now - self.jump_since < self.jump_recover_s:
                return None, self.latched_reason(now)
            self.recovered = (f'position jump latch cleared after {now - self.jump_since:.1f} s '
                              f'of stable TF (was: {self.jump})')
            self.jump = None
            self.jump_since = None
            return None, 'waiting for stable TF and new odometry (after a position jump)'

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

    def latched_reason(self, now):
        left = self.jump_recover_s - (now - (self.jump_since or now))
        return f'{self.jump} (auto-recovery after {max(0.0, left):.1f} s more of stable TF)'

    def check_position(self, edge, transform, stamp, slack, now=0.0):
        """Returns True when this sample is a (new) jump."""
        xyz = transform.transform.translation
        xy = (xyz.x, xyz.y)
        if not all(math.isfinite(value) for value in xy):
            self.jump = f'{edge} returned a non-finite position; restart the robot bringup and explorer'
            self.jump_hard = True
            return True
        jumped = False
        last = self.positions.get(edge)
        if last is not None:
            old_stamp, old_xy = last
            if stamp <= old_stamp:
                return False
            elapsed = (stamp - old_stamp) / 1e9
            distance = math.dist(xy, old_xy)
            # Allow sampling gaps and up to 1 m/s, plus xy correction in the map frame.
            if distance > slack + elapsed:
                jumped = True
                hard = distance > self.hard_jump_m
                self.jump_hard = self.jump_hard or hard
                self.jump_since = now
                self.jump = (f'{edge} jumped {distance:.2f} m in {elapsed:.2f} s; '
                             + ('over the hard limit: restart the robot bringup and explorer'
                                if hard or self.jump_recover_s <= 0
                                else 'holding still until TF is stable again'))
        self.positions[edge] = (stamp, xy)
        return jumped
