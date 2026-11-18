#!/usr/bin/env python3
"""Frontier exploration node for the Lite3.

Loop:  initial 360 spin -> pick best frontier -> Nav2 goes there -> repeat.
Safety: goal time limit, progress check, blacklist of failed frontiers,
        back-up recovery, and stop when no frontiers are left.

Needs running:  SLAM (publishes /map and TF map -> base_link) and Nav2.
The robot must be in Auto mode (lite3_auto) to move.

Topics
  in   /map                    nav_msgs/OccupancyGrid   from SLAM
  in   /exploration/enable     std_msgs/Bool            mission controller pause/resume
  in   /exploration/bias       geometry_msgs/PointStamped  prefer frontiers near here
                                                          (send frame_id "" to clear)
  out  /exploration/status     std_msgs/String          current state
  out  /exploration/done       std_msgs/Bool            True when nothing left to explore
  out  /exploration/frontiers  visualization_msgs/MarkerArray  for RViz
  out  /cmd_vel                geometry_msgs/Twist      only during spin / back-up
"""

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:  # Lite3 custom message for /simple_cmd (gait switching)
    from transfer_interfaces.msg import MotionSimpleCMD
except ImportError:
    MotionSimpleCMD = None

from lite3_explore.frontiers import (
    MapInfo, find_frontiers, score_frontiers, still_frontier,
)


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        # ---------------- parameters (see config/explorer.yaml) ----------------
        p = self.declare_parameter
        p('dry_run', True)               # True: show frontiers in RViz, never move
        p('global_frame', 'map')
        p('robot_frame', 'base_link')
        p('cmd_vel_topic', '/nav_cmd_vel')  # twist_mux input, never /cmd_vel
        p('tick_hz', 5.0)
        p('gait_code', 0)                # gait to use for the whole run (0 = don't switch)

        p('initial_spin', True)          # Nav2 look-around spin at start (collision-checked)
        p('spin_speed', 0.5)             # rad/s
        p('spin_turns', 1.0)             # 1.0 = 360 deg

        p('min_frontier_size_cells', 5) #ignore tiny frontier clusters
        p('free_max', 25)
        p('occupied_min', 65)
        p('goal_clearance_m', 0.25)  

        #control which frontier is preferred
        p('w_size', 1.0)
        p('w_dist', 1.0)
        p('w_bias', 3.0)

        p('time_limit_base_s', 20.0)     # goal time limit = base + per_m * distance
        p('time_limit_per_m_s', 8.0)
        #how long to wait before checking progress
        p('progress_window_s', 10.0)     # must get min_progress_m closer within this
        p('min_progress_m', 0.3)
        p('recheck_period_s', 2.0)       # cancel if the goal stopped being a frontier

        p('blacklist_radius_m', 0.6)
        p('backup_speed', 0.15)          # m/s
        p('backup_time_s', 2.0)
        p('max_consecutive_failures', 3)
        p('done_after_empty_checks', 10) # no frontiers this many times in a row -> done

        g = lambda name: self.get_parameter(name).value
        self.P = {name: g(name) for name in [
            'dry_run', 'global_frame', 'robot_frame', 'tick_hz', 'initial_spin',
            'spin_speed', 'spin_turns', 'min_frontier_size_cells', 'free_max',
            'occupied_min', 'goal_clearance_m', 'w_size', 'w_dist', 'w_bias',
            'time_limit_base_s', 'time_limit_per_m_s', 'progress_window_s',
            'min_progress_m', 'recheck_period_s', 'blacklist_radius_m',
            'backup_speed', 'backup_time_s', 'max_consecutive_failures',
            'done_after_empty_checks', 'gait_code',
        ]}

        # ---------------- ROS interfaces ----------------
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.create_subscription(Bool, '/exploration/enable', self.on_enable, 10)
        self.create_subscription(PointStamped, '/exploration/bias', self.on_bias, 10)

        self.cmd_pub = self.create_publisher(Twist, g('cmd_vel_topic'), 10)
        self.status_pub = self.create_publisher(String, '/exploration/status', 10)
        self.done_pub = self.create_publisher(Bool, '/exploration/done', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/exploration/frontiers', 10)
        self.gait_pub = None
        if MotionSimpleCMD is not None:
            self.gait_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Nav2 Simple Commander (its own node, used like in the labs)
        self.nav = BasicNavigator(node_name='explorer_navigator')
        if not self.P['dry_run']:
            self.get_logger().info('Waiting for Nav2 (bt_navigator) to be active...')
            # Only waits for bt_navigator, so it works with SLAM instead of AMCL.
            self.nav._waitForNodeToActivate('bt_navigator')
            self.get_logger().info('Nav2 is active.')

        # ---------------- state ----------------
        self.grid = None
        self.info = None
        self.enabled = True
        self.bias_xy = None
        self.blacklist = []
        self.state = 'WAIT_MAP'
        self.empty_checks = 0
        self.failures_in_a_row = 0
        self.last_good_xy = None

        self.goal_xy = None
        self.goal_is_frontier = True
        self.goal_start = None
        self.goal_time_limit = 0.0
        self.window_start = None
        self.window_remaining = None
        self.last_recheck = None
        self.state_end = None

        self.create_timer(1.0 / self.P['tick_hz'], self.tick)
        mode = 'DRY RUN (will not move)' if self.P['dry_run'] else 'LIVE'
        self.get_logger().info(f'Frontier explorer started: {mode}')

    # ================= callbacks =================

    def on_map(self, msg):
        self.info = MapInfo(
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            width=msg.info.width,
            height=msg.info.height,
        )
        self.grid = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)

    def on_enable(self, msg):
        if self.enabled != msg.data:
            self.get_logger().info(f'Exploration {"enabled" if msg.data else "paused"}')
        self.enabled = msg.data
        if msg.data and self.state == 'DONE':
            self.empty_checks = 0
            self.set_state('SELECT')

    def on_bias(self, msg):
        if msg.header.frame_id == '':
            self.bias_xy = None
            self.get_logger().info('Bias cleared')
        else:
            self.bias_xy = (msg.point.x, msg.point.y)
            self.get_logger().info(f'Bias set to ({msg.point.x:.2f}, {msg.point.y:.2f})')

    # ================= helpers =================

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def robot_pose(self):
        """(x, y, yaw) of the robot in the map frame, or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.P['global_frame'], self.P['robot_frame'], Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def publish_cmd(self, linear=0.0, angular=0.0):
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)

    def set_state(self, state):
        if state != self.state:
            self.get_logger().info(f'State: {self.state} -> {state}')
        self.state = state
        self.status_pub.publish(String(data=state))

    def blacklist_goal(self, why):
        if self.goal_xy is not None and self.goal_is_frontier:
            self.blacklist.append(self.goal_xy)
            self.get_logger().warn(
                f'Blacklisted ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}): {why}')

    def send_goal(self, xy, robot_xy, is_frontier=True):
        yaw = math.atan2(xy[1] - robot_xy[1], xy[0] - robot_xy[0])
        goal = PoseStamped()
        goal.header.frame_id = self.P['global_frame']
        goal.header.stamp = self.nav.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = float(xy[0]), float(xy[1])
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        goal.pose.orientation.x, goal.pose.orientation.y = qx, qy
        goal.pose.orientation.z, goal.pose.orientation.w = qz, qw

        self.goal_xy = xy
        self.goal_is_frontier = is_frontier
        distance = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
        self.goal_time_limit = self.P['time_limit_base_s'] + self.P['time_limit_per_m_s'] * distance

        accepted = self.nav.goToPose(goal)
        if accepted is False:
            self.blacklist_goal('Nav2 rejected the goal')
            return False

        now = self.now_s()
        self.goal_start = now
        self.window_start = now
        self.window_remaining = None
        self.last_recheck = now
        self.get_logger().info(
            f'Goal ({xy[0]:.2f}, {xy[1]:.2f}), {distance:.1f} m away, '
            f'limit {self.goal_time_limit:.0f} s')
        return True

    def publish_markers(self, ranked):
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for i, f in enumerate(ranked[:20]):
            m = Marker()
            m.header.frame_id = self.P['global_frame']
            m.ns, m.id, m.type, m.action = 'frontiers', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(f.goal_xy[0]), float(f.goal_xy[1])
            m.pose.orientation.w = 1.0
            size = 0.3 if i == 0 else 0.18
            m.scale.x = m.scale.y = m.scale.z = size
            # Best frontier orange, others blue
            m.color.r, m.color.g, m.color.b = (1.0, 0.55, 0.1) if i == 0 else (0.2, 0.4, 0.9)
            m.color.a = 0.9
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    # ================= main loop =================

    def tick(self):
        pose = self.robot_pose()

        if not self.enabled:
            if self.state in ('NAVIGATE', 'SPIN') and not self.P['dry_run']:
                self.nav.cancelTask()
            if self.state in ('SPIN', 'RECOVER'):
                self.publish_cmd()
            if self.state != 'DONE':
                self.set_state('PAUSED')
            return
        if self.state == 'PAUSED':
            self.set_state('SELECT')

        handler = getattr(self, f'do_{self.state.lower()}')
        handler(pose)

    def do_wait_map(self, pose):
        if self.grid is None or pose is None:
            return
        if self.P['gait_code'] and not self.P['dry_run']:
            if self.gait_pub is None:
                self.get_logger().warn('transfer_interfaces not found: cannot set gait.')
            else:
                self.send_gait()
                self.gait_resent = False
                self.state_end = self.now_s() + 2.0   # stand still while the gait changes
                self.set_state('GAIT')
                return
        self.start_exploring()

    def send_gait(self):
        code = int(self.P['gait_code'])
        self.gait_pub.publish(MotionSimpleCMD(cmd_code=code, type=0))
        self.get_logger().info(f'Gait command sent: {hex(code)}')

    def do_gait(self, pose):
        self.publish_cmd()
        remaining = self.state_end - self.now_s()
        if remaining < 1.5 and not self.gait_resent:
            self.send_gait()      # send twice in case the first one was missed
            self.gait_resent = True
        if remaining <= 0:
            self.start_exploring()

    def start_exploring(self):
        # Opening look-around with Nav2's spin behaviour: it checks the robot footprint
        # against the costmap and STOPS instead of hitting a wall (the old cmd_vel spin
        # did not). The lidar sees 360 deg anyway; the spin is for the forward camera.
        if self.P['initial_spin'] and not self.P['dry_run']:
            angle = 2.0 * math.pi * self.P['spin_turns']
            limit = angle / max(self.P['spin_speed'], 1e-3) + 10.0
            if self.nav.spin(spin_dist=angle, time_allowance=int(limit)):
                self.state_end = self.now_s() + limit + 5.0
                self.get_logger().info(f'Look-around spin {math.degrees(angle):.0f} deg (Nav2, collision-checked)')
                self.set_state('SPIN')
                return
            self.get_logger().warn('Nav2 refused the spin, skipping it.')
        self.set_state('SELECT')

    def do_spin(self, pose):
        if self.nav.isTaskComplete():
            result = self.nav.getResult()
            if result != TaskResult.SUCCEEDED:
                self.get_logger().warn(f'Spin stopped early ({result}): too close to a wall. Exploring anyway.')
            self.set_state('SELECT')
        elif self.now_s() > self.state_end:
            self.nav.cancelTask()
            self.set_state('SELECT')

    #chooses the next destination
    def do_select(self, pose):
        if self.grid is None or pose is None:
            return
        frontiers = find_frontiers(
            self.grid, self.info,
            min_size_cells=self.P['min_frontier_size_cells'],
            free_max=self.P['free_max'],
            occupied_min=self.P['occupied_min'],
            clearance_m=self.P['goal_clearance_m'],
        )
        ranked = score_frontiers(
            frontiers, robot_xy=pose[:2],
            w_size=self.P['w_size'], w_dist=self.P['w_dist'], w_bias=self.P['w_bias'],
            bias_xy=self.bias_xy, blacklist=self.blacklist,
            blacklist_radius_m=self.P['blacklist_radius_m'],
        )
        self.publish_markers(ranked)

        if not ranked:
            self.empty_checks += 1
            if self.P['dry_run']:
                return  # keep watching the map while you drive by hand
            if self.empty_checks >= self.P['done_after_empty_checks']:
                self.get_logger().info('No reachable frontiers left: exploration done.')
                self.done_pub.publish(Bool(data=True))
                self.set_state('DONE')
            return
        self.empty_checks = 0

        if self.P['dry_run']:
            return  # only visualise

        best = ranked[0]
        if self.send_goal(best.goal_xy, pose[:2], is_frontier=True):
            self.set_state('NAVIGATE')

    def do_navigate(self, pose):
        now = self.now_s()

        # 1. Finished (success or failure)?
        if self.nav.isTaskComplete():
            result = self.nav.getResult()
            # Nav2 can report success without moving if the map->odom TF is too old.
            if result == TaskResult.SUCCEEDED and pose is not None:
                miss = math.hypot(self.goal_xy[0] - pose[0], self.goal_xy[1] - pose[1])
                if miss > 0.75:
                    self.get_logger().warn(
                        f'Nav2 said goal reached but robot is {miss:.1f} m away '
                        '(check TF / transform_tolerance).')
                    result = TaskResult.FAILED
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info('Goal reached.')
                self.failures_in_a_row = 0
                self.last_good_xy = self.goal_xy
                # Don't come straight back if this spot is still a frontier (e.g. glass).
                if self.goal_is_frontier:
                    self.blacklist.append(self.goal_xy)
                self.set_state('SELECT')
            else:
                self.blacklist_goal(f'Nav2 result {result}')
                self.start_recovery()
            return

        # 2. Over the time limit?
        if now - self.goal_start > self.goal_time_limit:
            self.nav.cancelTask()
            self.blacklist_goal('time limit')
            self.start_recovery()
            return

        # 3. Making progress?
        feedback = self.nav.getFeedback()
        remaining = getattr(feedback, 'distance_remaining', None) if feedback else None
        if remaining is not None and remaining > 0.0:
            if self.window_remaining is None:
                self.window_remaining = remaining
                self.window_start = now
            elif now - self.window_start > self.P['progress_window_s']:
                if self.window_remaining - remaining < self.P['min_progress_m']:
                    self.nav.cancelTask()
                    self.blacklist_goal(f'no progress in {self.P["progress_window_s"]:.0f} s')
                    self.start_recovery()
                    return
                self.window_remaining = remaining
                self.window_start = now

        # 4. Goal already explored on the way? Then pick a new one (no blacklist).
        if self.goal_is_frontier and now - self.last_recheck > self.P['recheck_period_s']:
            self.last_recheck = now
            if self.grid is not None and not still_frontier(
                    self.grid, self.info, self.goal_xy, radius_m=0.4,
                    free_max=self.P['free_max']):
                self.get_logger().info('Goal area already mapped, choosing a new frontier.')
                self.nav.cancelTask()
                self.set_state('SELECT')

    def start_recovery(self):
        self.failures_in_a_row += 1
        self.state_end = self.now_s() + self.P['backup_time_s']
        self.set_state('RECOVER')

    def do_recover(self, pose):
        if self.now_s() < self.state_end:
            self.publish_cmd(linear=-abs(self.P['backup_speed']))
            return
        self.publish_cmd()

        too_many = self.failures_in_a_row >= self.P['max_consecutive_failures']
        if too_many and self.last_good_xy is not None and pose is not None:
            self.get_logger().warn('Several failures in a row: returning to last good spot.')
            self.failures_in_a_row = 0
            if self.send_goal(self.last_good_xy, pose[:2], is_frontier=False):
                self.set_state('NAVIGATE')
                return
        self.set_state('SELECT')

    def do_done(self, pose):
        pass

    def do_paused(self, pose):
        pass

    def stop(self):
        try:
            if self.state in ('NAVIGATE', 'SPIN') and not self.P['dry_run']:
                self.nav.cancelTask()
        finally:
            self.publish_cmd()


def main(args=None):
    # Handle Ctrl+C ourselves so the stop command is sent before shutdown.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = FrontierExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.nav.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
