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
  [COVER] (only with coverage_enabled)
  out  /exploration/coverage   nav_msgs/OccupancyGrid   camera: seen=0, unseen free=100, else -1
  out  /exploration/trail      nav_msgs/Path            stored pose history
  out  /exploration/coverage_pct  std_msgs/Float32      camera coverage inside bounds, %
  [HOLD] (only with obstacle_hold_s > 0)
  in   /perception/obstacles   std_msgs/String          obstacle_detector detections (JSON)
"""

import json  # [HOLD]
import math
import os  # [COVER]

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path  # [COVER]
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Bool, ColorRGBA, String
from std_msgs.msg import Float32  # [COVER]
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

try:  # Lite3 custom message for /simple_cmd (gait switching)
    from transfer_interfaces.msg import MotionSimpleCMD
except ImportError:
    MotionSimpleCMD = None

from lite3_explore.gait import GAIT_COMMAND_QOS, send_gait
from lite3_explore.frontiers import (
    MapInfo, find_frontiers, score_frontiers, still_frontier,
)
# [COVER] camera coverage (pose history -> seen mask -> COVER goals)
from lite3_explore.coverage import (
    PoseHistory, SeenTracker, bounds_mask, coverage_fraction, default_coverage_path,
    find_coverage_goals, load, newest_coverage_file, save, visited_mask,
)

# [ROOM] arena cells, walls and rooms from the live map (cells.py)
from lite3_explore.cells import CellGrid, SIDES, wrap
from lite3_explore.mission_config import Arena, load_mission

# [HOLD] detections that may be the reason Nav2 is stuck (gait-change obstacles)
HOLD_KINDS = ('stairs', 'rubble', 'curtain', 'potholes')


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
        p('gait_via_manual', False)      # true = fallback: manual -> gait -> auto

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
        # Arena box in the map frame [xmin, ymin, xmax, ymax]; frontiers outside are ignored
        # (open walls / the final cell's missing wall let the lidar map the lab). [] = off.
        p('bounds', [0.0])
        # Large-map guard: when bounds is off, frontiers AND camera coverage are limited to a
        # box +/- guard_radius_m around the start pose (map origin), so the lab outside the
        # arena and ghost walls after a SLAM jump don't keep COVER going forever. 0 = off.
        p('guard_radius_m', 6.0)
        # Look-around after reaching a goal, so the camera sees side walls (QR codes are on
        # the left/right walls of rooms). 0 = off. Only if this far from the last one.
        p('goal_spin_deg', 0.0)
        p('goal_spin_min_gap_m', 2.0)
        # [COVER] camera coverage: when no frontiers are left, drive where the forward
        # camera has not looked yet (coverage.py). Defaults = off = old behaviour.
        p('coverage_enabled', False)
        p('coverage_target', 0.95)        # DONE when this share of free cells was seen
        p('camera_fov_deg', 70.0)
        p('camera_range_m', 2.5)
        p('robot_radius_m', 0.35)         # cells the robot stood on count as seen
        p('coverage_min_cluster_cells', 10)
        p('coverage_save_period_s', 10.0)  # 0 = only save on shutdown
        p('coverage_file', '')            # '' = ~/challenge_ws/coverage/coverage_<time>.npz
        p('resume_coverage', False)       # load coverage_file (or the newest file) at start
        # [HOLD] stand still so obstacle_detector can confirm an obstacle. 0 = off.
        p('goal_dwell_s', 0.0)            # after reaching a frontier / coverage goal
        p('obstacle_hold_s', 0.0)         # instead of blacklisting when an obstacle is ahead
        p('obstacle_hold_m', 1.5)         # detection this close ahead counts
        p('obstacle_recent_s', 2.0)       # ... if it arrived this recently
        # [ROOM] booklet: QR codes are only on the left/right walls of rooms. A room = a free
        # cell with three walls on the live map. room_check: stand in its doorway and face
        # into the room, then each side wall, so QR / wall pins / the VLM see them up close.
        p('mission_file', '')             # arena grid (cell size, start cell) for cells/rooms
        p('room_check', False)
        p('room_check_radius_m', 3.0)     # during frontier exploration: only rooms this close
        p('room_dwell_s', 2.5)            # still time per view (QR reader 3 Hz, VLM 1 photo)
        p('room_max_attempts', 2)
        p('cell_overlay', True)           # RViz /exploration/cells: cells, walls, rooms

        g = lambda name: self.get_parameter(name).value
        self.P = {name: g(name) for name in [
            'dry_run', 'global_frame', 'robot_frame', 'tick_hz', 'initial_spin',
            'spin_speed', 'spin_turns', 'min_frontier_size_cells', 'free_max',
            'occupied_min', 'goal_clearance_m', 'w_size', 'w_dist', 'w_bias',
            'time_limit_base_s', 'time_limit_per_m_s', 'progress_window_s',
            'min_progress_m', 'recheck_period_s', 'blacklist_radius_m',
            'backup_speed', 'backup_time_s', 'max_consecutive_failures',
            'done_after_empty_checks', 'gait_code', 'gait_via_manual',
            'bounds', 'guard_radius_m', 'goal_spin_deg', 'goal_spin_min_gap_m',
        ]}
        # [COVER] [HOLD] new params
        self.P.update({name: g(name) for name in [
            'coverage_enabled', 'coverage_target', 'camera_fov_deg', 'camera_range_m',
            'robot_radius_m', 'coverage_min_cluster_cells', 'coverage_save_period_s',
            'coverage_file', 'resume_coverage',
            'goal_dwell_s', 'obstacle_hold_s', 'obstacle_hold_m', 'obstacle_recent_s',
        ]})
        # [ROOM]
        self.P.update({name: g(name) for name in [
            'room_check', 'room_check_radius_m', 'room_dwell_s', 'room_max_attempts',
            'cell_overlay',
        ]})
        try:
            arena = Arena(load_mission(g('mission_file')))
        except Exception as e:  # noqa: BLE001  (fall back to the booklet grid)
            self.get_logger().warn(f'mission_file not readable ({e}); booklet grid 1.02 m')
            arena = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0.0, 0.0],
                                     'start_heading_deg': 0.0, 'final_cell_centre': None}})
        self.cells = CellGrid(arena)
        self.cell_info = {}               # cell -> CellInfo (last analysis)
        self.cells_seq = -1               # map_seq of that analysis
        self.visited_cells = set()
        self.room_status = {}             # cell -> {'attempts', 'done', 'views_done'}
        self.room = None                  # Room being checked
        self.room_idx = 0
        self.room_phase = None
        self.room_until = None
        b = [float(v) for v in (self.P['bounds'] or [])]
        self.bounds = b if len(b) == 4 else None     # [0.0] / [] in yaml = off
        guard = float(self.P['guard_radius_m'])
        if self.bounds is None and guard > 0:
            self.bounds = [-guard, -guard, guard, guard]  # large-map guard around the start pose
            self.get_logger().info(f'bounds off: large-map guard +/- {guard:.1f} m around the start')
        self.last_spin_xy = None

        # [COVER] pose history (map frame), seen mask, save file
        self.poses = PoseHistory(min_move_m=0.2, min_turn_deg=15.0)
        self.seen_tracker = SeenTracker(self.P['camera_fov_deg'], self.P['camera_range_m'],
                                        self.P['occupied_min'])
        self.map_seq = 0
        self.seen_key = None
        self.seen = None
        self.n_saved = 0
        self.coverage_path = (os.path.expanduser(self.P['coverage_file'])
                              if self.P['coverage_file'] else default_coverage_path())
        if self.P['coverage_enabled']:
            if self.bounds is None:
                self.get_logger().warn(
                    'coverage_enabled but bounds is off: camera coverage counts EVERY free '
                    'map cell (also the lab outside the arena). Set bounds in explorer.yaml.')
            if self.P['resume_coverage']:
                self.resume_coverage()

        # ---------------- ROS interfaces ----------------
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.create_subscription(Bool, '/exploration/enable', self.on_enable, 10)
        self.create_subscription(PointStamped, '/exploration/bias', self.on_bias, 10)
        # [HOLD] obstacle_detector's per-frame detections
        if self.P['obstacle_hold_s'] > 0:
            self.create_subscription(String, '/perception/obstacles', self.on_obstacles, 10)

        self.cmd_pub = self.create_publisher(Twist, g('cmd_vel_topic'), 10)
        self.status_pub = self.create_publisher(String, '/exploration/status', 10)
        self.done_pub = self.create_publisher(Bool, '/exploration/done', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/exploration/frontiers', 10)
        # [ROOM] RViz overlay + VLM trigger
        self.cells_pub = self.create_publisher(MarkerArray, '/exploration/cells', 10)
        self.look_pub = self.create_publisher(Bool, '/perception/look', 10)
        self.last_cells_pub = 0.0
        # [COVER] RViz: camera coverage grid (latched), pose trail, coverage %
        if self.P['coverage_enabled']:
            self.coverage_pub = self.create_publisher(OccupancyGrid, '/exploration/coverage', map_qos)
            self.trail_pub = self.create_publisher(Path, '/exploration/trail', 10)
            self.coverage_pct_pub = self.create_publisher(Float32, '/exploration/coverage_pct', 10)
        self.gait_pub = None
        if MotionSimpleCMD is not None:
            self.gait_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', GAIT_COMMAND_QOS)

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
        # [COVER] [HOLD] goal kind and per-goal flags (goal_is_frontier is kept as it was)
        self.goal_kind = 'frontier'       # 'frontier' | 'coverage' | 'return'
        self.goal_cells_xy = []           # coverage goal: world xy of its unseen cells
        self.goal_dwell_done = False      # arrival HOLD already done for this goal
        self.goal_hold_used = False       # obstacle HOLD already used for this goal
        self.hold_end = None
        self.hold_then = None             # ('navigate',) or ('recover', why)
        self.last_obstacle = None         # (time, kind, forward_m)
        self.covering = False             # no frontiers left, working on camera coverage
        self.cover_empty_checks = 0
        self.n_cover_markers = 0
        self.cover_markers = None
        self.last_coverage_publish = None
        self.last_coverage_save = None

        self.create_timer(1.0 / self.P['tick_hz'], self.tick)
        mode = 'DRY RUN (will not move)' if self.P['dry_run'] else 'LIVE'
        self.get_logger().info(f'Frontier explorer started: {mode}')
        self.get_logger().info(
            f'Frontier ranking: {self.P["w_size"]} × length - {self.P["w_dist"]} × distance '
            f'+ {self.P["w_bias"]} × bias, bounds={self.bounds}, '
            f'clearance={self.P["goal_clearance_m"]:.2f} m')

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
        self.map_seq += 1  # [COVER] new map -> recompute the cached seen mask

    def on_enable(self, msg):
        if self.enabled != msg.data:
            self.get_logger().info(f'Exploration {"enabled" if msg.data else "paused"}')
        self.enabled = msg.data
        if msg.data and self.state == 'DONE':
            self.empty_checks = 0
            self.covering = False  # [COVER] new area may be open: frontiers first again
            self.cover_empty_checks = 0
            self.set_state('SELECT')

    # [HOLD] remember the last detection that could be blocking the robot's path
    def on_obstacles(self, msg):
        try:
            dets = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(dets, list):
            return
        for d in dets:
            if not isinstance(d, dict) or d.get('kind') not in HOLD_KINDS:
                continue
            fwd, left = d.get('forward_m'), d.get('left_m')
            ahead = (fwd is not None and left is not None
                     and 0.0 < fwd <= self.P['obstacle_hold_m'] and abs(left) <= 0.5)
            if d.get('trigger') is True or ahead:
                self.last_obstacle = (self.now_s(), d['kind'], fwd)
                return

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
        # [COVER] coverage goals are blacklisted the same way as frontier goals
        if self.goal_xy is not None and (self.goal_is_frontier or self.goal_kind == 'coverage'):
            self.blacklist.append(self.goal_xy)
            self.get_logger().warn(
                f'Blacklisted ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}): {why}')

    def send_goal(self, xy, robot_xy, is_frontier=True, kind=None, yaw=None):  # [COVER] kind
        if yaw is None:  # [ROOM] room goals face into the room
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
        # [COVER] [HOLD] goal kind + per-goal flags
        self.goal_kind = kind or ('frontier' if is_frontier else 'return')
        self.goal_dwell_done = False
        self.goal_hold_used = False
        distance = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
        self.goal_time_limit = self.P['time_limit_base_s'] + self.P['time_limit_per_m_s'] * distance

        accepted = self.nav.goToPose(goal)
        if accepted is False:
            if is_frontier or self.goal_kind == 'coverage':  # [COVER]
                self.blacklist_goal('Nav2 rejected the goal')
            else:
                self.get_logger().warn(f'Nav2 rejected recovery goal {xy}')
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

    # ================= [COVER] camera coverage =================

    def update_coverage(self, pose):
        now = self.now_s()
        if pose is not None:
            self.poses.add(pose[0], pose[1], pose[2], now)
        if self.last_coverage_publish is None or now - self.last_coverage_publish >= 1.0:
            self.last_coverage_publish = now
            self.publish_coverage(pose)
        period = self.P['coverage_save_period_s']
        if self.last_coverage_save is None:
            self.last_coverage_save = now
        elif period > 0 and now - self.last_coverage_save >= period:
            self.last_coverage_save = now
            self.save_coverage()

    def current_seen(self):
        """Cells the camera has seen (+ cells the robot stood on), on the current map."""
        if self.grid is None:
            return None
        key = (self.map_seq, len(self.poses.poses), self.seen_tracker.map_key is None)
        if key != self.seen_key or self.seen is None:
            cam = self.seen_tracker.update(self.grid, self.info, self.poses.poses)
            visited = visited_mask(self.grid, self.info, self.poses.poses,
                                   self.P['robot_radius_m'], self.P['free_max'])
            self.seen = cam | visited
            self.seen_key = (self.map_seq, len(self.poses.poses), False)
        return self.seen

    def coverage_now(self):
        seen = self.current_seen()
        if seen is None:
            return None, 0.0
        return seen, coverage_fraction(self.grid, self.info, seen, self.bounds, self.P['free_max'])

    def coverage_goals(self, pose, seen):
        goals = find_coverage_goals(
            self.grid, self.info, seen, self.bounds, self.P['goal_clearance_m'],
            self.P['coverage_min_cluster_cells'], free_max=self.P['free_max'],
            occupied_min=self.P['occupied_min'],
            avoid=self.blacklist, avoid_radius_m=self.P['blacklist_radius_m'])
        return score_frontiers(
            goals, robot_xy=pose[:2],
            w_size=self.P['w_size'], w_dist=self.P['w_dist'], w_bias=self.P['w_bias'],
            bias_xy=self.bias_xy, blacklist=self.blacklist,
            blacklist_radius_m=self.P['blacklist_radius_m'],
        )

    def publish_coverage(self, pose):
        seen, frac = self.coverage_now()
        if seen is None:
            return
        free = (self.grid >= 0) & (self.grid <= self.P['free_max'])
        data = np.full(self.grid.shape, -1, dtype=np.int8)
        data[seen] = 0
        data[free & ~seen & bounds_mask(self.info, self.bounds)] = 100
        msg = OccupancyGrid()
        msg.header.frame_id = self.P['global_frame']
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = float(self.info.resolution)
        msg.info.width, msg.info.height = int(self.info.width), int(self.info.height)
        msg.info.origin.position.x = float(self.info.origin_x)
        msg.info.origin.position.y = float(self.info.origin_y)
        msg.info.origin.orientation.w = 1.0
        msg.data = data.ravel().tolist()
        self.coverage_pub.publish(msg)
        self.coverage_pct_pub.publish(Float32(data=100.0 * frac))

        path = Path()
        path.header = msg.header
        for x, y, yaw, _ in self.poses.poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = x, y
            qx, qy, qz, qw = yaw_to_quaternion(yaw)
            ps.pose.orientation.z, ps.pose.orientation.w = qz, qw
            path.poses.append(ps)
        self.trail_pub.publish(path)

        if self.P['dry_run'] and pose is not None:   # preview of COVER goals
            self.publish_cover_markers(self.coverage_goals(pose, seen))

    def publish_cover_markers(self, ranked):
        """Coverage goals: green, own namespace (frontier markers are left alone)."""
        arr = MarkerArray()
        shown = ranked[:20]
        for i, f in enumerate(shown):
            m = Marker()
            m.header.frame_id = self.P['global_frame']
            m.ns, m.id, m.type, m.action = 'coverage_goals', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(f.goal_xy[0]), float(f.goal_xy[1])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3 if i == 0 else 0.18
            m.color.r, m.color.g, m.color.b = (0.0, 0.9, 0.2) if i == 0 else (0.4, 0.8, 0.4)
            m.color.a = 0.9
            arr.markers.append(m)
        for i in range(len(shown), self.n_cover_markers):
            m = Marker()
            m.header.frame_id = self.P['global_frame']
            m.ns, m.id, m.action = 'coverage_goals', i, Marker.DELETE
            arr.markers.append(m)
        self.n_cover_markers = len(shown)
        self.cover_markers = arr
        self.marker_pub.publish(arr)

    def save_coverage(self):
        if not self.P['coverage_enabled'] or len(self.poses.poses) == self.n_saved:
            return
        meta = {
            'global_frame': self.P['global_frame'],
            'saved_at': self.now_s(),
            'camera_fov_deg': self.P['camera_fov_deg'],
            'camera_range_m': self.P['camera_range_m'],
            'bounds': self.bounds,
        }
        try:
            save(self.coverage_path, self.poses.poses, meta)
            self.n_saved = len(self.poses.poses)
            self.get_logger().info(f'Coverage saved: {self.n_saved} poses -> {self.coverage_path}',
                                   throttle_duration_sec=60.0)
        except OSError as e:
            self.get_logger().warn(f'Could not save coverage to {self.coverage_path}: {e}')

    def resume_coverage(self):
        src = (os.path.expanduser(self.P['coverage_file']) if self.P['coverage_file']
               else newest_coverage_file())
        if not src or not os.path.exists(src):
            self.get_logger().warn(f'resume_coverage: no coverage file found ({src}); starting empty.')
            return
        try:
            poses, _ = load(src)
        except Exception as e:   # corrupt / old format: never stop the explorer for this
            self.get_logger().error(f'resume_coverage: could not load {src}: {e!r}; starting empty.')
            return
        self.poses.poses = list(poses)
        self.get_logger().warn(
            f'Resumed {len(poses)} coverage poses from {src}. This is ONLY valid if SLAM kept '
            'the same map frame (same saved map / same start pose); otherwise set '
            'resume_coverage: false.')

    def eligible_frontiers(self, pose):
        """Same frontier search + bounds filter + ranking as do_select (no markers)."""
        frontiers = find_frontiers(
            self.grid, self.info,
            min_size_cells=self.P['min_frontier_size_cells'],
            free_max=self.P['free_max'],
            occupied_min=self.P['occupied_min'],
            clearance_m=self.P['goal_clearance_m'],
        )
        if self.bounds is not None:
            xmin, ymin, xmax, ymax = self.bounds
            frontiers = [f for f in frontiers
                         if xmin <= f.goal_xy[0] <= xmax and ymin <= f.goal_xy[1] <= ymax]
        return score_frontiers(
            frontiers, robot_xy=pose[:2],
            w_size=self.P['w_size'], w_dist=self.P['w_dist'], w_bias=self.P['w_bias'],
            bias_xy=self.bias_xy, blacklist=self.blacklist,
            blacklist_radius_m=self.P['blacklist_radius_m'],
        )

    def unseen_goal_cells(self):
        """How many cells of the current coverage goal's cluster are still unseen."""
        seen = self.current_seen()
        if seen is None:
            return len(self.goal_cells_xy)
        n = 0
        for x, y in self.goal_cells_xy:
            r, c = self.info.world_to_cell(x, y)
            if not (self.info.in_bounds(r, c) and seen[r, c]):
                n += 1
        return n

    # ================= main loop =================

    def tick(self):
        pose = self.robot_pose()
        # [COVER] remember where we have been: every tick, also while paused
        # (TRAVERSE / INVESTIGATE) and in dry run
        if self.P['coverage_enabled']:
            self.update_coverage(pose)

        self.update_cells(pose)  # [ROOM]
        if not self.enabled:
            if self.state in ('NAVIGATE', 'SPIN', 'ROOM_LOOK') and not self.P['dry_run']:
                self.get_logger().info(f'Cancelling Nav2 {self.state.lower()} for exploration pause.')
                self.nav.cancelTask()
            if self.state in ('SPIN', 'RECOVER', 'HOLD', 'ROOM_LOOK'):  # [HOLD] stop on pause too
                self.publish_cmd()
            if self.state != 'DONE':
                self.set_state('PAUSED')
            return
        if self.state == 'PAUSED':
            self.set_state('SELECT')

        handler = getattr(self, f'do_{self.state.lower()}')
        handler(pose)
        # [COVER] dry run: do_select clears all markers every tick, re-show coverage goals
        if self.P['coverage_enabled'] and self.P['dry_run'] and self.cover_markers is not None:
            self.marker_pub.publish(self.cover_markers)

    def do_wait_map(self, pose):
        if self.grid is None or pose is None:
            self.log_missing_inputs(pose)
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

    def log_missing_inputs(self, pose):
        self.get_logger().info(
            f'{self.state}: waiting for exploration inputs '
            f'(map={"missing" if self.grid is None else "ready"}, '
            f'TF {self.P["global_frame"]} -> {self.P["robot_frame"]}='
            f'{"missing" if pose is None else "ready"})', throttle_duration_sec=5.0)

    def send_gait(self):
        send_gait(self.gait_pub, MotionSimpleCMD, self.P['gait_code'], self.get_logger(),
                  via_manual=self.P['gait_via_manual'])

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
            self.get_logger().warn('Look-around spin timed out; cancelling and selecting a frontier.')
            self.nav.cancelTask()
            self.set_state('SELECT')

    #chooses the next destination
    def do_select(self, pose):
        if self.grid is None or pose is None:
            self.log_missing_inputs(pose)
            return
        # [ROOM] a nearby unchecked room first (QR codes are on room side walls)
        if self.start_room(pose, self.P['room_check_radius_m']):
            return
        frontiers = find_frontiers(
            self.grid, self.info,
            min_size_cells=self.P['min_frontier_size_cells'],
            free_max=self.P['free_max'],
            occupied_min=self.P['occupied_min'],
            clearance_m=self.P['goal_clearance_m'],
        )
        detected = len(frontiers)
        if self.bounds is not None:
            xmin, ymin, xmax, ymax = self.bounds
            frontiers = [f for f in frontiers
                         if xmin <= f.goal_xy[0] <= xmax and ymin <= f.goal_xy[1] <= ymax]
        ranked = score_frontiers(
            frontiers, robot_xy=pose[:2],
            w_size=self.P['w_size'], w_dist=self.P['w_dist'], w_bias=self.P['w_bias'],
            bias_xy=self.bias_xy, blacklist=self.blacklist,
            blacklist_radius_m=self.P['blacklist_radius_m'],
        )
        self.publish_markers(ranked)
        counts = (f'{detected} detected, {detected - len(frontiers)} outside bounds, '
                  f'{len(frontiers) - len(ranked)} blacklisted, {len(ranked)} eligible')

        if not ranked:
            # [COVER] already covering (after a coverage goal / a pause): straight back
            if self.covering and not self.P['dry_run']:
                self.enter_cover()
                return
            self.empty_checks += 1
            self.get_logger().info(
                f'No eligible frontier ({counts}); empty checks '
                f'{self.empty_checks}/{self.P["done_after_empty_checks"]}',
                throttle_duration_sec=2.0)
            if self.P['dry_run']:
                return  # keep watching the map while you drive by hand
            if self.empty_checks >= self.P['done_after_empty_checks']:
                # [ROOM] no frontiers left: every remaining room, however far
                if self.start_room(pose, None):
                    return
                # [COVER] mapped is not seen: look where the camera has not been first
                if self.P['coverage_enabled']:
                    self.get_logger().info('No reachable frontiers left: checking camera coverage (COVER).')
                    self.covering = True
                    self.cover_empty_checks = 0
                    self.enter_cover()
                    return
                self.get_logger().info('No reachable frontiers left: exploration done.')
                self.done_pub.publish(Bool(data=True))
                self.set_state('DONE')
            return
        self.empty_checks = 0
        self.covering = False  # [COVER] frontiers again: normal exploration

        top = ', '.join(f'({f.goal_xy[0]:.2f}, {f.goal_xy[1]:.2f}) score={f.score:.2f} '
                        f'length={f.length_m:.2f} m distance={f.distance_m:.2f} m'
                        for f in ranked[:3])
        if self.P['dry_run']:
            self.get_logger().info(f'Dry-run frontier ranking ({counts}), best first: {top}',
                                   throttle_duration_sec=5.0)
            return  # only visualise

        best = ranked[0]
        self.get_logger().info(f'Selected highest-scoring frontier ({counts}), best first: {top}')
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
            if result == TaskResult.SUCCEEDED and self.goal_kind == 'room':  # [ROOM]
                self.failures_in_a_row = 0
                self.last_good_xy = self.goal_xy
                self.start_room_look(pose)
                return
            if result == TaskResult.SUCCEEDED:
                # [HOLD] arrival dwell first; this branch runs again (unchanged) after it
                if self.start_arrival_hold():
                    return
                self.get_logger().info(f'Goal ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) reached.')
                self.failures_in_a_row = 0
                self.last_good_xy = self.goal_xy
                # Don't come straight back if this spot is still a frontier (e.g. glass).
                if self.goal_is_frontier:
                    self.blacklist.append(self.goal_xy)
                # [COVER] reached coverage goals are blacklisted too
                if self.goal_kind == 'coverage':
                    self.blacklist.append(self.goal_xy)
                # Look-around in a new area (collision-checked Nav2 spin) so the VLM and the
                # QR reader see the side walls.
                if (self.P['goal_spin_deg'] > 0 and pose is not None
                        and (self.last_spin_xy is None
                             or math.hypot(pose[0] - self.last_spin_xy[0],
                                           pose[1] - self.last_spin_xy[1])
                             >= self.P['goal_spin_min_gap_m'])):
                    angle = math.radians(self.P['goal_spin_deg'])
                    limit = angle / max(self.P['spin_speed'], 1e-3) + 10.0
                    if self.nav.spin(spin_dist=angle, time_allowance=int(limit)):
                        self.last_spin_xy = pose[:2]
                        self.state_end = self.now_s() + limit + 5.0
                        self.get_logger().info('Starting look-around spin at the reached goal.')
                        self.set_state('SPIN')       # do_spin goes back to SELECT when done
                        return
                    self.get_logger().warn('Nav2 refused the look-around spin at the goal.')
                self.set_state('SELECT')
            else:
                self.start_recovery(f'Nav2 result {result.name}')
            return

        # 2. Over the time limit?
        if now - self.goal_start > self.goal_time_limit:
            self.nav.cancelTask()
            self.start_recovery(f'goal time limit ({self.goal_time_limit:.0f} s)')
            return

        # 3. Making progress?
        feedback = self.nav.getFeedback()
        remaining = getattr(feedback, 'distance_remaining', None) if feedback else None
        distance = f'{remaining:.2f} m' if remaining is not None else 'awaiting Nav2 feedback'
        self.get_logger().info(
            f'Navigating to ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}), '
            f'remaining={distance}, elapsed={now - self.goal_start:.1f}/{self.goal_time_limit:.0f} s',
            throttle_duration_sec=2.0)
        if remaining is not None and remaining > 0.0:
            if self.window_remaining is None:
                self.window_remaining = remaining
                self.window_start = now
            elif now - self.window_start > self.P['progress_window_s']:
                progress = self.window_remaining - remaining
                if progress < self.P['min_progress_m']:
                    self.nav.cancelTask()
                    self.start_recovery(f'progress {progress:.2f} m in '
                                        f'{now - self.window_start:.1f} s, '
                                        f'need {self.P["min_progress_m"]:.2f} m')
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

        # 5. [COVER] Coverage goal area already seen on the way? Pick a new one (no blacklist).
        if (self.goal_kind == 'coverage' and self.goal_cells_xy
                and now - self.last_recheck > self.P['recheck_period_s']):
            self.last_recheck = now
            unseen = self.unseen_goal_cells()
            if unseen < self.P['coverage_min_cluster_cells']:
                self.get_logger().info(
                    f'Coverage goal area already seen ({unseen} cells left), choosing a new goal.')
                self.nav.cancelTask()
                self.set_state('SELECT')

    def start_recovery(self, why):
        if self.goal_kind == 'room' and self.room is not None:  # [ROOM]
            self.get_logger().warn(f'Room {self.room.cell}: could not reach the doorway ({why})')
            self.room = None
        # [HOLD] obstacle ahead: hold instead of blacklisting (once per goal)
        if self.start_obstacle_hold(why):
            return
        self.blacklist_goal(why)
        self.failures_in_a_row += 1
        self.state_end = self.now_s() + self.P['backup_time_s']
        self.get_logger().warn(f'Recovery after {why}; consecutive failures={self.failures_in_a_row}, '
                               f'backup time={self.P["backup_time_s"]:.1f} s')
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

    # ================= [HOLD] stand still so obstacle_detector can confirm =================

    def start_hold(self, seconds, then, why):
        self.hold_end = self.now_s() + seconds
        self.hold_then = then
        self.publish_cmd()
        self.get_logger().info(f'HOLD {seconds:.1f} s {why} so obstacle_detector can confirm')
        self.set_state('HOLD')

    def start_arrival_hold(self):
        """Reached a frontier / coverage goal: dwell once, facing the travel direction
        (the Nav2 goal yaw points from the start towards the goal)."""
        if (self.goal_dwell_done or self.P['goal_dwell_s'] <= 0 or self.P['dry_run']
                or self.goal_kind not in ('frontier', 'coverage')):
            return False
        self.goal_dwell_done = True
        self.start_hold(self.P['goal_dwell_s'], ('navigate',), 'at goal')
        return True

    def recent_obstacle(self):
        if self.last_obstacle is None:
            return None
        t, kind, fwd = self.last_obstacle
        if self.now_s() - t > self.P['obstacle_recent_s']:
            return None
        return kind, fwd

    def start_obstacle_hold(self, why):
        """Nav2 failed / timed out / no progress, and a gait obstacle is right ahead:
        hold (no blacklist, no back-up) so the detector can fire. Once per goal."""
        if (self.P['obstacle_hold_s'] <= 0 or self.P['dry_run'] or self.goal_hold_used
                or self.goal_kind not in ('frontier', 'coverage')):
            return False
        obstacle = self.recent_obstacle()
        if obstacle is None:
            return False
        self.goal_hold_used = True
        kind, fwd = obstacle
        dist = f'{fwd:.2f}' if isinstance(fwd, (int, float)) else '?'
        self.get_logger().warn(
            f'obstacle ahead ({kind} at {dist} m): holding instead of blacklisting ({why})')
        self.nav.cancelTask()
        self.start_hold(self.P['obstacle_hold_s'], ('recover', why), 'at blocked goal')
        return True

    def do_hold(self, pose):
        self.publish_cmd()
        if self.now_s() < self.hold_end:
            return
        then, self.hold_then = self.hold_then, None
        if then is not None and then[0] == 'recover':
            self.start_recovery(then[1])     # exactly today's blacklist + recovery
        else:
            self.set_state('NAVIGATE')       # finished goal: arrival code runs now
            self.do_navigate(pose)

    # ================= [COVER] COVER state =================

    def enter_cover(self):
        if self.state != 'COVER':
            self.seen_tracker.reset()        # full recompute on the latest map
        self.set_state('COVER')

    def finish_coverage(self, why):
        _, frac = self.coverage_now()
        where = 'inside bounds' if self.bounds is not None else 'of the whole map (bounds off)'
        self.get_logger().info(
            f'Exploration done: {why}. Final camera coverage {100.0 * frac:.1f}% {where}.')
        self.save_coverage()
        self.done_pub.publish(Bool(data=True))
        self.set_state('DONE')

    def do_cover(self, pose):
        if self.grid is None or pose is None:
            self.log_missing_inputs(pose)
            return
        # 1. Frontiers back (map changed)? Then normal exploration.
        if self.eligible_frontiers(pose):
            self.get_logger().info('COVER: new frontiers appeared, back to frontier exploration.')
            self.covering = False
            self.empty_checks = 0
            self.cover_empty_checks = 0
            self.set_state('SELECT')
            return
        # [ROOM] remaining rooms before more coverage
        if self.start_room(pose, None):
            return
        # 2. Seen enough?
        seen, frac = self.coverage_now()
        target = self.P['coverage_target']
        if frac >= target:
            self.finish_coverage(f'camera coverage {100.0 * frac:.1f}% >= target {100.0 * target:.0f}%')
            return
        # 3. Best coverage goal (same ranking + blacklist as frontiers)
        ranked = self.coverage_goals(pose, seen)
        self.publish_cover_markers(ranked)
        if not ranked:
            self.cover_empty_checks += 1
            self.get_logger().info(
                f'COVER: no reachable coverage goals (coverage {100.0 * frac:.1f}%); empty checks '
                f'{self.cover_empty_checks}/{self.P["done_after_empty_checks"]}',
                throttle_duration_sec=2.0)
            if self.cover_empty_checks >= self.P['done_after_empty_checks']:
                self.finish_coverage('no reachable coverage goals left')
            return
        self.cover_empty_checks = 0
        best = ranked[0]
        self.get_logger().info(
            f'COVER: coverage {100.0 * frac:.1f}% (target {100.0 * target:.0f}%), '
            f'{len(ranked)} goals; best ({best.goal_xy[0]:.2f}, {best.goal_xy[1]:.2f}) '
            f'{len(best.cells)} unseen cells')
        if self.send_goal(best.goal_xy, pose[:2], is_frontier=False, kind='coverage'):
            self.goal_cells_xy = [self.info.cell_to_world(r, c) for r, c in best.cells]
            self.set_state('NAVIGATE')

    # ================= [ROOM] cells, walls, room check =================

    def update_cells(self, pose):
        """Analyse cells when the map changed (<= 1 Hz), mark visited cells, publish overlay."""
        now = self.now_s()
        if pose is not None:
            c = self.cells.cell_of(pose[0], pose[1])
            if c not in self.visited_cells and self.cells.in_visit_region(c, pose[0], pose[1]):
                self.visited_cells.add(c)
        if self.grid is None or now - self.last_cells_pub < 1.0:
            return
        if self.map_seq != self.cells_seq:
            self.cell_info = self.cells.analyze(self.grid, self.info, self.P['free_max'],
                                                self.P['occupied_min'])
            self.cells_seq = self.map_seq
        self.last_cells_pub = now
        if self.P['cell_overlay']:
            self.publish_cells()

    def room_candidates(self, pose, radius):
        out = []
        for room in self.cells.rooms(self.cell_info):
            if room.cell == (0, 0):
                continue                  # start cell: the initial look-around spin covers it
            st = self.room_status.get(room.cell, {'attempts': 0, 'done': False})
            if st['done'] or st['attempts'] >= self.P['room_max_attempts']:
                continue
            x, y = room.standpoint
            if self.bounds is not None:
                xmin, ymin, xmax, ymax = self.bounds
                if not (xmin <= x <= xmax and ymin <= y <= ymax):
                    continue
            d = math.hypot(x - pose[0], y - pose[1])
            if radius is None or d <= radius:
                out.append((d, room))
        return [r for _, r in sorted(out, key=lambda t: t[0])]

    def start_room(self, pose, radius):
        """Go to the nearest unchecked room's doorway. True if a goal was sent."""
        if (not self.P['room_check'] or self.P['dry_run'] or pose is None
                or not self.cell_info):
            return False
        for room in self.room_candidates(pose, radius):
            st = self.room_status.setdefault(room.cell, {'attempts': 0, 'done': False,
                                                         'views_done': []})
            st['attempts'] += 1
            self.get_logger().info(
                f'ROOM check {room.cell} (doorway {room.open_side}, QR walls {room.side_walls}), '
                f'attempt {st["attempts"]}/{self.P["room_max_attempts"]}')
            if self.send_goal(room.standpoint, pose[:2], is_frontier=False, kind='room',
                              yaw=room.views[0][1]):
                self.room = room
                self.set_state('NAVIGATE')
                return True
        return False

    def start_room_look(self, pose):
        self.room_idx = 0
        self.set_state('ROOM_LOOK')
        self.start_room_turn(pose)

    def start_room_turn(self, pose):
        """Turn (Nav2 spin, collision-checked) to the next view, or start the dwell."""
        name, yaw = self.room.views[self.room_idx]
        delta = wrap(yaw - pose[2]) if pose is not None else 0.0
        if abs(delta) > math.radians(8):
            limit = abs(delta) / max(self.P['spin_speed'], 1e-3) + 8.0
            if self.nav.spin(spin_dist=delta, time_allowance=int(limit)):
                self.room_phase = 'turn'
                self.room_until = self.now_s() + limit + 3.0
                return
            self.get_logger().warn(f'ROOM {self.room.cell}: Nav2 refused the turn to {name}')
        self.start_room_dwell()

    def start_room_dwell(self):
        name, _ = self.room.views[self.room_idx]
        self.room_phase = 'dwell'
        self.room_until = self.now_s() + self.P['room_dwell_s']
        self.publish_cmd()
        self.look_pub.publish(Bool(data=True))       # one VLM photo of this view (booklet J1)
        self.get_logger().info(f'ROOM {self.room.cell}: looking at {name}')

    def do_room_look(self, pose):
        if self.room is None:
            self.set_state('SELECT')
            return
        now = self.now_s()
        if self.room_phase == 'turn':
            if self.nav.isTaskComplete() or now > self.room_until:
                if not self.nav.isTaskComplete():
                    self.nav.cancelTask()
                self.start_room_dwell()
            return
        self.publish_cmd()
        if now < self.room_until:
            return
        st = self.room_status[self.room.cell]
        st['views_done'].append(self.room.views[self.room_idx][0])
        self.room_idx += 1
        if self.room_idx < len(self.room.views):
            self.start_room_turn(pose)
            return
        st['done'] = True
        self.get_logger().info(f'ROOM {self.room.cell} checked ({len(self.room.views)} views).')
        self.room = None
        self.set_state('SELECT')

    def publish_cells(self):
        """Cells (green visited / grey free), walls (yellow; blue once a room check looked at
        them), room labels (purple, blue when checked). Display only."""
        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        frame = self.P['global_frame']
        stamp = self.get_clock().now().to_msg()

        def mk(ns, mid, mtype):
            m = Marker()
            m.header.frame_id, m.header.stamp = frame, stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            return m
        cubes = mk('cells', 0, Marker.CUBE_LIST)
        cubes.scale.x = cubes.scale.y = 0.5
        cubes.scale.z = 0.02
        walls = mk('walls', 0, Marker.LINE_LIST)
        walls.scale.x = 0.06
        rooms = {r.cell for r in self.cells.rooms(self.cell_info)}
        h = self.cells.size / 2.0
        n = 1
        for cell, ci in self.cell_info.items():
            if ci.state != 'free' and cell not in self.visited_cells:
                continue
            cubes.points.append(Point(x=ci.centre[0], y=ci.centre[1], z=0.01))
            cubes.colors.append(ColorRGBA(r=0.1, g=0.8, b=0.2, a=0.5) if cell in self.visited_cells
                                else ColorRGBA(r=0.8, g=0.8, b=0.8, a=0.35))
            looked = set(self.room_status.get(cell, {}).get('views_done', []))
            for side in ci.walls:
                nu, nv = SIDES[side]
                col = (ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.95) if side in looked
                       else ColorRGBA(r=1.0, g=0.9, b=0.0, a=0.95))
                for t in (-1.0, 1.0):
                    du = nu * (h - 0.05) - nv * t * (h - 0.1)
                    dv = nv * (h - 0.05) + nu * t * (h - 0.1)
                    mx, my = self.cells.vec_to_map(du, dv)
                    walls.points.append(Point(x=ci.centre[0] + mx, y=ci.centre[1] + my, z=0.05))
                    walls.colors.append(col)
            if cell in rooms:
                st = self.room_status.get(cell, {})
                label = mk('rooms', n, Marker.TEXT_VIEW_FACING)
                n += 1
                label.pose.position.x, label.pose.position.y = ci.centre
                label.pose.position.z = 0.3
                label.scale.z = 0.18
                label.color = (ColorRGBA(r=0.3, g=0.6, b=1.0, a=1.0) if st.get('done')
                               else ColorRGBA(r=0.8, g=0.3, b=1.0, a=1.0))
                label.text = f'room {cell[0]},{cell[1]}' + (' ok' if st.get('done') else '')
                arr.markers.append(label)
        if cubes.points:
            arr.markers.append(cubes)
        if walls.points:
            arr.markers.append(walls)
        self.cells_pub.publish(arr)

    def stop(self):
        try:
            if self.state in ('NAVIGATE', 'SPIN', 'ROOM_LOOK') and not self.P['dry_run']:
                self.nav.cancelTask()
        finally:
            self.publish_cmd()
            self.save_coverage()  # [COVER] keep the pose history for the next run


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
