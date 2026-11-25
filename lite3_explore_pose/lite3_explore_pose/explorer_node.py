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
from std_msgs.msg import Bool, ColorRGBA, String
from std_msgs.msg import Float32  # [COVER]
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explore_pose.frontiers import (
    MapInfo, find_frontiers, score_frontiers, still_frontier,
)
import re  # [RECORD]

from lite3_explore_pose.frontier_record import FrontierRecord, record_path
from lite3_explore_pose.investigate import photo_is_stale
from lite3_explore_pose.frontier_goals import (
    cell_seen_fraction, frontier_step, target_cell, unknown_side_xy,
)
# [COVER] camera coverage (pose history -> seen mask -> COVER goals)
from lite3_explore_pose.coverage import (
    PoseHistory, SeenTracker, bounds_mask, coverage_fraction, default_coverage_path,
    find_coverage_goals, load, newest_coverage_file, save, visited_mask,
)

# [ROOM] arena cells, walls and rooms from the live map (cells.py)
from lite3_explore_pose.cells import CellGrid, SIDES, wrap
from lite3_explore_pose.mission_config import Arena, load_mission
from lite3_explore_pose.pose_health import PoseHealth

# [HOLD] detections that may be the reason Nav2 is stuck (gait-change obstacles)
HOLD_KINDS = ('stairs', 'rubble', 'curtain', 'potholes')


def failure_key(why):
    """Short, countable failure reason: 'Nav2 result ABORTED' -> 'Nav2 ABORTED'."""
    m = re.match(r'Nav2 result (\w+)', why)
    if m:
        return f'Nav2 {m.group(1)}'
    for key, short in (('goal time limit', 'time limit'), ('progress', 'no progress'),
                       ('rejected', 'Nav2 rejected')):
        if key in why:
            return short
    return why


def yaw_to_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class FrontierExplorer(Node):

    def __init__(self):
        super().__init__('frontier_explorer')

        # ---------------- parameters (see config/explorer.yaml) ----------------
        p = self.declare_parameter
        p('dry_run', True)               # True: show frontiers in RViz, never move
        p('global_frame', 'map')
        p('odom_frame', 'odom')
        p('robot_frame', 'base_link')
        p('pose_max_age_s', 2.5)
        p('slam_max_age_s', 5.0)
        p('pose_settle_s', 1.0)
        p('pose_result_wait_s', 3.0)
        p('pose_goal_tolerance_m', 0.35)
        p('pose_goal_cooldown_s', 15.0)
        p('cmd_vel_topic', '/nav_cmd_vel')  # twist_mux input, never /cmd_vel
        p('tick_hz', 5.0)
        # Gait at start (issue 4): asked from the gait owner (special_obstacles) on
        # /gait/request with a name from config/gait.yaml; the explorer never sends codes.
        # '' = keep the current gait.
        p('start_gait', 'flat')

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
        # [GRID] visit arena cells one by one (nearest free unvisited cell centre first) instead
        # of chasing frontiers; frontiers only uncover cells that are still unknown. Frontiers,
        # coverage and cells are limited to grid_bounds_cells (arena cells relative to the start
        # cell: [i_min, j_min, i_max, j_max], +i forward, +j left).
        p('grid_explore', False)
        p('grid_bounds_cells', [0, -2, 5, 2])
        p('grid_max_attempts', 2)
        p('cell_visit_edge_margin_m', 0.10)
        # [ENTER] issue 1: a frontier goal is done only when the robot entered the cell the
        # frontier leads into (or the camera saw it). A frontier mapped from afar becomes a
        # 'go into that cell' goal instead of being cancelled. False = old cancel-on-mapped.
        p('frontier_enter_cell', True)
        p('frontier_seen_done_frac', 0.9)   # camera saw this share of the target cell = done
        p('frontier_lead_m', 0.5)           # target = cell this far past the frontier
        # Position-jump latch (pose_health.py): clears after this many s of stable TF.
        # 0 = old permanent latch. Jumps over pose_jump_hard_m never clear.
        p('pose_jump_recover_s', 10.0)
        p('pose_jump_hard_m', 5.0)
        p('idle_warn_period_s', 5.0)        # throttled WARN 'not moving: state, reason'
        # [RECORD] issue 2: every chosen frontier is recorded (pending / explored / failed,
        # saved next to the coverage file). Before COVER / DONE every failed or still
        # pending frontier is retried once (frontier_retry_delay_s after its failure).
        p('frontier_retry_enabled', True)
        p('frontier_retry_delay_s', 20.0)
        # [WALLS] a wall turns blue in /exploration/cells once the camera saw this share
        # of it (coverage seen mask), not only after a room check. Walls of visited cells
        # are kept even if a later map analysis calls the cell partial.
        p('wall_seen_frac', 0.5)
        p('keep_visited_walls', True)
        # Issue 3: spectators above the 0.54 m walls show up as map noise and frontiers
        # outside the arena. true = frontiers / coverage limited to the grid_bounds_cells
        # box also without grid_explore. false = old (guard_radius_m box).
        p('use_grid_bounds_box', False)
        # Issue 5: each room view waits for /perception/look_done (a VLM answer takes
        # longer than room_dwell_s) before turning to the next view; at most
        # room_look_timeout_s. A stale answer (photo taken before the look) -> one more look.
        p('room_look_wait_done', True)
        p('room_look_timeout_s', 15.0)
        p('room_max_relooks', 1)

        g = lambda name: self.get_parameter(name).value
        self.P = {name: g(name) for name in [
            'dry_run', 'global_frame', 'robot_frame', 'tick_hz', 'initial_spin',
            'spin_speed', 'spin_turns', 'min_frontier_size_cells', 'free_max',
            'occupied_min', 'goal_clearance_m', 'w_size', 'w_dist', 'w_bias',
            'time_limit_base_s', 'time_limit_per_m_s', 'progress_window_s',
            'min_progress_m', 'recheck_period_s', 'blacklist_radius_m',
            'backup_speed', 'backup_time_s', 'max_consecutive_failures',
            'done_after_empty_checks', 'start_gait',
            'bounds', 'guard_radius_m', 'goal_spin_deg', 'goal_spin_min_gap_m',
            'odom_frame', 'pose_max_age_s', 'slam_max_age_s', 'pose_settle_s',
            'pose_result_wait_s', 'pose_goal_tolerance_m', 'pose_goal_cooldown_s',
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
            'cell_overlay', 'grid_explore', 'grid_bounds_cells', 'grid_max_attempts',
            'cell_visit_edge_margin_m',
            'frontier_enter_cell', 'frontier_seen_done_frac', 'frontier_lead_m',
            'idle_warn_period_s', 'frontier_retry_enabled', 'frontier_retry_delay_s',
            'wall_seen_frac', 'keep_visited_walls', 'room_look_wait_done',
            'room_look_timeout_s', 'room_max_relooks',
        ]})
        try:
            arena = Arena(load_mission(g('mission_file')))
        except Exception as e:  # noqa: BLE001  (fall back to the booklet grid)
            self.get_logger().warn(f'mission_file not readable ({e}); booklet grid 1.02 m')
            arena = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0.0, 0.0],
                                     'start_heading_deg': 0.0, 'final_cell_centre': None}})
        self.cells = CellGrid(
            arena, clearance_m=g('goal_clearance_m'),
            bounds_cells=list(g('grid_bounds_cells')) if g('grid_explore') else None)
        self.cell_attempts = {}           # [GRID] cell -> goals sent
        self.cell_info = {}               # cell -> CellInfo (last analysis)
        self.cells_seq = -1               # map_seq of that analysis
        self.visited_cells = set()
        self.visit_sample = None
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
        if g('use_grid_bounds_box') and not self.P['grid_explore']:
            self.bounds = CellGrid(arena, bounds_cells=list(g('grid_bounds_cells'))).map_box()
            self.get_logger().info(f'use_grid_bounds_box: frontiers/coverage limited to '
                                   f'{[round(v, 2) for v in self.bounds]}')
        if self.P['grid_explore']:          # [GRID] arena box instead of the large-map guard
            self.bounds = self.cells.map_box()
            self.get_logger().info(
                f'GRID explore: cells {self.P["grid_bounds_cells"]} -> map box '
                f'{[round(v, 2) for v in self.bounds]}')
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
        # [RECORD] frontier record next to the coverage file
        self.frontier_rec = FrontierRecord(self.P['blacklist_radius_m'])
        self.record_path = record_path(self.coverage_path)
        self.retry_queue = None           # ids still to retry in the final pass
        self.final_retry_done = False
        self.goal_record_xy = None        # frontier xy this goal belongs to (record key)
        self.walls_seen = set()           # [WALLS] (cell, side) the camera looked at
        self.wall_memory = {}             # [WALLS] visited cell -> walls ever found
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
        self.create_subscription(Bool, '/exploration/enable', self.on_enable, map_qos)
        self.turn_hold_active = False
        self.create_subscription(Bool, '/navigation/turn_hold_active', self.on_turn_hold, map_qos)
        self.create_subscription(PointStamped, '/exploration/bias', self.on_bias, 10)
        # [HOLD] obstacle_detector's per-frame detections
        if self.P['obstacle_hold_s'] > 0:
            self.create_subscription(String, '/perception/obstacles', self.on_obstacles, 10)

        self.cmd_pub = self.create_publisher(Twist, g('cmd_vel_topic'), 10)
        self.status_pub = self.create_publisher(String, '/exploration/status', 10)
        self.done_pub = self.create_publisher(Bool, '/exploration/done', 10)
        self.localization_pub = self.create_publisher(Bool, '/exploration/localization_ready', 1)
        self.marker_pub = self.create_publisher(MarkerArray, '/exploration/frontiers', 10)
        # [ROOM] RViz overlay + VLM trigger
        self.cells_pub = self.create_publisher(MarkerArray, '/exploration/cells', 10)
        self.look_pub = self.create_publisher(Bool, '/perception/look', 10)
        self.look_done_t = None           # [LOOK] issue 5
        self.scene_latency = None         # (arrival, latency_s) of the latest scene answer
        self.create_subscription(Bool, '/perception/look_done', self.on_look_done, 10)
        self.create_subscription(String, '/perception/scene', self.on_scene, 10)
        self.last_cells_pub = 0.0
        # [COVER] RViz: camera coverage grid (latched), pose trail, coverage %
        if self.P['coverage_enabled']:
            self.coverage_pub = self.create_publisher(OccupancyGrid, '/exploration/coverage', map_qos)
            self.trail_pub = self.create_publisher(Path, '/exploration/trail', 10)
            self.coverage_pct_pub = self.create_publisher(Float32, '/exploration/coverage_pct', 10)
        self.gait_req_pub = self.create_publisher(String, '/gait/request', 10)
        self.gait_current = None
        self.create_subscription(String, '/gait/current', self.on_gait_current, map_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_health = PoseHealth(
            self.tf_buffer, g('global_frame'), g('odom_frame'), g('robot_frame'),
            g('pose_max_age_s'), g('slam_max_age_s'), g('pose_settle_s'),
            jump_recover_s=g('pose_jump_recover_s'), hard_jump_m=g('pose_jump_hard_m'))
        self.pose_problem = None

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
        self.goal_yaw = None
        self.goal_pose_stamp = None
        self.goal_odom_stamp = None
        self.success_seen_at = None
        self.pose_retry_pending = False
        self.goal_was_retried = False
        self.pose_cancel_pending = False
        self.deferred_goals = {}
        self.state_end = None
        # [COVER] [HOLD] goal kind and per-goal flags (goal_is_frontier is kept as it was)
        self.goal_kind = 'frontier'       # 'frontier' | 'coverage' | 'return'
        self.goal_cells_xy = []           # coverage goal: world xy of its unseen cells
        self.goal_dwell_done = False      # arrival HOLD already done for this goal
        self.goal_hold_used = False       # obstacle HOLD already used for this goal
        self.goal_target_cell = None      # [ENTER] cell a frontier goal leads into
        self.goal_converted = False       # [ENTER] frontier mapped -> now 'enter that cell'
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
            if msg.data:
                self.pose_health.reset()
        self.enabled = msg.data
        if msg.data and self.state == 'DONE':
            self.empty_checks = 0
            self.covering = False  # [COVER] new area may be open: frontiers first again
            self.cover_empty_checks = 0
            self.set_state('SELECT')

    def on_look_done(self, msg):
        if msg.data:
            self.look_done_t = self.now_s()

    def on_scene(self, msg):
        try:
            latency = json.loads(msg.data).get('latency_s')
        except (ValueError, AttributeError):
            return
        if latency is not None:
            self.scene_latency = (self.now_s(), float(latency))

    def on_gait_current(self, msg):
        self.gait_current = msg.data

    def on_turn_hold(self, msg):
        self.turn_hold_active = msg.data

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
        """Return the robot pose only while localization is ready."""
        previous_problem = self.pose_problem
        t, self.pose_problem = self.pose_health.sample(self.now_s())
        if getattr(self.pose_health, 'recovered', None):
            self.get_logger().warn(self.pose_health.recovered)
            self.pose_health.recovered = None
        if t is None:
            if self.pose_health.jump is not None and self.pose_problem != previous_problem:
                self.get_logger().error(self.pose_problem)
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
            if self.goal_is_frontier and getattr(self, 'goal_record_xy', None) is not None:
                self.frontier_rec.failed(self.goal_record_xy, self.now_s(), failure_key(why))

    def send_goal(self, xy, robot_xy, is_frontier=True, kind=None, yaw=None, retry=False,
                  target=None, converted=False):
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
        self.goal_yaw = yaw
        self.goal_was_retried = retry
        self.goal_pose_stamp = self.pose_health.pose_stamp
        self.goal_odom_stamp = self.pose_health.odom_stamp
        self.success_seen_at = None
        self.goal_is_frontier = is_frontier
        # [COVER] [HOLD] goal kind + per-goal flags
        self.goal_kind = kind or ('frontier' if is_frontier else 'return')
        self.goal_dwell_done = False
        self.goal_hold_used = False
        self.goal_target_cell = target      # [ENTER]
        self.goal_converted = converted
        if not converted:                   # [RECORD] converted goals keep their frontier
            self.goal_record_xy = xy if is_frontier else None
        distance = math.hypot(xy[0] - robot_xy[0], xy[1] - robot_xy[1])
        self.goal_time_limit = self.P['time_limit_base_s'] + self.P['time_limit_per_m_s'] * distance

        # Humble's commander retains feedback across actions until new feedback arrives.
        self.nav.feedback = None
        accepted = self.nav.goToPose(goal)
        if accepted is False:
            if retry:
                self.defer_pose_goal('Nav2 rejected the localization retry')
            elif is_frontier or self.goal_kind == 'coverage':
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

    def release_goal_attempt(self):
        """Restore an arena-cell or room attempt interrupted by localization."""
        if self.goal_kind == 'cell':
            cell = self.cells.cell_of(*self.goal_xy)
            self.cell_attempts[cell] = max(0, self.cell_attempts[cell] - 1)
        elif self.goal_kind == 'room' and self.room is not None:
            self.room_status[self.room.cell]['attempts'] -= 1
            self.room = None

    def defer_pose_goal(self, why):
        self.deferred_goals[self.goal_xy] = self.now_s() + self.P['pose_goal_cooldown_s']
        self.release_goal_attempt()
        self.get_logger().warn(f'Deferred goal {self.goal_xy} ({why}); eligible again after cooldown.')

    def is_deferred(self, xy):
        radius = self.P['blacklist_radius_m']
        return any(math.dist(xy, point) < radius for point in self.deferred_goals)

    def wait_for_pose(self, why, retry_goal=False, cancel=False):
        self.get_logger().warn(f'Localization interrupted {self.state}: {why}')
        if cancel:
            self.nav.cancelTask()
            self.pose_cancel_pending = True
        self.publish_cmd()
        self.pose_health.reset()
        self.empty_checks = self.cover_empty_checks = 0
        if retry_goal and not self.goal_was_retried:
            self.pose_retry_pending = True
        elif retry_goal:
            self.defer_pose_goal(why)
        elif self.state == 'ROOM_LOOK':
            self.release_goal_attempt()
        self.set_state('WAIT_POSE')

    def do_wait_pose(self, pose):
        if pose is None:
            self.log_missing_inputs(pose)
            return
        if not self.pose_retry_pending:
            self.set_state('SELECT')
            return
        self.pose_retry_pending = False
        if (self.goal_is_frontier and not self.enter_mode() and not still_frontier(
                self.grid, self.info, self.goal_xy, radius_m=0.4, free_max=self.P['free_max'])):
            self.set_state('SELECT')
            return
        if self.goal_kind == 'coverage' and (
                self.unseen_goal_cells() < self.P['coverage_min_cluster_cells']):
            self.set_state('SELECT')
            return
        self.get_logger().info(f'Localization recovered; retrying goal {self.goal_xy} once.')
        accepted = self.send_goal(self.goal_xy, pose[:2], self.goal_is_frontier,
                                  self.goal_kind, self.goal_yaw, retry=True,
                                  **(self.enter_kwargs() if self.goal_is_frontier else {}))
        self.set_state('NAVIGATE' if accepted else 'SELECT')

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
        if pose is not None and (self.last_coverage_publish is None
                                 or now - self.last_coverage_publish >= 1.0):
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
            avoid=[*self.blacklist, *self.deferred_goals],
            avoid_radius_m=self.P['blacklist_radius_m'])
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

    def save_record(self):
        if not self.frontier_rec.entries:
            return
        try:
            self.frontier_rec.save(self.record_path)
        except OSError as e:
            self.get_logger().warn(f'Could not save frontier record {self.record_path}: {e}')

    def save_coverage(self):
        self.save_record()  # [RECORD]
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
        rec = record_path(src)
        if os.path.exists(rec):
            try:
                self.frontier_rec = FrontierRecord.load(rec, self.P['blacklist_radius_m'])
                self.get_logger().info(f'Resumed frontier record {rec}: '
                                       f'{self.frontier_rec.summary()}')
            except (OSError, ValueError) as e:
                self.get_logger().error(f'resume_coverage: frontier record {rec}: {e!r}')
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
        ranked = score_frontiers(
            frontiers, robot_xy=pose[:2],
            w_size=self.P['w_size'], w_dist=self.P['w_dist'], w_bias=self.P['w_bias'],
            bias_xy=self.bias_xy, blacklist=self.blacklist,
            blacklist_radius_m=self.P['blacklist_radius_m'],
        )
        return [goal for goal in ranked if not self.is_deferred(goal.goal_xy)]

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
        self.localization_pub.publish(Bool(data=(pose is not None and self.grid is not None)))
        now = self.now_s()
        self.deferred_goals = {xy: until for xy, until in self.deferred_goals.items()
                               if until > now}
        # [COVER] remember where we have been: every tick, also while paused
        # (TRAVERSE / INVESTIGATE) and in dry run
        if self.P['coverage_enabled']:
            self.update_coverage(pose)

        self.update_cells(pose)  # [ROOM]
        if not self.enabled:
            if self.pose_retry_pending:
                self.release_goal_attempt()
                self.pose_retry_pending = False
            if self.state in ('NAVIGATE', 'SPIN', 'ROOM_LOOK') and not self.P['dry_run']:
                self.get_logger().info(f'Cancelling Nav2 {self.state.lower()} for exploration pause.')
                self.nav.cancelTask()
            if self.state in ('SPIN', 'RECOVER', 'HOLD', 'ROOM_LOOK'):  # [HOLD] stop on pause too
                self.publish_cmd()
            if self.state != 'DONE':
                self.set_state('PAUSED')
            self.warn_idle('paused by mission_controller (/exploration/enable = false)')
            return
        if self.state == 'PAUSED':
            self.set_state('SELECT')
        if self.pose_cancel_pending:
            if not self.nav.isTaskComplete():
                self.get_logger().warn('Waiting for Nav2 to finish cancelling after pose loss.',
                                       throttle_duration_sec=5.0)
                return
            self.pose_cancel_pending = False
        if (not self.P['dry_run'] and pose is None
                and self.state not in ('WAIT_MAP', 'WAIT_POSE', 'DONE')):
            self.wait_for_pose(self.pose_problem,
                               retry_goal=self.state in ('NAVIGATE', 'HOLD'),
                               cancel=self.state in ('NAVIGATE', 'SPIN', 'ROOM_LOOK', 'HOLD'))
            return

        handler = getattr(self, f'do_{self.state.lower()}')
        handler(pose)
        # [COVER] dry run: do_select clears all markers every tick, re-show coverage goals
        if self.P['coverage_enabled'] and self.P['dry_run'] and self.cover_markers is not None:
            self.marker_pub.publish(self.cover_markers)

    def do_wait_map(self, pose):
        if self.grid is None or pose is None:
            self.log_missing_inputs(pose)
            return
        if self.P['start_gait'] and not self.P['dry_run']:
            self.send_gait()
            self.gait_resent = False
            self.state_end = self.now_s() + 2.0   # stand still while the gait changes
            self.set_state('GAIT')
            return
        self.start_exploring()

    def log_missing_inputs(self, pose):
        self.warn_idle(f'waiting for exploration inputs '
                       f'(map={"missing" if self.grid is None else "ready"}, '
                       f'pose={self.pose_problem if pose is None else "ready"})')

    def warn_idle(self, reason):
        """Issue 1: always say WHY the explorer is not moving (throttled WARN)."""
        self.get_logger().warn(f'Explorer not moving: state={self.state}, reason: {reason}',
                               throttle_duration_sec=self.P.get('idle_warn_period_s', 5.0))

    def send_gait(self):
        """Ask the gait owner (special_obstacles) for the start gait."""
        self.get_logger().info(f'Requesting gait {self.P["start_gait"]!r} on /gait/request')
        self.gait_req_pub.publish(String(data=self.P['start_gait']))

    def do_gait(self, pose):
        self.publish_cmd()
        remaining = self.state_end - self.now_s()
        if remaining < 1.5 and not self.gait_resent:
            self.send_gait()      # send twice in case the first one was missed
            self.gait_resent = True
        if remaining <= 0:
            if self.gait_current != self.P['start_gait']:
                self.get_logger().warn(
                    f'Gait owner did not confirm {self.P["start_gait"]!r} on /gait/current '
                    f'(now {self.gait_current!r}): is special_obstacles running? Exploring '
                    'in the current gait.')
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
        # [GRID] next unvisited free cell; frontiers only when none is left
        if self.start_cell(pose):
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
        blacklisted = len(frontiers) - len(ranked)
        available = len(ranked)
        ranked = [goal for goal in ranked if not self.is_deferred(goal.goal_xy)]
        self.publish_markers(ranked)
        counts = (f'{detected} detected, {detected - len(frontiers)} outside bounds, '
                  f'{blacklisted} blacklisted, {available - len(ranked)} deferred, '
                  f'{len(ranked)} eligible')

        if not ranked:
            if self.deferred_goals:
                return  # Give localization-deferred goals time to become eligible again.
            # [COVER] already covering (after a coverage goal / a pause): straight back
            if self.covering and not self.P['dry_run']:
                self.enter_cover()
                return
            # [RECORD] final retry pass already running: next one, no empty-check wait
            if (self.retry_queue is not None and not self.final_retry_done
                    and not self.P['dry_run'] and self.start_final_retry(pose, frontiers)):
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
                # [RECORD] nothing reachable is dropped silently: retry failed ones once
                if self.start_final_retry(pose, frontiers):
                    return
                # [COVER] mapped is not seen: look where the camera has not been first
                if self.P['coverage_enabled']:
                    self.get_logger().info('No reachable frontiers left: checking camera coverage (COVER).')
                    self.covering = True
                    self.cover_empty_checks = 0
                    self.enter_cover()
                    return
                self.get_logger().info('No reachable frontiers left: exploration done.')
                self.log_frontier_list('DONE')
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
        target = None
        if self.enter_mode():                                  # [ENTER]
            target = target_cell(self.cells.cell_of, best.goal_xy, pose[:2],
                                 unknown_side_xy(self.grid, self.info, best.cells),
                                 lead_m=self.P['frontier_lead_m'])
        self.get_logger().info(f'Selected highest-scoring frontier ({counts}), best first: {top}'
                               + (f'; leads into cell {target}' if target is not None else ''))
        if self.send_goal(best.goal_xy, pose[:2], is_frontier=True, target=target):
            self.frontier_rec.add(best.goal_xy, target, self.now_s())   # [RECORD]
            self.set_state('NAVIGATE')

    def do_navigate(self, pose):
        now = self.now_s()

        # 1. Finished (success or failure)?
        if self.nav.isTaskComplete():
            result = self.nav.getResult()
            if result == TaskResult.SUCCEEDED:
                if self.success_seen_at is None:
                    self.success_seen_at = now
                problem = None
                if (self.pose_health.pose_stamp <= self.goal_pose_stamp
                        or self.pose_health.odom_stamp <= self.goal_odom_stamp):
                    problem = 'Nav2 reported success without newer map-frame pose and odometry'
                else:
                    miss = math.dist(self.goal_xy, pose[:2])
                    if miss > self.P['pose_goal_tolerance_m']:
                        problem = f'Nav2 reported success but the pose is {miss:.2f} m away'
                    elif (self.goal_kind == 'cell'
                          and self.visit_cell(pose) != self.cells.cell_of(*self.goal_xy)):
                        problem = 'Nav2 reported success outside the target cell interior'
                if problem is not None:
                    if now - self.success_seen_at >= self.P['pose_result_wait_s']:
                        self.wait_for_pose(problem, retry_goal=True)
                    return
            if result == TaskResult.SUCCEEDED and self.goal_kind == 'cell':  # [GRID]
                if self.start_arrival_hold():          # let obstacle_detector look ahead
                    return
                cell = self.cells.cell_of(*self.goal_xy)
                self.record_visit(cell)
                self.failures_in_a_row = 0
                self.last_good_xy = self.goal_xy
                self.set_state('SELECT')
                return
            if result == TaskResult.SUCCEEDED and self.goal_kind == 'room':  # [ROOM]
                self.failures_in_a_row = 0
                self.last_good_xy = self.goal_xy
                self.start_room_look(pose)
                return
            if result == TaskResult.SUCCEEDED:
                # [ENTER] reached the frontier but not inside the cell it leads into: go in
                if (self.goal_is_frontier and self.enter_mode() and not self.goal_converted
                        and self.frontier_check(pose, still=False) == 'convert'):
                    return
                # [HOLD] arrival dwell first; this branch runs again (unchanged) after it
                if self.start_arrival_hold():
                    return
                self.get_logger().info(f'Goal ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) reached.')
                self.failures_in_a_row = 0
                self.last_good_xy = self.goal_xy
                # Don't come straight back if this spot is still a frontier (e.g. glass).
                if self.goal_is_frontier:
                    self.blacklist.append(self.goal_xy)
                    if getattr(self, 'goal_record_xy', None) is not None:
                        self.frontier_rec.explored(self.goal_record_xy, self.now_s())
                    if self.goal_target_cell is not None and self.visit_cell(pose) is not None:
                        self.record_visit(self.visit_cell(pose))
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
        if self.turn_hold_active:
            # A deliberate pause after turning should not count as a lack of progress.
            self.window_remaining = None
        elif remaining is not None and remaining > 0.0:
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

        # 4. Goal already explored on the way?
        #    [ENTER] (default): never cancel because it got mapped; finish only inside the
        #    target cell, else turn the goal into 'enter that cell'.
        #    Old behaviour (frontier_enter_cell false): pick a new one (no blacklist).
        if self.goal_is_frontier and now - self.last_recheck > self.P['recheck_period_s']:
            self.last_recheck = now
            if self.grid is None:
                pass
            elif self.enter_mode():
                if self.frontier_check(pose) == 'done':
                    self.nav.cancelTask()
                    self.finish_frontier(pose)
                    return
            elif not still_frontier(
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

    # ================= [ENTER] frontier -> enter the cell behind it =================

    def enter_mode(self):
        return bool(self.P.get('frontier_enter_cell')) and getattr(self, 'cells', None) is not None

    def enter_kwargs(self):
        return {'target': getattr(self, 'goal_target_cell', None),
                'converted': getattr(self, 'goal_converted', False)}

    def target_seen_frac(self):
        target = getattr(self, 'goal_target_cell', None)
        if target is None or not self.P.get('coverage_enabled'):
            return None
        half = self.cells.size / 2.0 - self.P['cell_visit_edge_margin_m']
        return cell_seen_fraction(self.current_seen(), self.grid, self.info,
                                  self.cells.centre(target), half, self.P['free_max'])

    def frontier_check(self, pose, still=None):
        """Recheck a running frontier goal: 'done', 'convert' (goal re-sent) or 'continue'."""
        target = self.goal_target_cell
        if still is None:
            still = still_frontier(self.grid, self.info, self.goal_xy, radius_m=0.4,
                                   free_max=self.P['free_max'])
        here = self.visit_cell(pose) if pose is not None else None
        step = frontier_step(still, here, target, self.target_seen_frac(),
                             self.P['frontier_seen_done_frac'], self.goal_converted)
        if step == 'convert':
            ci = self.cell_info.get(target) if getattr(self, 'cell_info', None) else None
            xy = ci.goal if ci is not None and ci.goal is not None else self.cells.centre(target)
            self.get_logger().info(
                f'Frontier ({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) '
                f'{"mapped before arrival" if not still else "reached at the edge"}: '
                f'not cancelling, going INTO cell {target} at ({xy[0]:.2f}, {xy[1]:.2f}).')
            if not self.send_goal(xy, pose[:2], is_frontier=True, kind='frontier',
                                  target=target, converted=True):
                self.set_state('SELECT')
            else:
                self.set_state('NAVIGATE')
        return step

    def finish_frontier(self, pose):
        target = self.goal_target_cell
        here = self.visit_cell(pose) if pose is not None else None
        why = 'entered' if here == target else 'camera already saw it'
        self.get_logger().info(f'Frontier done: cell {target} {why}.')
        if getattr(self, 'goal_record_xy', None) is not None:
            self.frontier_rec.explored(self.goal_record_xy, self.now_s(),
                                       f'reached: cell {target} {why}')
        if here is not None:
            self.record_visit(here)
        self.blacklist.append(self.goal_xy)
        self.failures_in_a_row = 0
        self.last_good_xy = self.goal_xy
        self.set_state('SELECT')

    # ================= [RECORD] final retry + list =================

    def start_final_retry(self, pose, detected=()):
        """Retry every failed / pending frontier once before COVER or DONE.
        True = a goal was sent or we wait for one to become due."""
        if not self.P['frontier_retry_enabled'] or self.final_retry_done or self.P['dry_run']:
            return False
        now = self.now_s()
        rec = self.frontier_rec
        if self.retry_queue is None:
            if self.bounds is not None:                 # list what was never eligible
                xmin, ymin, xmax, ymax = self.bounds
                for f in find_frontiers(self.grid, self.info,
                                        min_size_cells=self.P['min_frontier_size_cells'],
                                        free_max=self.P['free_max'],
                                        occupied_min=self.P['occupied_min'],
                                        clearance_m=self.P['goal_clearance_m']):
                    x, y = f.goal_xy
                    if not (xmin <= x <= xmax and ymin <= y <= ymax):
                        rec.note(f.goal_xy, self.cells.cell_of(x, y), now, 'failed',
                                 'outside bounds')
            self.retry_queue = [e['id'] for e in rec.retry_candidates(now)]
            self.get_logger().info(f'Final frontier retry pass: {len(self.retry_queue)} to retry '
                                   f'({rec.summary()})')
        while self.retry_queue:
            due = [i for i in self.retry_queue
                   if now - rec.entries[i]['t'] >= self.P['frontier_retry_delay_s']]
            if not due:
                self.warn_idle(f'final retry: waiting {self.P["frontier_retry_delay_s"]:.0f} s '
                               'after the last failure before retrying')
                return True
            e = rec.entries[due[0]]
            self.retry_queue.remove(due[0])
            e['retried'] = True
            xy = tuple(e['goal_xy'])
            radius = self.P['blacklist_radius_m']
            self.blacklist = [b for b in self.blacklist if math.dist(b, xy) >= radius]
            self.deferred_goals = {k: v for k, v in self.deferred_goals.items()
                                   if math.dist(k, xy) >= radius}
            target = tuple(e['cell']) if e['cell'] else None
            self.get_logger().info(f'Final retry #{e["id"]} ({xy[0]:.2f}, {xy[1]:.2f}) '
                                   f'cell {target}, was: {e["reason"]}')
            if self.send_goal(xy, pose[:2], is_frontier=True, target=target):
                rec.add(xy, target, now)
                self.set_state('NAVIGATE')
                return True
        self.final_retry_done = True
        self.log_frontier_list('end of the final retry pass')
        return False

    def log_frontier_list(self, when):
        lines = self.frontier_rec.lines()
        self.get_logger().info(f'Frontier list ({when}): {self.frontier_rec.summary()}'
                               + ''.join(f'\n  {ln}' for ln in lines))
        self.save_record()

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
        self.warn_idle('exploration DONE (resume with /exploration/enable true)')

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
                or self.goal_kind not in ('frontier', 'coverage', 'cell')):  # [GRID] cell
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
        self.log_frontier_list('DONE')
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
            if self.deferred_goals:
                return
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

    def visit_cell(self, pose):
        x, y = pose[:2]
        cell = self.cells.cell_of(x, y)
        half = self.cells.size / 2.0 - self.P['cell_visit_edge_margin_m']
        if self.cells.in_bounds(cell) and self.cells.in_visit_region(cell, x, y, half=half):
            return cell
        return None

    def record_visit(self, cell):
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            self.get_logger().info(f'Cell {cell} visited ({len(self.visited_cells)} total)')
            self.last_cells_pub = -math.inf

    def update_cells(self, pose):
        """Analyse cells when the map changed (<= 1 Hz), mark visited cells, publish overlay."""
        now = self.now_s()
        cell = self.visit_cell(pose) if pose is not None and self.grid is not None else None
        previous = self.visit_sample
        if cell is None:
            self.visit_sample = None
        else:
            stamp = self.pose_health.pose_stamp
            if previous is not None and previous[0] == cell and stamp > previous[1]:
                self.record_visit(cell)
            self.visit_sample = cell, stamp
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

    def start_cell(self, pose):
        """[GRID] Drive to the nearest free, unvisited cell inside the bounds."""
        if not self.P['grid_explore'] or self.P['dry_run'] or pose is None or not self.cell_info:
            return False
        here = self.cells.cell_of(pose[0], pose[1])
        todo = []
        for cell, ci in self.cell_info.items():
            if (ci.state != 'free' or ci.goal is None or cell in self.visited_cells
                    or cell == here and self.cells.in_visit_region(cell, pose[0], pose[1])
                    or not self.cells.in_bounds(cell)
                    or self.is_deferred(ci.goal)
                    or self.cell_attempts.get(cell, 0) >= self.P['grid_max_attempts']):
                continue
            todo.append((math.hypot(ci.goal[0] - pose[0], ci.goal[1] - pose[1]), cell, ci))
        for _, cell, ci in sorted(todo):
            self.cell_attempts[cell] = self.cell_attempts.get(cell, 0) + 1
            self.get_logger().info(
                f'GRID: going to cell {cell} ({len(todo)} unvisited free cells, attempt '
                f'{self.cell_attempts[cell]}/{self.P["grid_max_attempts"]})')
            if self.send_goal(ci.goal, pose[:2], is_frontier=False, kind='cell'):
                self.set_state('NAVIGATE')
                return True
        return False

    def start_room(self, pose, radius):
        """Go to the nearest unchecked room's doorway. True if a goal was sent."""
        if (not self.P['room_check'] or self.P['dry_run'] or pose is None
                or not self.cell_info):
            return False
        for room in self.room_candidates(pose, radius):
            if self.is_deferred(room.standpoint):
                continue
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

    def start_room_dwell(self, relook=False):
        name, _ = self.room.views[self.room_idx]
        self.room_phase = 'dwell'
        self.room_until = self.now_s() + self.P['room_dwell_s']
        self.room_look_sent = self.now_s()
        if not relook:
            self.room_relooks = 0
        self.publish_cmd()
        self.look_pub.publish(Bool(data=True))       # one VLM photo of this view (booklet J1)
        self.get_logger().info(f'ROOM {self.room.cell}: looking at {name}'
                               + (' (again: last answer was a stale photo)' if relook else ''))

    def room_view_finished(self, now):
        """[LOOK] dwell over AND (look_done for this look, or timeout)."""
        if now < self.room_until:
            return False
        if not self.P.get('room_look_wait_done', False):
            return True
        sent = self.room_look_sent
        done = self.look_done_t is not None and self.look_done_t >= sent
        if not done:
            if now - sent > self.P['room_look_timeout_s']:
                self.get_logger().warn(f'ROOM {self.room.cell}: no look_done after '
                                       f'{now - sent:.0f} s, next view')
                return True
            return False
        lat = self.scene_latency
        latency = lat[1] if lat is not None and lat[0] >= sent else None
        if (photo_is_stale(sent, self.look_done_t, latency)
                and getattr(self, 'room_relooks', 0) < self.P['room_max_relooks']):
            self.room_relooks = getattr(self, 'room_relooks', 0) + 1
            self.start_room_dwell(relook=True)
            return False
        return True

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
        if not self.room_view_finished(now):
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
        clear.header.frame_id = self.P['global_frame']
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
        cubes.color.a = 0.75
        cubes.scale.x = cubes.scale.y = 0.5
        cubes.scale.z = 0.02
        wall_lines = mk('walls', 0, Marker.LINE_LIST)
        wall_lines.scale.x = 0.06
        rooms = {r.cell for r in self.cells.rooms(self.cell_info)}
        h = self.cells.size / 2.0
        n = 1
        displayed = self.visited_cells | {cell for cell, ci in self.cell_info.items()
                                          if ci.state == 'free'}
        for cell in sorted(displayed):
            x, y = self.cells.centre(cell)
            cubes.points.append(Point(x=x, y=y, z=0.03))
            cubes.colors.append(ColorRGBA(r=0.1, g=0.8, b=0.2, a=1.0) if cell in self.visited_cells
                                else ColorRGBA(r=0.8, g=0.8, b=0.8, a=1.0))
        seen = self.current_seen() if self.P.get('coverage_enabled') else None
        for cell in sorted(displayed):
            ci = self.cell_info.get(cell)
            walls = set(ci.walls) if ci is not None else set()
            if cell in self.visited_cells and self.P.get('keep_visited_walls', True):
                self.wall_memory.setdefault(cell, set()).update(walls)   # [WALLS] keep them
                walls |= self.wall_memory[cell]
            if not walls:
                continue
            centre = ci.centre if ci is not None else self.cells.centre(cell)
            looked = set(self.room_status.get(cell, {}).get('views_done', []))
            for side in walls:                               # [WALLS] camera looked at it?
                if (cell, side) not in self.walls_seen and seen is not None and (
                        self.cells.wall_seen_frac(self.grid, self.info, seen, cell, side,
                                                  self.P['occupied_min'])
                        >= self.P.get('wall_seen_frac', 0.5)):
                    self.walls_seen.add((cell, side))
                if (cell, side) in self.walls_seen:
                    looked.add(side)
            for side in sorted(walls):
                nu, nv = SIDES[side]
                col = (ColorRGBA(r=0.1, g=0.4, b=1.0, a=0.95) if side in looked
                       else ColorRGBA(r=1.0, g=0.9, b=0.0, a=0.95))
                for t in (-1.0, 1.0):
                    du = nu * (h - 0.05) - nv * t * (h - 0.1)
                    dv = nv * (h - 0.05) + nu * t * (h - 0.1)
                    mx, my = self.cells.vec_to_map(du, dv)
                    wall_lines.points.append(Point(x=centre[0] + mx, y=centre[1] + my, z=0.05))
                    wall_lines.colors.append(col)
            if cell in rooms and ci is not None:
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
        if wall_lines.points:
            arr.markers.append(wall_lines)
        # [RECORD] frontier status: explored green, failed red, pending blue
        colours = {'explored': (0.1, 0.8, 0.2), 'failed': (0.9, 0.1, 0.1),
                   'pending': (0.2, 0.4, 1.0)}
        for e in getattr(self, 'frontier_rec', FrontierRecord()).entries:
            m = mk('frontier_status', e['id'], Marker.SPHERE)
            m.pose.position.x, m.pose.position.y = e['goal_xy']
            m.pose.position.z = 0.15
            m.scale.x = m.scale.y = m.scale.z = 0.22
            r, g_, b = colours.get(e['status'], (0.5, 0.5, 0.5))
            m.color = ColorRGBA(r=r, g=g_, b=b, a=0.9)
            arr.markers.append(m)
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
