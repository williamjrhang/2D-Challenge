#!/usr/bin/env python3
"""Mission controller: the "boss" that decides which mode the robot is in.

Only ONE thing drives the robot at a time. This node decides which:

    Priority (highest first), like our behaviour tree:
      1. TRAVERSE     stairs / rubble / curtain ahead  -> special obstacle node drives
      2. INVESTIGATE  new area / person / QR seen      -> robot stops, perception looks
      3. EXPLORE      nothing special                  -> explorer + Nav2 drive
      4. GO_TO_EXIT   tasks done or nothing left       -> Nav2 drives to the exit
      then LAUNCH -> DONE

Inputs (who sends them)
  /perception/event      std_msgs/String   perception (fast CV or VLM), e.g.
                                           "stairs_ahead", "rubble_ahead", "curtain_ahead",
                                           "new_area", "person", "qr"
  /perception/look_done  std_msgs/Bool     perception: finished looking (investigate)
  /special/done          std_msgs/String   special obstacle node: "success" or "failed"
  /exploration/done      std_msgs/Bool     explorer: nothing left to explore
  /mission/tasks_done    std_msgs/Bool     semantic map: every mission task found
  /mission/exit_pose     geometry_msgs/PointStamped  perception: where the exit is

Outputs
  /exploration/enable    std_msgs/Bool     pause / resume the explorer
  /special/start         std_msgs/String   "stairs", "rubble" or "curtain"
  /perception/look       std_msgs/Bool     ask perception to look now (investigate)
  /launch_drone          std_msgs/Bool     handoff to the drone team
  /mission/state         std_msgs/String   current mode, for RViz / logging

Every mode has a timeout, so the robot can never get stuck waiting.
You can test all of this WITHOUT perception, by publishing the inputs by hand
(see README).
"""

import json
import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

from lite3_explore_pose.crossing import approach_pose, far_side
from lite3_explore_pose.investigate import LookCooldown, photo_is_stale
from lite3_explore_pose.mission_config import Arena, load_mission

# perception event -> which special behaviour to start
SPECIAL_EVENTS = {
    'stairs_ahead': 'stairs',
    'rubble_ahead': 'rubble',
    'curtain_ahead': 'curtain',
    'potholes_ahead': 'potholes',
}
# perception events that make the robot stop and look
LOOK_EVENTS = {'new_area', 'person', 'qr'}
# events we know about but do not act on yet (no gait crosses potholes; Nav2 should avoid them)
IGNORED_EVENTS = set()


class MissionController(Node):

    def __init__(self):
        super().__init__('mission_controller')

        p = self.declare_parameter
        p('use_nav2', True)                 # False: skip Nav2 (for testing logic only)
        p('require_localization_ready', False)
        p('global_frame', 'map')
        p('tick_hz', 5.0)
        p('mission_file', '')               # config/mission.yaml (time limit, final cell)
        p('look_every_s', 0.0)              # regular stop-and-look (0 = off: the VLM now asks
                                            # in the background while moving)
        p('investigate_timeout_s', 20.0)    # give up looking after this
        p('traverse_timeout_s', 150.0)      # backup only: keep ABOVE special_obstacles'
                                            # stairs_max_time_s (120) + backoff + restore,
                                            # else 'stop' exits AI mode on the steps
        p('exit_timeout_s', 120.0)
        p('look_cooldown_s', 5.0)           # ignore new look events right after a look
        # APPROACH: drive on purpose to obstacles the camera saw (the lidar sees the curtain /
        # rubble as a wall, so the explorer never plans a path through them).
        # Only rubble (22/11: YOLO-World 'curtains' near the start sent the robot back 3x, all
        # 'not confirmed up close'; a false 'stairs' would switch on AI mode). A real curtain or
        # stairs is still crossed when the robot is right in front of it.
        p('approach_kinds', ['rubble'])
        p('approach_min_sightings', 3)     # seen this many times before we drive there
        p('approach_merge_m', 0.8)         # sightings closer than this = same obstacle
        p('approach_standoff_m', 0.6)      # stop this far before it (base_link; nose is ~0.3 m ahead)
        p('approach_timeout_s', 40.0)
        p('approach_wait_s', 6.0)          # wait this long at the spot for the close-range trigger
        # Position-jump guard (live 20/11: Fast-LIO jumped tens of metres, goals 32 m away,
        # messy map). SLAM corrections of ~1 m are normal; real breakage jumped 20+ m. More than pose_jump_m between two ticks (0.2 s)
        # = broken tracking -> stop everything. Restart the robot launch. 0 = off.
        p('pose_jump_m', 3.0)                # 1.0 fired on a normal 1.0 m SLAM correction (20/11)
        # Issue 1: a jump between pose_jump_m and pose_jump_hard_m no longer ends the run.
        # LOCALIZE: explorer paused, robot still, until localization has been ready and
        # jump-free for pose_jump_recover_s, then EXPLORE again. Bigger jumps (20/11: tens
        # of metres) still stop everything (DONE). pose_jump_recover_s 0 = old: always DONE.
        p('pose_jump_hard_m', 8.0)
        p('pose_jump_recover_s', 10.0)
        # Issue 5: stop and let scene understanding finish (vlm_watcher unchanged).
        p('scene_investigate', True)        # /perception/room answer -> INVESTIGATE (new cell)
        p('investigate_settle_s', 1.0)      # stand still this long before /perception/look
        p('max_relooks', 1)                 # extra looks when the answered photo was stale
        p('stale_margin_s', 0.3)
        p('look_cell_cooldown_s', 60.0)     # no second stop-and-look in the same cell
        p('qr_wait_place', True)            # QR look: stay until qr_reader sent the code
        # Issue 7: curtain APPROACH (approach_kinds must include 'curtain') and the
        # passable-curtain record (portal) so the area behind it is not given up.
        p('curtain_min_sightings', 4)       # curtain needs more sightings than rubble
        p('curtain_standoff_m', 0.5)        # stand this far before it, square to the doorway
        p('portal_return', True)            # explored everything on this side -> cross back
        p('portal_max_crossings', 3)
        g = lambda n: self.get_parameter(n).value
        self.P = {n: g(n) for n in [
            'use_nav2', 'require_localization_ready', 'global_frame', 'tick_hz', 'investigate_timeout_s',
            'traverse_timeout_s', 'exit_timeout_s', 'look_cooldown_s', 'look_every_s',
            'approach_kinds', 'approach_min_sightings', 'approach_merge_m',
            'approach_standoff_m', 'approach_timeout_s', 'approach_wait_s', 'pose_jump_m',
            'pose_jump_hard_m', 'pose_jump_recover_s', 'scene_investigate',
            'investigate_settle_s', 'max_relooks', 'stale_margin_s', 'look_cell_cooldown_s',
            'qr_wait_place', 'curtain_min_sightings', 'curtain_standoff_m', 'portal_return',
            'portal_max_crossings']}

        cfg = load_mission(g('mission_file'))
        self.arena = Arena(cfg)
        self.time_limit = float(cfg.get('time_limit_s', 0))
        self.exit_margin = float(cfg.get('exit_margin_s', 0))

        # ---- inputs ----
        self.create_subscription(String, '/perception/event', self.on_event, 10)
        self.create_subscription(Bool, '/perception/look_done', self.on_look_done, 10)
        self.create_subscription(String, '/special/done', self.on_special_done, 10)
        self.create_subscription(Bool, '/exploration/done', self.on_explore_done, 10)
        self.create_subscription(Bool, '/mission/tasks_done', self.on_tasks_done, 10)
        self.create_subscription(PointStamped, '/mission/exit_pose', self.on_exit_pose, 10)
        self.localization_ready_at = None
        if self.P['require_localization_ready']:
            self.create_subscription(Bool, '/exploration/localization_ready',
                                     self.on_localization_ready, 1)
        self.create_subscription(String, '/perception/obstacle_seen', self.on_obstacle_seen, 10)
        # [LOOK] issue 5: scene answers (vlm_watcher) and QR placements (qr_reader)
        self.create_subscription(String, '/perception/room', self.on_room, 10)
        self.create_subscription(String, '/perception/scene', self.on_scene, 10)
        self.create_subscription(String, '/perception/found', self.on_found, 50)
        # Issue 6: special_obstacles says AI mode is still on outside a crossing (stopped ON
        # the stairs). Then nothing may drive the robot until a person sends exit_ai.
        self.ai_active = False
        self.create_subscription(Bool, '/special/ai_active', self.on_ai_active,
                                 QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- outputs ----
        enable_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.explore_pub = self.create_publisher(Bool, '/exploration/enable', enable_qos)
        self.special_pub = self.create_publisher(String, '/special/start', 10)
        self.look_pub = self.create_publisher(Bool, '/perception/look', 10)
        self.launch_pub = self.create_publisher(Bool, '/launch_drone', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', enable_qos)

        self.nav = None
        if self.P['use_nav2']:
            from nav2_simple_commander.robot_navigator import BasicNavigator
            self.nav = BasicNavigator()

        # ---- memory ----
        self.state = 'EXPLORE'
        self.state_since = self.now_s()
        self.pending_special = None     # e.g. 'stairs', set by a perception event
        self.current_special = None
        self.waiting_for_exit = False
        self.pending_look = False
        self.look_done = False
        self.special_result = None
        self.explore_done = False
        self.tasks_done = False
        # Final cell from mission.yaml (arena coords) -> map. /mission/exit_pose can override.
        self.exit_xy = (self.arena.arena_to_map(*self.arena.final)
                        if self.arena.final is not None else None)
        if self.exit_xy is None:
            self.get_logger().warn('final_cell_centre not set: robot will explore and then '
                                   'stop where it is (no drive to a final cell).')
        self.start_time = self.now_s()
        self.out_of_time = False
        self.last_look_end = self.now_s()
        self.last_special_end = -1e9
        self.last_special_kind = None
        self.exit_goal_sent = False
        self.obstacles = []          # {kind, x, y, rx, ry, n, status: new | done | failed}
        self.target = None
        self.approach_sent = False
        self.arrived_at = None
        self.traverse_xy = None
        self.last_xy = None
        self.stable_since = None     # LOCALIZE: ready and jump-free since
        # [LOOK] issue 5
        self.looks = LookCooldown(self.P['look_cell_cooldown_s'])
        self.look_reason = None      # 'scene' | 'qr' | 'person' | 'new_area' | 'scheduled'
        self.look_cell = None
        self.look_sent_t = None
        self.relooks = 0
        self.last_latency = None     # (arrival time, latency_s) of the latest scene answer
        self.qr_event_t = None
        self.last_qr_found_t = None
        # [PORTAL] issue 7: crossed curtains {xy, from, to, crossings, status}
        self.portals = []
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.portal_pub = self.create_publisher(String, '/exploration/portal_crossed', latched)

        self.create_timer(1.0 / self.P['tick_hz'], self.tick)
        self.enter('EXPLORE')

    # ================= inputs =================

    def on_localization_ready(self, msg):
        self.localization_ready_at = self.now_s() if msg.data else None

    def localization_ready(self):
        if not self.P['require_localization_ready']:
            return True
        return (self.localization_ready_at is not None
                and 0.0 <= self.now_s() - self.localization_ready_at <= 2.0)

    def on_event(self, msg):
        event = msg.data.strip()
        if event in SPECIAL_EVENTS:
            if not self.localization_ready():
                return
            # Ignore while already crossing (detector keeps firing on the obstacle) and,
            # just after a crossing, the SAME kind only (else it would cross it again). A
            # different terrain right behind it is crossed in its own gait.
            kind = SPECIAL_EVENTS[event]
            recent = self.now_s() - self.last_special_end < 10.0
            if self.state == 'TRAVERSE' or (recent and kind == self.last_special_kind):
                return
            self.pending_special = kind
            self.get_logger().info(f'Event: {event} -> will traverse {kind}')
        elif event in LOOK_EVENTS:
            if event == 'qr':
                self.qr_event_t = self.now_s()
            cell = self.current_cell()
            if not self.looks.may_look(cell, self.now_s()):
                self.get_logger().info(f'Event: {event} in cell {cell}: looked there '
                                       f'< {self.P["look_cell_cooldown_s"]:.0f} s ago, skipping')
            elif self.now_s() - self.last_look_end > self.P['look_cooldown_s'] or event == 'qr':
                self.pending_look = event
                self.get_logger().info(f'Event: {event} -> will investigate (cell {cell})')
        elif event in IGNORED_EVENTS:
            self.get_logger().warn(f'Event: {event} (no action yet)')
        else:
            self.get_logger().warn(f'Unknown perception event: "{event}"')

    def on_ai_active(self, msg):
        if msg.data != self.ai_active:
            (self.get_logger().error if msg.data else self.get_logger().warn)(
                f'special_obstacles: AI mode {"STILL ON (stairs)" if msg.data else "off"}')
        self.ai_active = msg.data

    def do_hold_ai(self):
        if self.ai_active:
            self.get_logger().error(
                'HOLD_AI: AI mode is still on (robot stopped on the stairs). Walk it off with '
                'the joystick, then: ros2 topic pub --once /special/start '
                'std_msgs/msg/String "{data: exit_ai}"', throttle_duration_sec=5.0)
            return
        self.get_logger().warn('HOLD_AI: AI mode off again, back to EXPLORE.')
        self.explore_done = False
        self.enter('GO_TO_EXIT' if self.out_of_time else 'EXPLORE')

    def on_look_done(self, msg):
        self.look_done = msg.data
        if msg.data:
            self.look_done_t = self.now_s()

    # ---------------- [LOOK] issue 5 ----------------
    def current_cell(self):
        here = self.robot_xy()
        if here is None:
            return None
        return self.arena.cell_index(*self.arena.map_to_arena(*here))

    def on_room(self, msg):
        """Scene answer (vlm_watcher, already checked against min_confidence)."""
        scene = msg.data.strip()
        if (not self.P.get('scene_investigate') or scene in ('', 'unknown')
                or self.state != 'EXPLORE' or self.pending_look):
            return
        cell = self.current_cell()
        if self.looks.may_scene_look(cell, self.now_s()):
            self.pending_look = 'scene'
            self.get_logger().info(f'Scene answer {scene!r} in new cell {cell}: will stop and '
                                   'investigate')

    def on_scene(self, msg):
        try:
            latency = json.loads(msg.data).get('latency_s')
        except (ValueError, AttributeError):
            return
        if latency is not None:
            self.last_latency = (self.now_s(), float(latency))

    def on_found(self, msg):
        try:
            if json.loads(msg.data).get('task_type') == 'qr':
                self.last_qr_found_t = self.now_s()
        except (ValueError, AttributeError):
            pass

    def send_look(self):
        self.look_done = False
        self.look_sent_t = self.now_s()
        self.look_pub.publish(Bool(data=True))

    def on_special_done(self, msg):
        self.special_result = msg.data.strip()

    def on_explore_done(self, msg):
        self.explore_done = msg.data

    def on_tasks_done(self, msg):
        self.tasks_done = msg.data

    def on_exit_pose(self, msg):
        self.exit_xy = (msg.point.x, msg.point.y)
        self.get_logger().info(f'Exit at ({msg.point.x:.2f}, {msg.point.y:.2f})')

    def on_obstacle_seen(self, msg):
        if not self.localization_ready():
            return
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        if s.get('kind') not in self.P['approach_kinds']:
            return
        if s['kind'] == 'curtain':
            self.get_logger().info(f'CURTAIN sighting at ({s["x"]:.2f}, {s["y"]:.2f}), wrinkle '
                                   f'{s.get("wrinkle_mm")}', throttle_duration_sec=2.0)
        for o in self.obstacles:
            if (o['kind'] == s['kind']
                    and math.hypot(o['x'] - s['x'], o['y'] - s['y']) < self.P['approach_merge_m']):
                if o['status'] == 'new':
                    k = o['n']
                    o['x'] = (o['x'] * k + s['x']) / (k + 1)
                    o['y'] = (o['y'] * k + s['y']) / (k + 1)
                    o['rx'], o['ry'] = s['rx'], s['ry']
                    o['n'] = k + 1
                return
        self.obstacles.append({**s, 'n': 1, 'status': 'new'})
        self.get_logger().info(f'Obstacle seen: {s["kind"]} at ({s["x"]:.2f}, {s["y"]:.2f})')

    def robot_xy(self):
        try:
            t = self.tf_buffer.lookup_transform(self.P['global_frame'], 'base_link', Time())
        except TransformException:
            return None
        return t.transform.translation.x, t.transform.translation.y

    def min_sightings(self, kind):
        return (self.P.get('curtain_min_sightings', self.P['approach_min_sightings'])
                if kind == 'curtain' else self.P['approach_min_sightings'])

    def pick_target(self):
        if self.nav is None or self.out_of_time:
            return None
        here = self.robot_xy()
        cands = [o for o in self.obstacles
                 if o['status'] == 'new' and o['n'] >= self.min_sightings(o['kind'])]
        if not cands or here is None:
            return None
        return min(cands, key=lambda o: math.hypot(o['x'] - here[0], o['y'] - here[1]))

    def mark_obstacles(self, kind, xy, status, radius=2.0):
        if xy is None:
            return
        for o in self.obstacles:
            if (o['kind'] == kind and o['status'] == 'new'
                    and math.hypot(o['x'] - xy[0], o['y'] - xy[1]) < radius):
                o['status'] = status

    # ================= helpers =================

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def time_in_state(self):
        return self.now_s() - self.state_since

    def enter(self, state):
        """Switch mode and do the one-off actions for the new mode."""
        old = getattr(self, 'state', None)
        self.state = state
        self.state_since = self.now_s()
        self.get_logger().info(f'MODE: {old} -> {state}')
        self.state_pub.publish(String(data=state))

        # Explorer only runs in EXPLORE; every other mode pauses it.
        self.explore_pub.publish(Bool(data=(state == 'EXPLORE')))

        if state == 'TRAVERSE':
            self.traverse_xy = self.robot_xy()
            self.special_result = None
            self.special_pub.publish(String(data=self.current_special))
        elif state == 'APPROACH':
            self.approach_sent = False
            self.arrived_at = None
        elif state == 'INVESTIGATE':
            self.look_done = False
            self.look_sent_t = None                # sent after investigate_settle_s (still)
            self.relooks = 0
            self.look_reason = self.look_reason or 'scheduled'
            self.look_cell = self.current_cell()
            self.looks.mark(self.look_cell, self.now_s(), scene=self.look_reason == 'scene')
            self.get_logger().info(f'INVESTIGATE ({self.look_reason}) in cell {self.look_cell}')
        elif state == 'GO_TO_EXIT':
            self.exit_goal_sent = False
        elif state == 'LAUNCH':
            self.launch_pub.publish(Bool(data=True))

    # ================= main loop =================

    def stop_for_localization(self, why):
        self.get_logger().error(f'{why}. Stopping the mission. Restart robot bringup and explorer.')
        if self.state == 'TRAVERSE':
            self.special_pub.publish(String(data='stop'))
        self.cancel_nav()
        self.enter('DONE')

    def tick(self):
        ready = self.localization_ready()
        # Issue 6: AI still on outside a crossing -> nobody drives (explorer paused).
        if (getattr(self, 'ai_active', False)
                and self.state not in ('TRAVERSE', 'DONE', 'HOLD_AI')):
            self.cancel_nav()
            self.enter('HOLD_AI')
            return
        # Position jumped (broken SLAM / odometry)? Stop instead of chasing garbage goals.
        if ready and self.P['pose_jump_m'] > 0 and self.state != 'DONE':
            here = self.robot_xy()
            if here is not None and self.last_xy is not None:
                jump = math.hypot(here[0] - self.last_xy[0], here[1] - self.last_xy[1])
                if jump > self.P['pose_jump_m']:
                    self.last_xy = here
                    if self.jump_recoverable(jump):
                        self.stable_since = None
                        if self.state != 'LOCALIZE':
                            self.get_logger().error(
                                f'Robot position jumped {jump:.1f} m in one tick: pausing in '
                                f'LOCALIZE until it is stable for '
                                f'{self.P["pose_jump_recover_s"]:.0f} s.')
                            self.cancel_nav()
                            self.enter('LOCALIZE')
                        return
                    self.stop_for_localization(f'Robot position jumped {jump:.1f} m in one tick')
                    return
            if here is not None:
                self.last_xy = here
        if self.state != 'DONE' and not ready:
            self.last_xy = None
            self.stable_since = None             # LOCALIZE: needs a full stable period
            self.pending_special = None
            self.obstacles = [o for o in self.obstacles if o['status'] != 'new']
            if self.state in ('TRAVERSE', 'APPROACH', 'GO_TO_EXIT'):
                self.stop_for_localization(f'Localization unavailable during {self.state}')
            else:
                self.get_logger().info('Waiting for explorer localization readiness before mission motion.',
                                       throttle_duration_sec=5.0)
            return
        # Overall time limit: leave enough time to reach the final cell.
        elapsed = self.now_s() - self.start_time
        if (self.time_limit > 0 and not self.out_of_time
                and elapsed > self.time_limit - self.exit_margin
                and self.state in ('EXPLORE', 'INVESTIGATE', 'APPROACH')):
            self.out_of_time = True
            self.get_logger().warn(f'{elapsed:.0f} s used: time to head for the final cell.')
            self.cancel_nav()
            self.enter('GO_TO_EXIT')
            return

        # Priority 1 and 2 can interrupt EXPLORE (and GO_TO_EXIT, APPROACH).
        if self.state in ('EXPLORE', 'GO_TO_EXIT', 'APPROACH'):
            if self.pending_special:
                self.current_special = self.pending_special
                self.pending_special = None
                self.cancel_nav()
                self.enter('TRAVERSE')
                return
            if self.pending_look and self.state == 'EXPLORE':
                self.look_reason = (self.pending_look if isinstance(self.pending_look, str)
                                    else 'event')
                self.pending_look = False
                self.enter('INVESTIGATE')
                return

        getattr(self, f'do_{self.state.lower()}')()

    def jump_recoverable(self, jump):
        return (self.P.get('pose_jump_recover_s', 0) > 0
                and jump <= self.P.get('pose_jump_hard_m', 0)
                and self.state in ('EXPLORE', 'INVESTIGATE', 'APPROACH', 'GO_TO_EXIT',
                                   'LOCALIZE'))

    def do_localize(self):
        """Robot still (explorer paused, turn_hold zero), until the pose is stable again."""
        now = self.now_s()
        if self.stable_since is None:
            self.stable_since = now
        left = self.P['pose_jump_recover_s'] - (now - self.stable_since)
        if left > 0:
            self.get_logger().warn(f'LOCALIZE: waiting for a stable pose ({left:.1f} s more)',
                                   throttle_duration_sec=2.0)
            return
        self.get_logger().warn('LOCALIZE: pose stable again, back to EXPLORE.')
        self.last_xy = None
        self.enter('GO_TO_EXIT' if self.out_of_time else 'EXPLORE')

    def do_explore(self):
        target = self.pick_target()
        if target is not None:
            self.target = target
            self.get_logger().info(
                f'Going to {target["kind"]} at ({target["x"]:.2f}, {target["y"]:.2f})')
            self.enter('APPROACH')
            return
        if not (self.tasks_done or self.explore_done):
            # Stop now and then so the scene VLM can name the room / spot hazards.
            if (self.P['look_every_s'] > 0
                    and self.now_s() - self.last_look_end > self.P['look_every_s']):
                self.get_logger().info('Scheduled look-around is due; pausing exploration.')
                self.look_reason = 'scheduled'
                self.enter('INVESTIGATE')
            return
        if self.explore_done and not self.tasks_done and self.portal_back():
            return
        if self.exit_xy is not None or self.explore_done:
            why = 'All mission tasks done' if self.tasks_done else 'Nothing left to explore'
            self.get_logger().info(f'{why}: heading for the exit.')
            self.enter('GO_TO_EXIT')
        elif not self.waiting_for_exit:
            # Tasks done but exit not seen yet: keep exploring until perception finds it.
            self.get_logger().info('Tasks done, still looking for the exit.')
            self.waiting_for_exit = True

    # ---------------- [PORTAL] issue 7 ----------------
    def record_portal(self):
        """A crossed curtain stays a wall on the map: remember it as passable and tell the
        explorer which side is now open (it forgets failures over there)."""
        here, start = self.robot_xy(), self.traverse_xy
        if here is None or start is None:
            return
        t = self.target if self.target is not None and self.target['kind'] == 'curtain' else None
        xy = (t['x'], t['y']) if t is not None else ((here[0] + start[0]) / 2,
                                                     (here[1] + start[1]) / 2)
        idx = t.get('portal') if t is not None else None
        if idx is None:
            idx = next((i for i, p in enumerate(self.portals)
                        if math.dist(p['xy'], xy) < self.P['approach_merge_m']), None)
        if idx is None:
            self.portals.append({'xy': list(xy), 'crossings': 0, 'status': 'ok'})
            idx = len(self.portals) - 1
        p = self.portals[idx]
        p.update(**{'from': list(start), 'to': list(here)})
        p['crossings'] += 1
        self.get_logger().warn(f'CURTAIN: passable curtain #{idx} at ({xy[0]:.2f}, {xy[1]:.2f}) '
                               f'recorded, crossing {p["crossings"]}; the explorer re-opens the '
                               'cells behind it')
        self.portal_pub.publish(String(data=json.dumps({'id': idx, **p})))

    def portal_back(self):
        """Nothing left on this side: walk back through a curtain we came through."""
        if not self.P.get('portal_return') or self.nav is None:
            return False
        here = self.robot_xy()
        if here is None:
            return False
        for i, p in enumerate(self.portals):
            if (p['status'] != 'ok' or p['crossings'] >= self.P['portal_max_crossings']
                    or not far_side(here, p['xy'], p['from'], p['to'])):
                continue
            self.target = {'kind': 'curtain', 'x': p['xy'][0], 'y': p['xy'][1],
                           'rx': here[0], 'ry': here[1], 'n': 99, 'status': 'new',
                           'portal': i}
            self.explore_done = False
            self.get_logger().warn(f'CURTAIN: nothing left on this side, going back through '
                                   f'curtain #{i} at ({p["xy"][0]:.2f}, {p["xy"][1]:.2f})')
            self.enter('APPROACH')
            return True
        return False

    def do_approach(self):
        from nav2_simple_commander.robot_navigator import TaskResult
        o = self.target
        if self.time_in_state() < 1.0:        # let the explorer cancel its own goal first
            return
        if not self.approach_sent:
            curtain = o['kind'] == 'curtain'
            sx = (self.P.get('curtain_standoff_m', 0.5) if curtain
                  else self.P['approach_standoff_m'])
            # direction we first saw it from; curtain: square to the doorway (issue 7)
            gx, gy, yaw = approach_pose((o['x'], o['y']), (o['rx'], o['ry']), sx,
                                        self.arena, square=curtain)
            goal = PoseStamped()
            goal.header.frame_id = self.P['global_frame']
            goal.header.stamp = self.nav.get_clock().now().to_msg()
            goal.pose.position.x = gx
            goal.pose.position.y = gy
            goal.pose.orientation.z = math.sin(yaw / 2)
            goal.pose.orientation.w = math.cos(yaw / 2)
            if self.nav.goToPose(goal) is False:
                return self.approach_failed('Nav2 rejected the goal')
            self.approach_sent = True
            self.get_logger().info(
                f'Nav2 accepted approach to {o["kind"]} at '
                f'({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})')
            return
        if self.arrived_at is None:
            if self.nav.isTaskComplete():
                result = self.nav.getResult()
                if result != TaskResult.SUCCEEDED:
                    return self.approach_failed(f'Nav2 result {result}')
                self.arrived_at = self.now_s()
                self.get_logger().info(f'At the {o["kind"]}, waiting for the close-range check.')
            elif self.time_in_state() > self.P['approach_timeout_s']:
                self.cancel_nav()
                return self.approach_failed('timed out')
            return
        # Standing in front of it: the detector should fire -> tick() switches to TRAVERSE.
        if self.now_s() - self.arrived_at > self.P['approach_wait_s']:
            self.approach_failed('not confirmed up close (false detection?)')

    def approach_failed(self, why):
        self.get_logger().warn(f'{"CURTAIN: " if self.target["kind"] == "curtain" else ""}'
                               f'Approach to {self.target["kind"]} failed: {why}')
        if self.target.get('portal') is not None:
            self.portals[self.target['portal']]['status'] = 'failed'
        self.target['status'] = 'failed'
        self.target = None
        self.enter('EXPLORE')

    def do_investigate(self):
        """Stand still -> look -> (stale photo? look again) -> (QR? until placed) -> EXPLORE.
        investigate_timeout_s always ends it (desktop / link down never blocks the robot)."""
        if self.time_in_state() > self.P['investigate_timeout_s']:
            self.get_logger().warn(f'Investigate ({self.look_reason}) timed out after '
                                   f'{self.P["investigate_timeout_s"]:.0f} s, carrying on.')
            self.finish_look()
            return
        if self.look_sent_t is None:
            if self.time_in_state() >= self.P.get('investigate_settle_s', 0.0):
                self.send_look()
            return
        if not self.look_done:
            return
        lat = self.last_latency
        latency = lat[1] if lat is not None and lat[0] >= self.look_sent_t else None
        done_t = getattr(self, 'look_done_t', self.now_s())
        if (photo_is_stale(self.look_sent_t, done_t, latency, self.P.get('stale_margin_s', 0.3))
                and self.relooks < self.P.get('max_relooks', 1)):
            self.relooks += 1
            self.get_logger().info(f'Look answered with a photo from ~{done_t - latency:.1f} '
                                   f'(look sent {self.look_sent_t:.1f}): stale, looking again')
            self.send_look()
            return
        if (self.look_reason == 'qr' and self.P.get('qr_wait_place', True)
                and (self.last_qr_found_t is None or self.qr_event_t is None
                     or self.last_qr_found_t < self.qr_event_t)):
            self.get_logger().info('Waiting for qr_reader to place and send the QR code',
                                   throttle_duration_sec=2.0)
            return
        self.finish_look()

    def finish_look(self):
        self.last_look_end = self.now_s()
        self.pending_look = False   # drop looks that came in while looking
        self.look_reason = None
        self.enter('EXPLORE')

    def do_traverse(self):
        if self.special_result is not None:
            self.get_logger().info(f'{self.current_special}: {self.special_result}')
        elif self.time_in_state() > self.P['traverse_timeout_s']:
            self.get_logger().warn(
                f'{self.current_special} timed out: ask marshal for manual help if stuck.')
            self.special_pub.publish(String(data='stop'))
        else:
            return
        self.last_special_end = self.now_s()
        self.last_special_kind = self.current_special
        ok = self.special_result == 'success'
        if self.current_special == 'curtain':
            self.get_logger().warn(f'CURTAIN: crossing {self.special_result or "timed out"}')
            if ok:
                self.record_portal()
        self.mark_obstacles(self.current_special, self.traverse_xy, 'done' if ok else 'failed')
        if self.target is not None and self.target['kind'] == self.current_special:
            self.target['status'] = 'done' if ok else 'failed'
        self.target = None
        # A new area may be open now (e.g. top of the stairs), so explore again.
        # Out of time: keep heading for the exit (the time check only fires once).
        self.explore_done = False
        self.enter('GO_TO_EXIT' if self.out_of_time else 'EXPLORE')

    def do_go_to_exit(self):
        if self.exit_xy is None:
            # Everything explored but the exit was never seen: wait, then give up.
            if self.time_in_state() > self.P['exit_timeout_s']:
                self.get_logger().error('Exit never found. Stopping.')
                self.enter('DONE')
            return

        if self.nav is None:          # logic test mode: pretend we arrived
            self.enter('LAUNCH')
            return

        if not self.exit_goal_sent:
            goal = PoseStamped()
            goal.header.frame_id = self.P['global_frame']
            goal.header.stamp = self.nav.get_clock().now().to_msg()
            goal.pose.position.x, goal.pose.position.y = self.exit_xy
            goal.pose.orientation.w = 1.0   # TODO: face the launch target
            self.nav.goToPose(goal)
            self.exit_goal_sent = True
            return

        if self.nav.isTaskComplete():
            self.enter('LAUNCH')
        elif self.time_in_state() > self.P['exit_timeout_s']:
            self.get_logger().error('Could not reach the exit in time.')
            self.cancel_nav()
            self.enter('DONE')

    def do_launch(self):
        # TODO: align to the launch heading before triggering (agree with drone team).
        self.enter('DONE')

    def do_done(self):
        pass

    def cancel_nav(self):
        if self.nav is None:
            return
        if ((self.state == 'GO_TO_EXIT' and self.exit_goal_sent)
                or (self.state == 'APPROACH' and self.approach_sent)):
            self.nav.cancelTask()


def main(args=None):
    # Handle Ctrl+C ourselves so the stop command is sent before shutdown.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.explore_pub.publish(Bool(data=False))
        if node.nav is not None:
            node.nav.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
