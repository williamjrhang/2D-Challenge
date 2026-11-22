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
from rclpy.time import Time
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformException, TransformListener

from lite3_explore.mission_config import Arena, load_mission

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
        g = lambda n: self.get_parameter(n).value
        self.P = {n: g(n) for n in [
            'use_nav2', 'global_frame', 'tick_hz', 'investigate_timeout_s',
            'traverse_timeout_s', 'exit_timeout_s', 'look_cooldown_s', 'look_every_s',
            'approach_kinds', 'approach_min_sightings', 'approach_merge_m',
            'approach_standoff_m', 'approach_timeout_s', 'approach_wait_s', 'pose_jump_m']}

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
        self.create_subscription(String, '/perception/obstacle_seen', self.on_obstacle_seen, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ---- outputs ----
        self.explore_pub = self.create_publisher(Bool, '/exploration/enable', 10)
        self.special_pub = self.create_publisher(String, '/special/start', 10)
        self.look_pub = self.create_publisher(Bool, '/perception/look', 10)
        self.launch_pub = self.create_publisher(Bool, '/launch_drone', 10)
        self.state_pub = self.create_publisher(String, '/mission/state', 10)

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

        self.create_timer(1.0 / self.P['tick_hz'], self.tick)
        self.enter('EXPLORE')

    # ================= inputs =================

    def on_event(self, msg):
        event = msg.data.strip()
        if event in SPECIAL_EVENTS:
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
            if self.now_s() - self.last_look_end > self.P['look_cooldown_s']:
                self.pending_look = True
                self.get_logger().info(f'Event: {event} -> will investigate')
        elif event in IGNORED_EVENTS:
            self.get_logger().warn(f'Event: {event} (no action yet)')
        else:
            self.get_logger().warn(f'Unknown perception event: "{event}"')

    def on_look_done(self, msg):
        self.look_done = msg.data

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
        try:
            s = json.loads(msg.data)
        except ValueError:
            return
        if s.get('kind') not in self.P['approach_kinds']:
            return
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

    def pick_target(self):
        if self.nav is None or self.out_of_time:
            return None
        here = self.robot_xy()
        cands = [o for o in self.obstacles
                 if o['status'] == 'new' and o['n'] >= self.P['approach_min_sightings']]
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
            self.look_pub.publish(Bool(data=True))
        elif state == 'GO_TO_EXIT':
            self.exit_goal_sent = False
        elif state == 'LAUNCH':
            self.launch_pub.publish(Bool(data=True))

    # ================= main loop =================

    def tick(self):
        # Position jumped (broken SLAM / odometry)? Stop instead of chasing garbage goals.
        if self.P['pose_jump_m'] > 0 and self.state != 'DONE':
            here = self.robot_xy()
            if here is not None and self.last_xy is not None:
                jump = math.hypot(here[0] - self.last_xy[0], here[1] - self.last_xy[1])
                if jump > self.P['pose_jump_m']:
                    self.get_logger().error(
                        f'Robot position jumped {jump:.1f} m in one tick: tracking (Fast-LIO / '
                        f'SLAM) is broken. STOPPING. Ctrl+C, lite3_manual, restart the robot launch.')
                    if self.state == 'TRAVERSE':
                        self.special_pub.publish(String(data='stop'))
                    self.cancel_nav()
                    self.last_xy = None
                    self.enter('DONE')
                    return
            if here is not None:
                self.last_xy = here
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
                self.pending_look = False
                self.enter('INVESTIGATE')
                return

        getattr(self, f'do_{self.state.lower()}')()

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
                self.enter('INVESTIGATE')
            return
        if self.exit_xy is not None or self.explore_done:
            why = 'All mission tasks done' if self.tasks_done else 'Nothing left to explore'
            self.get_logger().info(f'{why}: heading for the exit.')
            self.enter('GO_TO_EXIT')
        elif not self.waiting_for_exit:
            # Tasks done but exit not seen yet: keep exploring until perception finds it.
            self.get_logger().info('Tasks done, still looking for the exit.')
            self.waiting_for_exit = True

    def do_approach(self):
        from nav2_simple_commander.robot_navigator import TaskResult
        o = self.target
        if self.time_in_state() < 1.0:        # let the explorer cancel its own goal first
            return
        if not self.approach_sent:
            dx, dy = o['x'] - o['rx'], o['y'] - o['ry']
            d = math.hypot(dx, dy) or 1.0
            ux, uy = dx / d, dy / d               # direction we first saw it from
            sx = self.P['approach_standoff_m']
            goal = PoseStamped()
            goal.header.frame_id = self.P['global_frame']
            goal.header.stamp = self.nav.get_clock().now().to_msg()
            goal.pose.position.x = o['x'] - sx * ux
            goal.pose.position.y = o['y'] - sx * uy
            yaw = math.atan2(uy, ux)              # face the obstacle
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
        self.get_logger().warn(f'Approach to {self.target["kind"]} failed: {why}')
        self.target['status'] = 'failed'
        self.target = None
        self.enter('EXPLORE')

    def do_investigate(self):
        if self.look_done:
            self.finish_look()
        elif self.time_in_state() > self.P['investigate_timeout_s']:
            self.get_logger().warn('Investigate timed out, carrying on.')
            self.finish_look()

    def finish_look(self):
        self.last_look_end = self.now_s()
        self.pending_look = False   # drop looks that came in while looking
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
