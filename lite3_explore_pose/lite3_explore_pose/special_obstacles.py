#!/usr/bin/env python3
"""Special obstacle node: crosses stairs, rubble (uneven terrain) and curtains.

The mission controller starts it by publishing on /special/start:
    "stairs", "rubble" or "curtain"   (or "stop" to abort)
It answers on /special/done with "success" or "failed".

Every crossing follows the same steps:
    SETTLE  stop for a moment (robot steady, everyone sees it's switching)
    GAIT    (optional) switch to a special gait, if you set a gait code
    CROSS   walk STRAIGHT, slowly, holding the starting heading
            - stairs:  done when the robot tilted up (on the steps) and is level again
            - rubble:  done after walking rubble_distance_m
            - curtain: done after walking curtain_distance_m
    ABORT   (if tilted too much or not moving) stop, back off slowly -> "failed"
    FINISH  stop, restore the normal gait, report

Nav2 is NOT used while crossing: this node drives /cmd_vel directly, like
robot_goto_goal.py from Lab 3. The explorer is paused by the mission controller.

!! The numbers in config/special.yaml are PLACEHOLDERS. Measure them in the lab:
   tilt on a step, safe tilt limit, how slow is safe, how far to walk.
"""

import json
import math

from lite3_explore_pose.crossing import gait_after, stairs_exit
from lite3_explore_pose.gait import (GAIT_COMMAND_QOS, GAIT_NAMES, GaitState, GaitTable,
                                     default_gait_file, send_gait)

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32, String

try:  # Lite3 custom message (exists on the lab computer, used for /simple_cmd)
    from transfer_interfaces.msg import MotionSimpleCMD
except ImportError:  # e.g. on a laptop without the Lite3 packages
    MotionSimpleCMD = None


def quat_to_rpy(q):
    """Quaternion -> (roll, pitch, yaw) in radians."""
    roll = math.atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    sin_p = max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))
    pitch = math.asin(sin_p)
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    return roll, pitch, yaw


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SpecialObstacles(Node):

    def __init__(self):
        super().__init__('special_obstacles')

        names = []

        def p(name, default):
            names.append(name)
            self.declare_parameter(name, default)

        # Topics: CHECK with `ros2 topic list` on the robot
        p('imu_topic', '/imu/data')          # TODO: real Lite3 IMU topic name
        # Odometry sources, best first. The freshest one in this order is used; if none is
        # fresh, the crossing walks by TIME (speed x time) instead of failing.
        p('odom_topics', ['/leg_odom2', '/odom'])
        p('cmd_vel_topic', '/nav_cmd_vel')    # twist_mux input, never /cmd_vel
        p('tick_hz', 20.0)

        # Gait / mode codes: ONLY from the shared table config/gait.yaml (issue 4), never
        # guessed here. '' = the installed config/gait.yaml. This node OWNS the gait:
        # every switch is logged and published latched on /gait/current.
        p('gait_file', '')
        # Stairs: organisers allow the robot's AI gait ONLY for the stairs (enter at most
        # 1 cell before, exit at most 1 cell after). Gait codes do not work while in AI state,
        # so: exit AI first, wait, then the flat gait.
        p('stairs_use_ai_mode', True)
        # 22/11: in AI mode the robot ignored /cmd_vel from code (joystick in manual mode
        # worked). Entering AI may hand control back to the controller, so send AUTO
        # (gait.yaml 'auto') again 1 s after Enter AI.
        p('ai_resend_auto', True)
        # Stairs-only progress check: climbing the first step is slow.
        p('stairs_progress_window_s', 10.0)
        p('stairs_min_progress_m', 0.05)
        # Fallback only: true = switch manual -> gait -> auto. Gait codes DO work in auto
        # mode; on 18/11 they were dropped by a /simple_cmd QoS mismatch (fixed in gait.py).
        p('gait_via_manual', False)

        # Speeds (m/s) - start SLOW
        p('stairs_speed', 0.15)              # 22/11: 0.15 'flew' in AI mode (roll 36 deg)
        p('rubble_speed', 0.12)
        p('curtain_speed', 0.10)
        p('potholes_speed', 0.10)
        p('heading_gain', 1.0)               # keeps the robot walking straight
        p('max_turn_speed', 0.3)

        # Stairs: measure these on a real step
        p('on_stairs_pitch_deg', 8.0)        # tilt that means "I'm on the steps"
        p('level_pitch_deg', 3.0)            # tilt that means "level again"
        p('level_hold_s', 1.5)               # must stay level this long to finish
        p('stairs_min_distance_m', 0.5)
        # True: the stairs go UP, (platform), then DOWN. Finish only after a tilt one way,
        # then a tilt the OTHER way, then level -> AI stays on for the whole thing and is not
        # switched off on the platform. Works whatever the IMU's pitch sign convention is.
        p('stairs_up_and_down', True)
        p('stairs_max_time_s', 120.0)        # stairs time limit (up + platform + down is slow)
        # Issue 6: leaving AI mode after the stairs. Exit only when level for level_hold_s
        # AND odometry shows stairs_exit_after_m walked since the last tilted step (rear legs
        # off); never later than stairs_exit_max_m (rule: exit at most 1 cell after).
        p('stairs_exit_after_m', 0.35)
        p('stairs_exit_max_m', 0.9)
        # After Exit AI: wait ai_exit_wait_s, send AUTO (gait.yaml 'auto'), wait
        # auto_to_gait_s, then the gait for the terrain ahead (rubble/potholes -> high step,
        # flat or no fresh detector data -> flat).
        p('ai_exit_resend_auto', True)
        p('ai_exit_wait_s', 1.5)
        p('auto_to_gait_s', 1.0)
        p('terrain_fresh_s', 1.5)            # detector frame older than this = no data

        # Distances to walk
        p('rubble_distance_m', 1.2)          # TODO: size of a rubble cell + margin
        p('curtain_distance_m', 1.0)         # TODO: curtain depth + robot length
        p('potholes_distance_m', 1.6)        # pothole area + robot length
        # Walk distance = measured distance to the obstacle + its length + the robot's tail,
        # so a crossing that starts 1.2 m before the rubble does not stop on top of it. The
        # *_distance_m values above act as the minimum.
        p('rubble_length_m', 1.02)           # MEASURE: rubble depth along the walking direction
        p('potholes_length_m', 1.02)         # MEASURE
        p('curtain_length_m', 0.05)
        p('tail_m', 0.45)                    # base_link -> robot tail (~0.3) + margin

        # Safety
        p('max_tilt_deg', 25.0)              # roll or pitch beyond this -> abort
        p('stairs_max_pitch_deg', 45.0)      # 22/11: a step briefly pitched 40 deg and aborted
        p('tilt_hold_s', 0.5)                # tilt must stay over the limit this long to abort
        # 22/11: AI mode climbed to the platform, then stood still at the top edge (needs a
        # stronger push to step down). Stalled on the stairs for stairs_stall_s -> speed
        # +stairs_boost_step, up to stairs_max_speed; back down as soon as it moves again.
        p('stairs_stall_s', 4.0)
        p('stairs_boost_step', 0.02)
        p('stairs_max_speed', 0.20)
        p('stairs_max_roll_deg', 20.0)       # 22/11: tipped to 36 deg sideways at 0.15 m/s in AI
        # Curtain: the floor is flat, so a big tilt means climbing / pushing a WALL (22/11: a
        # false curtain crossing reached roll 27 / pitch 23 deg and the robot sat down).
        p('curtain_max_tilt_deg', 12.0)
        p('settle_s', 1.0)
        p('progress_window_s', 6.0)          # must move min_progress_m in this time
        p('min_progress_m', 0.10)
        p('backoff_speed', 0.08)
        p('backoff_s', 2.0)
        p('max_time_s', 45.0)                # whole crossing time limit
        # Stop rubble / pothole crossings when a wall (depth, >= 20 cm tall) is this close to
        # base_link (robot nose is ~0.3 m ahead of base_link). Not for curtain / stairs: the
        # camera sees the curtain and the steps as walls.
        p('wall_stop_m', 0.45)
        # Rubble / potholes: follow the DETECTOR instead of a fixed distance (22/11). Keep the
        # terrain gait while rubble/potholes are still detected ahead; once the floor ahead is
        # clear for terrain_clear_s, walk terrain_clear_extra_m more (floor check looks from
        # 0.4 m ahead + the back legs are still on it) and finish. No detector data -> the
        # fixed walk distance as before. Never more than terrain_max_factor x that distance.
        p('terrain_follow', True)
        p('terrain_clear_s', 1.0)
        p('terrain_clear_extra_m', 0.7)
        p('terrain_max_factor', 1.6)

        self.P = {name: self.get_parameter(name).value for name in names}
        path = self.P['gait_file'] or default_gait_file()
        try:
            self.gaits, errors = GaitTable.load(path)
        except (OSError, ValueError) as e:
            self.gaits, errors = GaitTable(), [f'cannot read gait table {path}: {e!r}']
        for err in errors:
            self.get_logger().error(err)
        self.get_logger().info(f'Gait table {path}: '
                               + ', '.join(f'{k}={hex(v)}' for k, v in self.gaits.codes.items())
                               + f'; terrain {self.gaits.terrain}')
        self.gait = GaitState()

        # ---- ROS interfaces ----
        self.create_subscription(String, '/special/start', self.on_start, 10)
        # Wall ahead from the depth camera (obstacle_detector). Rubble / pothole crossings
        # walk straight by distance and must not walk into a wall behind the cell.
        self.wall = None                     # (time, distance m)
        # Last detections from obstacle_detector (forward_m of what triggered the crossing).
        self.last_dets = None                # (time, detections)
        self.create_subscription(String, '/perception/obstacles', self.on_dets, 10)
        self.walk_m = 0.0
        self.create_subscription(Float32, '/perception/wall_ahead',
                                 lambda m: setattr(self, 'wall', (self.now_s(), m.data)), 10)
        self.create_subscription(Imu, self.P['imu_topic'], self.on_imu, qos_profile_sensor_data)
        self.odom = {}                       # topic -> (time, xy, yaw)
        for topic in self.P['odom_topics']:
            self.create_subscription(Odometry, topic,
                                     lambda m, t=topic: self.on_odom(m, t), 10)
        self.cmd_pub = self.create_publisher(Twist, self.P['cmd_vel_topic'], 10)
        self.done_pub = self.create_publisher(String, '/special/done', 10)
        self.state_pub = self.create_publisher(String, '/special/state', 10)
        self.gait_pub = None
        if MotionSimpleCMD is not None:
            self.gait_pub = self.create_publisher(MotionSimpleCMD, '/simple_cmd', GAIT_COMMAND_QOS)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.gait_cur_pub = self.create_publisher(String, '/gait/current', latched)
        self.gait_cur_pub.publish(String(data=self.gait.current))
        # Other nodes (explorer at start) ask for a gait here instead of sending codes.
        self.create_subscription(String, '/gait/request', self.on_gait_request, 10)
        # Issue 6: True while AI mode is still on outside a crossing (stopped ON the stairs).
        # mission_controller holds the robot still until a person walks it off and sends
        # /special/start exit_ai.
        self.ai_pub = self.create_publisher(Bool, '/special/ai_active', latched)
        self.ai_stuck = False
        self.ai_pub.publish(Bool(data=False))

        # ---- memory ----
        self.roll = self.pitch = 0.0
        self.have_imu = False
        self.xy = None
        self.yaw = None
        self.kind = None           # 'stairs' | 'rubble' | 'curtain'
        self.phase = 'IDLE'
        self.phase_since = 0.0
        self.start_time = 0.0
        self.start_xy = None
        self.start_yaw = None
        self.was_on_stairs = False
        self.level_since = None
        self.first_tilt_sign = 0    # +1 / -1: sign of pitch on the first flight (0 = not yet)
        self.second_flight = False  # tilted the other way after the first flight
        self.platform_logged = False
        self.window_start = 0.0
        self.window_xy = None
        self.in_ai = False
        self.pending_result = None
        self.odom_src = None
        self.dr_dist = 0.0
        self.dr_last = 0.0
        self.restore_sent = False

        self.create_timer(1.0 / self.P['tick_hz'], self.tick)
        self.get_logger().info('Special obstacle node ready (waiting on /special/start)')

    # ================= inputs =================

    def on_start(self, msg):
        cmd = msg.data.strip()
        if cmd == 'stop':
            if self.phase != 'IDLE':
                self.get_logger().warn('Stop requested by mission controller.')
                self.stop_robot()
                if self.in_ai and self.kind == 'stairs' and self.phase != 'RESTORE':
                    # 22/11 lesson: never leave AI mode ON the stairs, also not on 'stop'
                    # (localization loss / traverse timeout). Stand still, AI stays on.
                    self.stairs_stuck('stop requested while on the stairs')
                    return
                self.restore_gait()
                self.set_phase('IDLE')
            return
        if cmd == 'exit_ai':
            self.manual_exit_ai()
            return
        if self.ai_stuck or (self.in_ai and self.phase == 'IDLE'):
            self.get_logger().error(f'{cmd}: refused, AI mode is still ON (stairs). Walk the '
                                    'robot off with the joystick, then /special/start exit_ai')
            self.done_pub.publish(String(data='failed'))
            return
        if cmd not in ('stairs', 'rubble', 'curtain', 'potholes'):
            self.get_logger().warn(f'Unknown obstacle "{cmd}"')
            return
        src = self.fresh_odom()
        self.odom_src = src[0] if src else None
        if src is None:
            self.get_logger().warn('No odometry: crossing will walk by TIME (speed x time).')
        else:
            self.get_logger().info(f'Using odometry {self.odom_src}')
        self.kind = cmd
        self.tilt_since = None
        self.wall_free_m = 0.0               # wall stop only after walking this far
        self.start_dist = 0.0                # measured distance to the terrain at the start
        if cmd != 'stairs':
            self.walk_m = self.P[f'{cmd}_distance_m']
            dist = self.measured_distance(cmd)
            # The wall stop is for the wall BEHIND the cell. On the rubble the body is higher,
            # so the rubble's own blocks looked >= 20 cm tall and stopped the crossing after
            # ~0.35 m (live 22/11). Ignore 'walls' until the robot is past the cell.
            self.wall_free_m = (dist or 0.0) + self.P[f'{cmd}_length_m']
            self.start_dist = dist or 0.0
            self.min_walk_m = (dist or 0.0) + 0.3      # at least onto the terrain
            if dist is not None:
                self.walk_m = max(self.walk_m,
                                  dist + self.P[f'{cmd}_length_m'] + self.P['tail_m'])
            self.get_logger().info(f'{cmd}: starts {dist} m ahead -> walk {self.walk_m:.2f} m')
        self.start_time = self.now_s()
        # 'Rubble' / 'potholes' right at a wall = the bottom of the wall, not terrain (22/11:
        # false rubble at the start pushed the wall for 18 s). Don't walk into it.
        if cmd in ('rubble', 'potholes') and self.wall is not None \
                and self.now_s() - self.wall[0] < 1.5 \
                and 0.0 < self.wall[1] < max(self.start_dist, self.P['wall_stop_m']) + 0.10:
            # +0.10: the wall is AT the 'terrain' (its bottom). A 'wall' further behind the
            # terrain start is the rubble's own tall blocks (22/11: 0.71 m vs 0.54 m = real rubble)
            self.get_logger().warn(f'{cmd}: wall {self.wall[1]:.2f} m ahead, at the {cmd} '
                                   f'({self.start_dist:.2f} m): not terrain, NOT crossing')
            self.report('failed')
            return
        self.get_logger().info(f'Starting {cmd} crossing. Terrain stats: {self.terrain_stats(cmd)}')
        self.set_phase('SETTLE')

    def on_dets(self, msg):
        try:
            self.last_dets = (self.now_s(), json.loads(msg.data))
        except ValueError:
            pass

    def terrain_stats(self, kind):
        """Detector numbers behind this crossing (bump / hole / wall fractions, distance)."""
        if self.last_dets is None:
            return 'no detector data'
        age = self.now_s() - self.last_dets[0]
        keys = ('forward_m', 'located_by', 'bump', 'hole', 'wall', 'wall_at', 'bar_at', 'n',
                'suppressed', 'wrinkle_mm', 'steps', 'rises', 'yaw_deg')
        out = [{k: d[k] for k in keys if k in d} for d in self.last_dets[1]
               if d.get('kind') == kind]
        return f'{out or "none of this kind"} ({age:.1f} s old)'

    def measured_distance(self, kind):
        """Distance to the obstacle that triggered this crossing, or None."""
        if self.last_dets is None or self.now_s() - self.last_dets[0] > 2.0:
            return None
        d = [x['forward_m'] for x in self.last_dets[1]
             if x.get('kind') == kind and x.get('trigger') and 'forward_m' in x]
        return min(d) if d else None

    def on_imu(self, msg):
        self.roll, self.pitch, _ = quat_to_rpy(msg.orientation)
        self.have_imu = True

    def on_odom(self, msg, topic):
        _, _, yaw = quat_to_rpy(msg.pose.pose.orientation)
        self.odom[topic] = (self.now_s(), (msg.pose.pose.position.x, msg.pose.pose.position.y), yaw)

    def fresh_odom(self):
        """(topic, xy, yaw) of the first fresh source in odom_topics order, else None."""
        for topic in self.P['odom_topics']:
            o = self.odom.get(topic)
            if o and self.now_s() - o[0] < 0.5:
                return topic, o[1], o[2]
        return None

    # ================= helpers =================

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def set_phase(self, phase):
        if phase != self.phase:
            self.get_logger().info(f'{self.kind}: {self.phase} -> {phase}')
        self.phase = phase
        self.phase_since = self.now_s()
        self.state_pub.publish(String(data=f'{self.kind}:{phase}'))

    def in_phase_s(self):
        return self.now_s() - self.phase_since

    def drive(self, speed, turn=0.0):
        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(turn)
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.drive(0.0, 0.0)

    def send_gait(self, code):
        if code == 0:
            return
        if self.gait_pub is None:
            self.get_logger().warn('transfer_interfaces not found, cannot switch gait.')
            return
        send_gait(self.gait_pub, MotionSimpleCMD, code, self.get_logger(),
                  via_manual=self.P['gait_via_manual'])

    def gait_code_for(self, kind):
        name = self.gaits.for_terrain(kind)          # curtain has no special gait
        return self.gaits.code(name) if name else 0

    def set_gait(self, name, why):
        """THE way to change gait / mode: table code -> /simple_cmd, log, /gait/current.
        name: 'flat' | 'high_step' | 'ai' (enter) | 'ai_exit' | 'auto'."""
        code = {'ai': self.gaits.code('ai_enter')}.get(name, self.gaits.code(name))
        if not code:
            self.get_logger().error(f'GAIT: no tested code for {name!r} in the gait table, '
                                    f'NOT switching ({why})')
            return False
        self.send_gait(code)
        state = {'ai_exit': 'ai_off', 'auto': self.gait.current}.get(name, name)
        old = self.gait.current
        self.gait.switch(state, why, self.now_s())
        self.get_logger().warn(f'GAIT: {old} -> {state} ({hex(code)}, {why})')
        self.gait_cur_pub.publish(String(data=state))
        return True

    def on_gait_request(self, msg):
        name = msg.data.strip()
        if name not in GAIT_NAMES:
            self.get_logger().warn(f'/gait/request {name!r}: unknown (use {GAIT_NAMES})')
            return
        if self.phase != 'IDLE' or self.in_ai:
            self.get_logger().warn(f'/gait/request {name!r} ignored: {self.kind} {self.phase}'
                                   f'{" (AI on)" if self.in_ai else ""}')
            return
        if self.gait.current == name:
            self.gait_cur_pub.publish(String(data=name))   # confirm
            return
        self.set_gait(name, 'requested on /gait/request')

    def restore_gait(self):
        # Only switch back if we switched away from the normal gait.
        if self.in_ai:
            self.set_gait('ai_exit', 'stop: leaving AI')     # immediate (used for 'stop')
            self.in_ai = False
            self.set_gait('flat', 'stop: back to flat')
        elif self.kind and self.gait_code_for(self.kind):
            self.set_gait('flat', f'{self.kind}: back to flat')

    def update_pose(self):
        o = self.odom.get(self.odom_src) if self.odom_src else None
        if o and self.now_s() - o[0] < 0.5:
            self.xy, self.yaw = o[1], o[2]
        else:
            self.xy = None              # lost (or never had) odometry -> time mode

    def walked_enough_for(self, target):
        """Odometry can jump (live test: 6 of 8 curtain crossings 'finished' in < 0.5 s).
        Done only if odometry says so AND speed x time agrees (>= 70 %), or if
        speed x time alone reaches 130 % (odometry lagging / missing)."""
        timed = self.dr_dist
        if self.xy is None or self.start_xy is None:
            return timed >= target
        odo = math.hypot(self.xy[0] - self.start_xy[0], self.xy[1] - self.start_xy[1])
        return (odo >= target and timed >= 0.7 * target) or timed >= 1.3 * target

    def travelled(self):
        if self.xy is None or self.start_xy is None:
            return self.dr_dist          # dead reckoning: commanded speed x time
        return math.hypot(self.xy[0] - self.start_xy[0], self.xy[1] - self.start_xy[1])

    def abort(self, reason):
        self.get_logger().warn(f'{self.kind}: ABORT ({reason})')
        self.set_phase('ABORT')

    def stairs_stuck(self, why):
        """Stopped ON the stairs: AI stays on, robot still, everybody is told."""
        self.stop_robot()
        self.ai_stuck = True
        self.ai_pub.publish(Bool(data=True))
        self.state_pub.publish(String(data='stairs:AI_STILL_ON'))
        for _ in range(3):
            self.get_logger().error(
                f'stairs: {why}. AI mode stays ON, robot standing still. Walk it off the '
                'stairs with the joystick (lite3_manual), then: ros2 topic pub --once '
                "/special/start std_msgs/msg/String \"{data: exit_ai}\"")
        self.set_phase('IDLE')

    def manual_exit_ai(self):
        """/special/start exit_ai: a person walked the robot off the stairs."""
        if not (self.in_ai or self.ai_stuck):
            self.get_logger().warn('exit_ai: AI mode is not on (nothing to do)')
            self.ai_pub.publish(Bool(data=False))
            return
        if self.phase not in ('IDLE',):
            self.get_logger().warn(f'exit_ai: refused during {self.kind} {self.phase}')
            return
        self.get_logger().warn('exit_ai: manual recovery, leaving AI mode now')
        self.kind = 'stairs'
        self.set_gait('ai_exit', 'manual exit_ai')
        self.in_ai = False
        self.ai_stuck = False
        self.pending_result = None              # nobody waits for /special/done
        self.restore_sent = self.restore_auto_sent = False
        self.set_phase('RESTORE')

    def finish(self, result):
        self.stop_robot()
        if self.in_ai and self.kind == 'stairs' and result == 'failed':
            # 22/11: an abort on top of the stairs switched AI off -> normal gait on the steps.
            # Stay in AI mode and stand still; a person finishes with the joystick.
            self.report(result)
            self.stairs_stuck('stopped ON the stairs (crossing failed)')
            return
        if self.in_ai:
            # Leave AI mode, stand still, then normal gait, THEN report (see do_restore).
            self.set_gait('ai_exit', 'stairs finished')
            self.in_ai = False
            self.get_logger().info('stairs: AI mode OFF')
            self.pending_result = result
            self.restore_sent = self.restore_auto_sent = False
            self.set_phase('RESTORE')
            return
        self.restore_gait()
        self.report(result)

    def do_restore(self):
        """After Exit AI: wait -> AUTO -> wait -> gait for the terrain ahead -> report."""
        self.stop_robot()
        t = self.in_phase_s()
        wait, to_gait = self.P['ai_exit_wait_s'], self.P['auto_to_gait_s']
        if (not getattr(self, 'restore_auto_sent', False) and t >= wait
                and self.P['ai_exit_resend_auto']):
            self.send_gait(self.gaits.code('auto'))
            self.get_logger().info('stairs: AUTO re-sent after Exit AI')
            self.restore_auto_sent = True
        if not self.restore_sent and t >= wait + to_gait:
            age = None if self.last_dets is None else self.now_s() - self.last_dets[0]
            name, why = gait_after(None if self.last_dets is None else self.last_dets[1], age,
                                   self.P['terrain_fresh_s'],
                                   terrain_gait=self.gaits.for_terrain)
            self.set_gait(name, f'after the stairs: {why}')
            self.restore_sent = True
        if t >= wait + to_gait + 1.0:
            self.ai_pub.publish(Bool(data=False))
            if self.pending_result is None:
                self.set_phase('IDLE')
            else:
                self.report(self.pending_result)

    def report(self, result):
        self.get_logger().info(f'{self.kind}: {result}')
        self.done_pub.publish(String(data=result))
        self.set_phase('IDLE')

    # ================= main loop =================

    def tick(self):
        if self.phase == 'IDLE':
            return

        # Safety first, in every phase except backing off
        if self.phase in ('SETTLE', 'GAIT', 'CROSS'):
            limit = math.radians(self.P['curtain_max_tilt_deg'] if self.kind == 'curtain'
                                 else self.P['stairs_max_roll_deg'] if self.kind == 'stairs'
                                 else self.P['max_tilt_deg'])
            pitch_limit = (math.radians(self.P['stairs_max_pitch_deg'])
                           if self.kind == 'stairs' else limit)
            over = abs(self.roll) > limit or abs(self.pitch) > pitch_limit
            if not over:
                self.tilt_since = None
            elif getattr(self, 'tilt_since', None) is None:
                self.tilt_since = self.now_s()     # brief peak while stepping: wait tilt_hold_s
            if over and self.now_s() - self.tilt_since >= self.P['tilt_hold_s']:
                self.tilt_since = None
                self.abort(f'tilt roll={math.degrees(self.roll):.0f} '
                           f'pitch={math.degrees(self.pitch):.0f} deg')
            elif self.now_s() - self.start_time > (self.P['stairs_max_time_s']
                                                   if self.kind == 'stairs'
                                                   else self.P['max_time_s']):
                self.abort('took too long')

        getattr(self, f'do_{self.phase.lower()}')()

    def do_settle(self):
        self.stop_robot()
        if self.in_phase_s() >= self.P['settle_s']:
            if self.kind == 'stairs' and not self.have_imu:
                self.get_logger().error(f'No IMU on {self.P["imu_topic"]}: unsafe to climb.')
                self.finish('failed')
                return
            if self.kind == 'stairs' and self.P['stairs_use_ai_mode']:
                self.set_gait('ai', 'stairs ahead')
                self.in_ai = True
                self.ai_auto_count = 0
                self.get_logger().info('stairs: AI mode ON')
                self.set_phase('GAIT')
                return
            code = self.gait_code_for(self.kind)
            if code:
                self.set_gait(self.gaits.for_terrain(self.kind), f'{self.kind} crossing')
                self.gait_resent = False
                self.set_phase('GAIT')
            else:
                self.begin_cross()

    def do_gait(self):
        self.stop_robot()
        # Send the terrain gait a second time after 1 s, in case the first message was lost.
        # Same code twice is harmless. Not for AI mode (enter AI only once).
        if (not self.in_ai and not getattr(self, 'gait_resent', True)
                and self.in_phase_s() >= 1.0):
            self.send_gait(self.gait_code_for(self.kind))
            self.gait_resent = True
        # AI mode: re-assert AUTO (ROS commands) at 1 s and 2.5 s after Enter AI. 22/11: one
        # send at 1 s sometimes landed while AI was still switching on and was ignored.
        if self.in_ai and self.P['ai_resend_auto']:
            sent = getattr(self, 'ai_auto_count', 2)
            due = (1.0, 2.5)
            if sent < len(due) and self.in_phase_s() >= due[sent]:
                self.send_gait(self.gaits.code('auto'))
                self.ai_auto_count = sent + 1
                self.get_logger().info(f'stairs: AUTO re-sent after entering AI mode ({sent + 1}/2)')
        wait = 3.0 if self.in_ai else 2.0    # AI mode takes longer to switch on
        if self.in_phase_s() >= wait:   # give the robot time to change gait / enter AI
            self.begin_cross()

    def terrain_ahead(self):
        """True / False = rubble or potholes detected ahead in the latest detector frame;
        None = no fresh detector data (then the fixed distance is used)."""
        if self.last_dets is None or self.now_s() - self.last_dets[0] > 1.5:
            return None
        for d in self.last_dets[1]:
            if d.get('kind') in ('rubble', 'potholes') and (
                    d.get('trigger') or 0.0 < d.get('forward_m', 99.0) <= 1.2):
                return True
        return False

    def begin_cross(self):
        self.stall_xy = None                       # stairs push restarts
        self.last_terrain_seen = self.now_s()     # it was seen to start this crossing
        self.clear_from = None
        self.update_pose()
        self.start_xy = self.xy
        self.start_yaw = self.yaw if self.xy is not None else None
        self.dr_dist = 0.0
        self.dr_last = self.now_s()
        self.was_on_stairs = False
        self.level_since = None
        self.first_tilt_sign = 0
        self.second_flight = False
        self.platform_logged = False
        self.window_start = self.now_s()
        self.window_xy = self.xy
        self.set_phase('CROSS')
        mode = f'odometry {self.odom_src}' if self.xy is not None else 'TIME (no odometry)'
        self.get_logger().info(f'{self.kind}: crossing, distance by {mode}')

    def stairs_push(self, now, base):
        """Stairs: if the robot has not moved for stairs_stall_s, push a little harder."""
        if self.xy is None:
            return base
        if getattr(self, 'stall_xy', None) is None:
            self.stall_xy, self.stall_t, self.stairs_boost = self.xy, now, 0.0
        if math.hypot(self.xy[0] - self.stall_xy[0], self.xy[1] - self.stall_xy[1]) >= 0.03:
            self.stall_xy, self.stall_t = self.xy, now          # moving again
            self.stairs_boost = max(0.0, self.stairs_boost - self.P['stairs_boost_step'])
        elif now - self.stall_t > self.P['stairs_stall_s']:
            top = self.P['stairs_max_speed'] - base
            if self.stairs_boost < top:
                self.stairs_boost = min(top, self.stairs_boost + self.P['stairs_boost_step'])
                self.get_logger().warn(f'stairs: not moving, pushing harder: '
                                       f'{base + self.stairs_boost:.2f} m/s')
            if self.in_ai and self.P['ai_resend_auto']:
                self.send_gait(self.gaits.code('auto'))  # maybe AUTO was not taken: again
                self.get_logger().warn('stairs: not moving, AUTO re-sent')
            self.stall_t = now
        return base + self.stairs_boost

    def do_cross(self):
        # Walk straight: steer back to the starting heading
        self.update_pose()
        now = self.now_s()
        speed = self.P[f'{self.kind}_speed']
        if self.kind == 'stairs':
            speed = self.stairs_push(now, speed)
        turn = 0.0
        if self.xy is not None and self.start_yaw is not None:
            turn = self.P['heading_gain'] * wrap(self.start_yaw - self.yaw)
            turn = max(-self.P['max_turn_speed'], min(self.P['max_turn_speed'], turn))
        self.drive(speed, turn)
        self.dr_dist += speed * (now - self.dr_last)
        self.dr_last = now

        # Moving at all? (only measurable with odometry)
        if self.xy is not None and self.window_xy is not None \
                and now - self.window_start > (self.P['stairs_progress_window_s']
                                               if self.kind == 'stairs'
                                               else self.P['progress_window_s']):
            moved = math.hypot(self.xy[0] - self.window_xy[0], self.xy[1] - self.window_xy[1])
            need = (self.P['stairs_min_progress_m'] if self.kind == 'stairs'
                    else self.P['min_progress_m'])
            if moved < need:
                self.abort(f'only moved {moved:.2f} m in {now - self.window_start:.0f} s')
                return
            self.window_start, self.window_xy = now, self.xy

        # Wall right ahead (rubble / potholes only): stop here, the cell is behind us.
        if self.kind in ('rubble', 'potholes') and self.wall is not None \
                and (self.travelled() < self.start_dist - 0.1
                     or self.travelled() >= self.wall_free_m) \
                and now - self.wall[0] < 1.5 and 0.0 < self.wall[1] < self.P['wall_stop_m']:
            self.get_logger().warn(f'{self.kind}: wall {self.wall[1]:.2f} m ahead, '
                                   f'stopping the crossing here')
            self.finish('success')
            return

        # Finished?
        dist = self.walked_enough_for
        if self.kind == 'stairs':
            pitch_deg = math.degrees(self.pitch)
            if abs(pitch_deg) > self.P['on_stairs_pitch_deg']:
                sign = 1 if pitch_deg > 0 else -1
                if self.first_tilt_sign == 0:
                    self.first_tilt_sign = sign
                    self.get_logger().info(f'stairs: on the first flight (pitch {pitch_deg:.0f} deg)')
                elif sign != self.first_tilt_sign and not self.second_flight:
                    self.second_flight = True
                    self.get_logger().info(f'stairs: on the second flight (pitch {pitch_deg:.0f} deg)')
                self.was_on_stairs = True
                self.level_since = None
                self.last_tilt_m = self.travelled()     # issue 6: exit distance from here
            elif self.was_on_stairs and abs(pitch_deg) < self.P['level_pitch_deg']:
                if self.P['stairs_up_and_down'] and not self.second_flight:
                    # Level, but only after one flight: this is the platform. Keep AI on and
                    # keep walking straight until the robot goes down the other side.
                    self.level_since = None
                    if not self.platform_logged:
                        self.platform_logged = True
                        self.get_logger().info('stairs: level on the platform, keep walking (AI stays on)')
                else:
                    if self.level_since is None:
                        self.level_since = now
                    past = self.travelled() - getattr(self, 'last_tilt_m', 0.0)
                    step = stairs_exit(now - self.level_since, past, True,
                                       self.P['level_hold_s'], self.P['stairs_exit_after_m'],
                                       self.P['stairs_exit_max_m'])
                    if step != 'wait' and dist(self.P['stairs_min_distance_m']):
                        self.get_logger().info(
                            f'stairs: off the stairs, level {now - self.level_since:.1f} s, '
                            f'{past:.2f} m past the last step'
                            + (' (1-cell limit reached)' if step == 'exit_cap' else '')
                            + ' -> exit AI')
                        self.finish('success')
        elif self.kind in ('rubble', 'potholes') and self.P['terrain_follow']:
            self.follow_terrain(now, dist)
        elif dist(self.walk_m):
            self.finish('success')

    def follow_terrain(self, now, dist):
        """Rubble / potholes: stay in the terrain gait while terrain is detected ahead."""
        walked = self.travelled()
        ahead = self.terrain_ahead()
        if ahead is None:                            # detector silent: fixed distance
            if dist(self.walk_m):
                self.finish('success')
            return
        if ahead:
            self.last_terrain_seen = now
            if self.clear_from is not None:
                self.get_logger().info(f'{self.kind}: more terrain ahead, keep going')
            self.clear_from = None
        if dist(self.walk_m * self.P['terrain_max_factor']):
            self.get_logger().warn(f'{self.kind}: max distance {walked:.2f} m, finishing')
            self.finish('success')
            return
        if walked < getattr(self, 'min_walk_m', 0.0) or ahead:
            return
        if now - self.last_terrain_seen < self.P['terrain_clear_s']:
            return
        if self.clear_from is None:
            self.clear_from = walked
            self.get_logger().info(f'{self.kind}: floor ahead clear at {walked:.2f} m, '
                                   f'walking {self.P["terrain_clear_extra_m"]:.1f} m off it')
        if walked - self.clear_from >= self.P['terrain_clear_extra_m']:
            self.get_logger().info(f'{self.kind}: off the terrain after {walked:.2f} m')
            self.finish('success')

    def do_abort(self):
        # Back off slowly and straight, then report failure. NOT on the stairs: backing
        # down steps is dangerous (22/11) -> just stop.
        if self.kind != 'stairs' and self.in_phase_s() < self.P['backoff_s']:
            self.drive(-abs(self.P['backoff_speed']))
        else:
            self.finish('failed')

    def do_idle(self):
        pass


def main(args=None):
    # Handle Ctrl+C ourselves so the stop command is sent before shutdown.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = SpecialObstacles()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
