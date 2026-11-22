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

from lite3_explore.gait import GAIT_COMMAND_QOS, send_gait

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String

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

        # Gait codes for /simple_cmd. 0 = don't switch gait.
        # TODO: ask TA / Lite3 manual. Never guess codes on the real robot.
        p('normal_gait_code', 0)
        p('stairs_gait_code', 0)
        p('rubble_gait_code', 0)
        p('potholes_gait_code', 0)
        # Stairs: organisers allow the robot's AI gait ONLY for the stairs (enter at most
        # 1 cell before, exit at most 1 cell after). Gait codes do not work while in AI state,
        # so: exit AI first, wait, then send normal_gait_code.
        p('stairs_use_ai_mode', True)
        p('ai_enter_code', 553714984)        # 0x21010528 Enter AI
        p('ai_exit_code', 553714987)         # 0x2101052B Exit AI
        # Fallback only: true = switch manual -> gait -> auto. Gait codes DO work in auto
        # mode; on 18/11 they were dropped by a /simple_cmd QoS mismatch (fixed in gait.py).
        p('gait_via_manual', False)

        # Speeds (m/s) - start SLOW
        p('stairs_speed', 0.10)
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
        p('stairs_max_pitch_deg', 40.0)      # stairs climb steeper than 25 deg; roll limit stays
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

        self.P = {name: self.get_parameter(name).value for name in names}

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
                self.restore_gait()
                self.set_phase('IDLE')
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
        self.wall_free_m = 0.0               # wall stop only after walking this far
        if cmd != 'stairs':
            self.walk_m = self.P[f'{cmd}_distance_m']
            dist = self.measured_distance(cmd)
            # The wall stop is for the wall BEHIND the cell. On the rubble the body is higher,
            # so the rubble's own blocks looked >= 20 cm tall and stopped the crossing after
            # ~0.35 m (live 22/11). Ignore 'walls' until the robot is past the cell.
            self.wall_free_m = (dist or 0.0) + self.P[f'{cmd}_length_m']
            if dist is not None:
                self.walk_m = max(self.walk_m,
                                  dist + self.P[f'{cmd}_length_m'] + self.P['tail_m'])
            self.get_logger().info(f'{cmd}: starts {dist} m ahead -> walk {self.walk_m:.2f} m')
        self.start_time = self.now_s()
        self.get_logger().info(f'Starting {cmd} crossing.')
        self.set_phase('SETTLE')

    def on_dets(self, msg):
        try:
            self.last_dets = (self.now_s(), json.loads(msg.data))
        except ValueError:
            pass

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
        return self.P.get(f'{kind}_gait_code', 0)   # curtain has no special gait

    def restore_gait(self):
        # Only switch back if we switched away from the normal gait.
        if self.in_ai:
            self.send_gait(self.P['ai_exit_code'])        # immediate (used for 'stop')
            self.in_ai = False
            self.send_gait(self.P['normal_gait_code'])
        elif self.kind and self.gait_code_for(self.kind):
            self.send_gait(self.P['normal_gait_code'])

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

    def finish(self, result):
        self.stop_robot()
        if self.in_ai:
            # Leave AI mode, stand still, then normal gait, THEN report (see do_restore).
            self.send_gait(self.P['ai_exit_code'])
            self.in_ai = False
            self.get_logger().info('stairs: AI mode OFF')
            self.pending_result = result
            self.restore_sent = False
            self.set_phase('RESTORE')
            return
        self.restore_gait()
        self.report(result)

    def do_restore(self):
        self.stop_robot()
        if not self.restore_sent and self.in_phase_s() >= 1.5:
            self.send_gait(self.P['normal_gait_code'])
            self.restore_sent = True
        if self.in_phase_s() >= 3.0:
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
            limit = math.radians(self.P['max_tilt_deg'])
            pitch_limit = (math.radians(self.P['stairs_max_pitch_deg'])
                           if self.kind == 'stairs' else limit)
            if abs(self.roll) > limit or abs(self.pitch) > pitch_limit:
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
                self.send_gait(self.P['ai_enter_code'])
                self.in_ai = True
                self.get_logger().info('stairs: AI mode ON')
                self.set_phase('GAIT')
                return
            code = self.gait_code_for(self.kind)
            if code:
                self.send_gait(code)
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
        if self.in_phase_s() >= 2.0:   # give the robot time to change gait / enter AI
            self.begin_cross()

    def begin_cross(self):
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

    def do_cross(self):
        # Walk straight: steer back to the starting heading
        self.update_pose()
        now = self.now_s()
        speed = self.P[f'{self.kind}_speed']
        turn = 0.0
        if self.xy is not None and self.start_yaw is not None:
            turn = self.P['heading_gain'] * wrap(self.start_yaw - self.yaw)
            turn = max(-self.P['max_turn_speed'], min(self.P['max_turn_speed'], turn))
        self.drive(speed, turn)
        self.dr_dist += speed * (now - self.dr_last)
        self.dr_last = now

        # Moving at all? (only measurable with odometry)
        if self.xy is not None and self.window_xy is not None \
                and now - self.window_start > self.P['progress_window_s']:
            moved = math.hypot(self.xy[0] - self.window_xy[0], self.xy[1] - self.window_xy[1])
            if moved < self.P['min_progress_m']:
                self.abort(f'only moved {moved:.2f} m in {self.P["progress_window_s"]:.0f} s')
                return
            self.window_start, self.window_xy = now, self.xy

        # Wall right ahead (rubble / potholes only): stop here, the cell is behind us.
        if self.kind in ('rubble', 'potholes') and self.wall is not None \
                and self.travelled() >= self.wall_free_m \
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
            elif self.was_on_stairs and abs(pitch_deg) < self.P['level_pitch_deg']:
                if self.P['stairs_up_and_down'] and not self.second_flight:
                    # Level, but only after one flight: this is the platform. Keep AI on and
                    # keep walking straight until the robot goes down the other side.
                    self.level_since = None
                    if not self.platform_logged:
                        self.platform_logged = True
                        self.get_logger().info('stairs: level on the platform, keep walking (AI stays on)')
                elif self.level_since is None:
                    self.level_since = now
                elif (now - self.level_since > self.P['level_hold_s']
                      and dist(self.P['stairs_min_distance_m'])):
                    self.finish('success')
        elif dist(self.walk_m):
            self.finish('success')

    def do_abort(self):
        # Back off slowly and straight, then report failure
        if self.in_phase_s() < self.P['backoff_s']:
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
