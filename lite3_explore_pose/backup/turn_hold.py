"""Gate Nav2 velocity between the smoother and the Lite3 joystick mux."""

import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String


class TurnHold:
    """Turn in place and give the forward camera time at the new heading."""

    start_rate = 0.20
    end_rate = 0.12
    min_turn_rad = 0.18
    command_timeout_s = 0.35

    def __init__(self, hold_s):
        self.hold_s = hold_s
        self.reset()

    def reset(self):
        self.turn_rad = 0.0
        self.turn_rate = 0.0
        self.last_command = None
        self.hold_until = 0.0

    def finish_turn(self, now):
        if abs(self.turn_rad) >= self.min_turn_rad:
            self.hold_until = now + self.hold_s
        self.turn_rad = 0.0
        self.turn_rate = 0.0

    def update(self, now, turn_rate):
        """Return 'pass', 'turn' (angular only), or 'hold' (zero velocity)."""
        if self.hold_s <= 0.0:
            self.last_command = now
            return 'pass'
        self.expire(now)
        if self.turn_rate:
            self.turn_rad += self.turn_rate * (now - self.last_command)
        self.last_command = now
        if now < self.hold_until:
            return 'hold'
        threshold = self.end_rate if self.turn_rate else self.start_rate
        if abs(turn_rate) >= threshold:
            self.turn_rate = turn_rate
            return 'turn'
        if self.turn_rate:
            self.finish_turn(now)
            if now < self.hold_until:
                return 'hold'
        return 'pass'

    def expire(self, now):
        """Stop on a missing command stream and finish any meaningful turn."""
        if self.last_command is None or now - self.last_command < self.command_timeout_s:
            return False
        if self.turn_rate:
            self.finish_turn(now)
        self.last_command = None
        return True


class TurnHoldNode(Node):
    def __init__(self):
        super().__init__('turn_hold')
        self.gate = TurnHold(self.declare_parameter('post_turn_hold_s', 3.0).value)
        self.nav_enabled = True
        self.holding = False
        last_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, '/nav_cmd_vel', 1)
        self.hold_pub = self.create_publisher(Bool, '/navigation/turn_hold_active', last_qos)
        self.create_subscription(Twist, '/nav_cmd_vel_raw', self.on_velocity, 1)
        self.create_subscription(String, '/mission/state', self.on_mission_state, last_qos)
        self.create_timer(0.05, self.tick)
        self.hold_pub.publish(Bool(data=False))

    def update_status(self, now):
        holding = self.nav_enabled and now < self.gate.hold_until
        if holding != self.holding:
            self.holding = holding
            self.hold_pub.publish(Bool(data=holding))
            if holding:
                self.get_logger().info(f'Holding still for {self.gate.hold_s:.1f} s after Nav2 turn.')

    def on_velocity(self, msg):
        if not self.nav_enabled:
            return
        now = time.monotonic()
        mode = self.gate.update(now, msg.angular.z)
        if mode == 'hold':
            msg = Twist()
        elif mode == 'turn':
            turn = Twist()
            turn.angular.z = msg.angular.z
            msg = turn
        self.cmd_pub.publish(msg)
        self.update_status(now)

    def on_mission_state(self, msg):
        enabled = msg.data in ('EXPLORE', 'APPROACH', 'GO_TO_EXIT')
        if self.nav_enabled and not enabled:
            self.cmd_pub.publish(Twist())
        if self.nav_enabled != enabled:
            self.gate.reset()
            self.nav_enabled = enabled
            self.update_status(time.monotonic())

    def tick(self):
        now = time.monotonic()
        if self.nav_enabled and self.gate.expire(now):
            self.cmd_pub.publish(Twist())
        self.update_status(now)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = TurnHoldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            if node.nav_enabled:
                node.cmd_pub.publish(Twist())
            node.destroy_node()
            rclpy.shutdown()
