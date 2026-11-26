from unittest.mock import Mock

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from lite3_explore_pose.turn_hold import TurnHold, TurnHoldNode


def test_significant_turn_gates_translation_and_then_holds_three_seconds():
    gate = TurnHold(3.0)
    assert gate.update(0.0, 0.0) == 'pass'
    for index in range(1, 6):
        assert gate.update(index / 10, 0.5) == 'turn'
    assert gate.update(0.6, 0.0) == 'hold'
    assert gate.update(3.59, 0.5) == 'hold'
    assert gate.update(3.60, 0.0) == 'pass'


def test_small_correction_and_disabled_hold_allow_forward_motion():
    gate = TurnHold(3.0)
    assert gate.update(1.0, 0.1) == 'pass'
    assert gate.update(1.1, -0.4) == 'turn'
    assert gate.update(1.2, -0.15) == 'turn'  # Hysteresis keeps a turn continuous.
    assert gate.update(1.3, 0.0) == 'pass'
    assert TurnHold(0.0).update(1.0, 1.0) == 'pass'


def test_lost_command_stops_and_hold_survives_until_a_fresh_command():
    gate = TurnHold(3.0)
    for index in range(5):
        assert gate.update(index / 10, -0.6) == 'turn'
    assert gate.expire(0.8)
    assert not gate.expire(0.9)
    assert gate.update(1.0, 0.0) == 'hold'
    assert gate.update(3.8, 0.0) == 'pass'


def test_traverse_stops_and_suppresses_nav2_until_control_returns():
    node = object.__new__(TurnHoldNode)
    node.gate = TurnHold(3.0)
    node.nav_enabled = True
    node.holding = False
    node.cmd_pub = Mock()
    node.hold_pub = Mock()
    velocity = Twist()
    velocity.linear.x = 0.3
    velocity.angular.z = 0.5
    node.on_velocity(velocity)
    turning = node.cmd_pub.publish.call_args.args[0]
    assert turning.linear.x == 0.0 and turning.angular.z == 0.5
    node.on_mission_state(String(data='TRAVERSE'))
    assert node.cmd_pub.publish.call_args.args[0] == Twist()
    node.cmd_pub.publish.reset_mock()
    node.on_velocity(velocity)
    node.tick()
    node.cmd_pub.publish.assert_not_called()
    node.on_mission_state(String(data='EXPLORE'))
    assert node.nav_enabled
