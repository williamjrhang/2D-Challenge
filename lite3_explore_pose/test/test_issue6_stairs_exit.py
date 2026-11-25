"""Issue 6: leave AI mode after the stairs (level + odometry distance), AUTO, then the gait
for the terrain ahead; an abort / stop on the stairs keeps AI on; manual exit_ai."""
import math
from types import SimpleNamespace

from std_msgs.msg import Bool, String

import lite3_explore_pose.special_obstacles as so
from lite3_explore_pose.crossing import gait_after, stairs_exit
from lite3_explore_pose.special_obstacles import SpecialObstacles
import os

from node_fakes import CFG, make

if so.MotionSimpleCMD is None:                      # laptop without transfer_interfaces
    so.MotionSimpleCMD = lambda cmd_code, type: SimpleNamespace(cmd_code=cmd_code)

FLAT, HSTEP, AI_IN, AI_OUT, AUTO = 0x21010300, 0x21010407, 0x21010528, 0x2101052B, 0x21010C03


def test_gait_after_stairs_from_terrain_ahead():
    th = {'rubble': 'high_step', 'potholes': 'high_step'}.get
    assert gait_after([{'kind': 'rubble', 'forward_m': 0.8}], 0.5, terrain_gait=th)[0] == 'high_step'
    assert gait_after([{'kind': 'potholes', 'trigger': True}], 0.5, terrain_gait=th)[0] == 'high_step'
    assert gait_after([], 0.5, terrain_gait=th)[0] == 'flat'
    assert gait_after([{'kind': 'rubble', 'forward_m': 0.8}], 3.0, terrain_gait=th)[0] == 'flat'
    assert gait_after(None, None)[0] == 'flat'


def test_stairs_exit_rule():
    assert stairs_exit(5.0, 0.5, False) == 'wait'            # platform between flights
    assert stairs_exit(0.5, 0.5, True) == 'wait'             # not level long enough
    assert stairs_exit(2.0, 0.2, True) == 'wait'             # rear legs still on the step
    assert stairs_exit(2.0, 0.4, True) == 'exit'
    assert stairs_exit(0.1, 0.95, True) == 'exit_cap'        # never more than ~1 cell


def node():
    n = make(SpecialObstacles, 'special.yaml', 'special_obstacles',
             gait_file=os.path.join(CFG, 'gait.yaml'))
    n.have_imu = True
    return n


def codes(n):
    return [c.args[0].cmd_code for c in n.pubs['/simple_cmd'].publish.call_args_list]


def step(n, dt, pitch_deg=0.0, dx=0.0):
    n.now_s.t += dt
    n.pitch = math.radians(pitch_deg)
    x = n._x = getattr(n, '_x', 0.0) + dx
    n.odom['/odom'] = (n.now_s.t, (x, 0.0), 0.0)
    n.tick()


def start_stairs(n):
    n.odom['/odom'] = (n.now_s.t, (0.0, 0.0), 0.0)
    n.on_start(String(data='stairs'))
    for _ in range(int(1.1 / 0.05)):
        step(n, 0.05)                                  # SETTLE -> AI on
    assert n.in_ai and codes(n)[0] == AI_IN
    for _ in range(int(3.1 / 0.05)):
        step(n, 0.05)                                  # GAIT (AUTO re-sent) -> CROSS
    assert n.phase == 'CROSS'


def test_full_stairs_up_platform_down_then_exit_and_gait_for_terrain():
    n = node()
    start_stairs(n)
    for _ in range(40):
        step(n, 0.1, 12.0, 0.01)                       # up
    for _ in range(40):
        step(n, 0.1, 0.0, 0.01)                        # platform: level, must NOT exit
    assert n.in_ai and n.phase == 'CROSS'
    for _ in range(40):
        step(n, 0.1, -12.0, 0.01)                      # down
    for _ in range(10):
        step(n, 0.1, 0.0, 0.01)                        # level 1 s, 0.1 m past: wait
    assert n.in_ai
    for _ in range(30):
        step(n, 0.1, 0.0, 0.015)                       # walks on: exit once >= 0.35 m
        if not n.in_ai:
            break
    assert not n.in_ai and n.phase == 'RESTORE'
    assert codes(n)[-1] == AI_OUT
    n.last_dets = (n.now_s.t, [{'kind': 'rubble', 'forward_m': 0.9, 'trigger': True}])
    for _ in range(int(3.6 / 0.05)):
        n.last_dets = (n.now_s.t, n.last_dets[1])
        step(n, 0.05)
    assert codes(n)[-2:] == [AUTO, HSTEP]               # AUTO, then high step for rubble
    assert n.gait.current == 'high_step' and n.phase == 'IDLE'
    n.pubs['/special/done'].publish.assert_called_with(String(data='success'))
    n.pubs['/gait/current'].publish.assert_called_with(String(data='high_step'))


def test_stop_on_the_stairs_keeps_ai_on_and_exit_ai_recovers():
    n = node()
    start_stairs(n)
    for _ in range(10):
        step(n, 0.1, 12.0, 0.01)
    n.on_start(String(data='stop'))                    # e.g. localization lost
    assert n.in_ai and n.ai_stuck and n.phase == 'IDLE'
    assert AI_OUT not in codes(n)
    n.pubs['/special/ai_active'].publish.assert_called_with(Bool(data=True))
    n.on_start(String(data='rubble'))                  # refused while AI is on
    assert n.phase == 'IDLE'
    n.on_start(String(data='exit_ai'))                 # a person walked it off
    assert codes(n)[-1] == AI_OUT and n.phase == 'RESTORE'
    for _ in range(int(3.6 / 0.05)):
        step(n, 0.05)
    assert codes(n)[-2:] == [AUTO, FLAT]                # no detector data -> flat
    n.pubs['/special/ai_active'].publish.assert_called_with(Bool(data=False))
    assert n.phase == 'IDLE' and not n.ai_stuck


def test_mission_holds_still_while_ai_is_on():
    from unittest.mock import Mock
    from lite3_explore_pose.mission_controller import MissionController
    m = object.__new__(MissionController)
    m.state, m.ai_active = 'EXPLORE', False
    m.get_logger = Mock()
    m.cancel_nav = Mock()
    m.enter = Mock(side_effect=lambda st: setattr(m, 'state', st))
    m.out_of_time = False
    m.on_ai_active(Bool(data=True))
    m.localization_ready = Mock(return_value=True)
    m.tick()
    assert m.state == 'HOLD_AI'
    m.do_hold_ai()
    assert m.state == 'HOLD_AI'
    m.on_ai_active(Bool(data=False))
    m.do_hold_ai()
    assert m.state == 'EXPLORE'
