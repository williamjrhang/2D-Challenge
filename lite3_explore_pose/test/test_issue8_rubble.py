"""Issue 8: rubble crossing finishes on odometry (never on time alone), VERIFY after the
switch back to flat reverts to the terrain gait when stuck, no back-up after a crossing."""
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import Mock

from std_msgs.msg import String

import lite3_explore_pose.special_obstacles as so
from lite3_explore_pose.crossing import odom_jumped, verify_result, walked_enough
from lite3_explore_pose.special_obstacles import SpecialObstacles
from node_fakes import CFG, make

if so.MotionSimpleCMD is None:
    so.MotionSimpleCMD = lambda cmd_code, type: SimpleNamespace(cmd_code=cmd_code)
FLAT, HSTEP = 0x21010300, 0x21010407


def test_pure_rules():
    assert not walked_enough(0.3, 3.0, 2.0, odom_fresh=True)    # stuck: time alone never
    assert walked_enough(2.1, 1.5, 2.0, odom_fresh=True)
    assert not walked_enough(2.1, 1.0, 2.0, odom_fresh=True)    # odometry jump suspicion
    assert walked_enough(None, 2.7, 2.0, odom_fresh=False)      # no odometry: 130 %
    assert odom_jumped(1.0, 12.3, 1.0) and not odom_jumped(1.0, 1.1, 1.0)
    assert verify_result(0.02, 0.1, 1.0, 6.0) == 'stuck'
    assert verify_result(0.2, 0.1, 9.0, 6.0) == 'tilted'
    assert verify_result(0.2, 0.1, 2.0, 6.0) == 'ok'


def node(**over):
    n = make(SpecialObstacles, 'special.yaml', 'special_obstacles',
             gait_file=os.path.join(CFG, 'gait.yaml'), **over)
    n.have_imu = True
    n._x = 0.0
    return n


def codes(n):
    return [c.args[0].cmd_code for c in n.pubs['/simple_cmd'].publish.call_args_list]


def step(n, dt=0.05, dx=0.0, dets=None, pitch=0.0):
    n.now_s.t += dt
    n._x += dx
    n.pitch = math.radians(pitch)
    n.odom['/odom'] = (n.now_s.t, (n._x, 0.0), 0.0)
    if dets is not None:
        n.last_dets = (n.now_s.t, dets)
    n.tick()


RUBBLE = [{'kind': 'rubble', 'forward_m': 0.5, 'trigger': True, 'bump': 0.4}]


def start_rubble(n):
    n.odom['/odom'] = (n.now_s.t, (0.0, 0.0), 0.0)
    n.last_dets = (n.now_s.t, RUBBLE)
    n.on_start(String(data='rubble'))
    for _ in range(80):                       # SETTLE 1 s + GAIT 2 s
        step(n, dets=RUBBLE)
        if n.phase == 'CROSS':
            break
    assert n.phase == 'CROSS' and codes(n)[-1] == HSTEP


def test_stuck_with_fresh_odometry_never_finishes_on_time():
    n = node()
    start_rubble(n)                           # walk_m = 0.5 + 1.02 + 0.45 = 1.97
    for _ in range(int(30 / 0.05)):           # 30 s, odometry shows 2 cm/s (0.6 m)
        step(n, dx=0.001, dets=[])            # old rule: speed x time 3.6 m -> 'success'
    assert n.phase == 'CROSS'                 # new: odometry decides, keeps going
    for _ in range(int(20 / 0.05)):           # until max_time_s (45 s) aborts it
        step(n, dx=0.001, dets=[])
    assert n.phase == 'IDLE'
    results = [c.args[0] for c in n.pubs['/special/done'].publish.call_args_list]
    assert results == [String(data='failed')]


def test_clear_floor_early_still_needs_the_full_distance_then_verify_ok():
    n = node()
    start_rubble(n)
    walked_at_finish = None
    for _ in range(int(40 / 0.05)):
        step(n, dx=0.006, dets=[])            # detector: floor clear from the start
        if n.phase != 'CROSS':
            walked_at_finish = n._x
            break
    assert walked_at_finish >= 1.97 - 1e-6    # not after clear_extra 0.7 m
    assert n.phase == 'VERIFY' and codes(n)[-1] == FLAT
    for _ in range(int(4.5 / 0.05)):
        step(n, dx=0.004)                     # moving fine in flat gait
    assert n.phase == 'IDLE'
    rep = json.loads(n.pubs['/special/report'].publish.call_args.args[0].data)
    assert rep['result'] == 'success' and rep['verify_reverts'] == 0


def test_verify_stuck_reverts_to_previous_gait_then_succeeds():
    n = node()
    start_rubble(n)
    while n.phase == 'CROSS':
        step(n, dx=0.006, dets=[])
    assert n.phase == 'VERIFY'
    for _ in range(int(4.5 / 0.05)):
        step(n, dx=0.0)                       # flat gait, no progress: stuck on the rubble
    assert codes(n)[-1] == HSTEP and n.phase in ('GAIT', 'CROSS')
    assert n.verify_reverts == 1 and math.isclose(n.walk_m, 0.5)
    for _ in range(int(20 / 0.05)):
        step(n, dx=0.006, dets=[])
        if n.phase == 'VERIFY':
            break
    for _ in range(int(4.5 / 0.05)):
        step(n, dx=0.004)
    rep = json.loads(n.pubs['/special/report'].publish.call_args.args[0].data)
    assert rep == {'kind': 'rubble', 'result': 'success', 'verify_reverts': 1, 't': rep['t']}


def test_odometry_jump_switches_to_time_mode():
    n = node()
    start_rubble(n)
    step(n, dx=0.006, dets=RUBBLE)
    step(n, dx=11.0, dets=RUBBLE)             # FAST-LIO jump
    assert n.odom_src is None and n.phase == 'CROSS'


def test_mission_does_not_ignore_same_kind_after_a_revert():
    from lite3_explore_pose.mission_controller import MissionController
    m = object.__new__(MissionController)
    m.P = {'require_localization_ready': False, 'same_kind_ignore_s': 10.0}
    m.state, m.pending_special = 'EXPLORE', None
    m.last_special_end, m.last_special_kind = 100.0, 'rubble'
    m.now_s = Mock(return_value=103.0)
    m.get_logger = Mock()
    m.on_event(String(data='rubble_ahead'))
    assert m.pending_special is None
    m.on_special_report(String(data=json.dumps({'kind': 'rubble', 'verify_reverts': 1})))
    m.on_event(String(data='rubble_ahead'))
    assert m.pending_special == 'rubble'


def test_explorer_no_backup_right_after_a_crossing():
    from lite3_explore_pose.explorer_node import FrontierExplorer
    e = object.__new__(FrontierExplorer)
    e.P = {'no_backup_after_crossing_s': 30.0, 'no_backup_bt_dir': os.path.join(CFG, 'bt'),
           'backup_time_s': 2.0}
    e.now_s = Mock(return_value=100.0)
    e.get_logger = Mock()
    assert e.bt_file('navigate_to_pose') == ''
    e.on_special_report(String(data=json.dumps({'kind': 'rubble'})))
    e.now_s.return_value = 110.0
    path = e.bt_file('navigate_to_pose')
    assert path.endswith('navigate_to_pose_no_backup.xml') and '<BackUp' not in open(path).read()
    e.now_s.return_value = 131.0
    assert e.bt_file('navigate_to_pose') == ''
