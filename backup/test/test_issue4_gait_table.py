"""Issue 4: one gait table, untested codes refused, one owner publishing /gait/current."""
import os
from unittest.mock import Mock

from std_msgs.msg import String

from lite3_explore_pose.gait import GaitState, GaitTable
from lite3_explore_pose.special_obstacles import SpecialObstacles

CFG = os.path.join(os.path.dirname(__file__), '..', 'config', 'gait.yaml')


def test_installed_table_has_the_confirmed_codes():
    t, errors = GaitTable.load(CFG)
    assert errors == []
    assert t.code('flat') == 0x21010300 == 553714432
    assert t.code('high_step') == 0x21010407 == 553714695
    assert t.code('ai_enter') == 0x21010528 and t.code('ai_exit') == 0x2101052B
    assert t.code('auto') == 0x21010C03
    assert t.for_terrain('rubble') == 'high_step' == t.for_terrain('potholes')
    assert t.for_terrain('curtain') is None


def test_untested_or_bad_codes_are_disabled():
    t, errors = GaitTable.from_dict({'codes': {'flat': '0x21010300', 'grip': '0x21010402',
                                               'junk': 'abc'},
                                     'terrain': {'rubble': 'nope'}})
    assert t.code('grip') == 0 and t.code('junk') == 0 and t.code('flat') == 0x21010300
    assert len(errors) == 3
    assert t.for_terrain('rubble') is None
    assert t.code('missing') == 0


def owner():
    n = object.__new__(SpecialObstacles)
    n.gaits, _ = GaitTable.load(CFG)
    n.gait = GaitState()
    n.P = {'gait_via_manual': False}
    n.phase, n.kind, n.in_ai = 'IDLE', None, False
    n.gait_pub = Mock()
    n.gait_cur_pub = Mock()
    n.get_logger = Mock()
    n.now_s = Mock(return_value=1.0)
    return n


def test_owner_switches_logs_and_publishes_current():
    import lite3_explore_pose.special_obstacles as so
    so.MotionSimpleCMD = so.MotionSimpleCMD or (lambda cmd_code, type: Mock(cmd_code=cmd_code))
    n = owner()
    n.on_gait_request(String(data='high_step'))
    assert n.gait.current == 'high_step'
    n.gait_cur_pub.publish.assert_called_with(String(data='high_step'))
    assert n.gait_pub.publish.call_args.args[0].cmd_code == 0x21010407
    n.phase = 'CROSS'
    n.on_gait_request(String(data='flat'))                 # busy crossing: ignored
    assert n.gait.current == 'high_step'
    n.phase = 'IDLE'
    n.on_gait_request(String(data='grip'))                 # not a gait name: ignored
    assert n.gait.current == 'high_step'
    assert n.set_gait('ai', 'stairs') and n.gait.current == 'ai'
