"""Issue 5: scene answer -> stop and look; stale photo -> look again; QR -> wait for placement."""
import json
from unittest.mock import Mock

from std_msgs.msg import Bool, String

from lite3_explore_pose.investigate import LookCooldown, photo_is_stale
from lite3_explore_pose.mission_config import Arena
from lite3_explore_pose.mission_controller import MissionController


def test_photo_is_stale():
    # look sent at 100, answer arrives at 103 after a 5 s ask -> photo from ~98: stale
    assert photo_is_stale(100.0, 103.0, 5.0)
    assert not photo_is_stale(100.0, 106.0, 5.0)        # photo ~101: fresh
    assert not photo_is_stale(100.0, 103.0, None)        # unknown latency: don't loop


def test_cell_cooldown_and_scene_once_per_cell():
    c = LookCooldown(60.0)
    assert c.may_scene_look((1, 0), 0.0)
    c.mark((1, 0), 0.0, scene=True)
    assert not c.may_scene_look((1, 0), 100.0)           # scene: only once per cell
    assert c.may_look((1, 0), 61.0) and not c.may_look((1, 0), 30.0)
    assert c.may_scene_look((2, 0), 1.0)


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def controller():
    m = object.__new__(MissionController)
    m.P = {'scene_investigate': True, 'investigate_settle_s': 1.0, 'max_relooks': 1,
           'stale_margin_s': 0.3, 'look_cell_cooldown_s': 60.0, 'qr_wait_place': True,
           'investigate_timeout_s': 20.0, 'look_cooldown_s': 5.0, 'pose_jump_m': 0.0,
           'require_localization_ready': False, 'tick_hz': 5.0}
    m.arena = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0, 0],
                               'start_heading_deg': 0}})
    m.now_s = Clock()
    m.get_logger = Mock()
    m.robot_xy = Mock(return_value=(1.0, 0.1))
    m.look_pub, m.explore_pub, m.state_pub = Mock(), Mock(), Mock()
    m.special_pub, m.launch_pub = Mock(), Mock()
    m.looks = LookCooldown(60.0)
    m.state, m.state_since = 'EXPLORE', 100.0
    m.pending_look = m.pending_special = False
    m.look_reason = m.look_cell = m.look_sent_t = None
    m.relooks = 0
    m.look_done = False
    m.last_latency = m.qr_event_t = m.last_qr_found_t = None
    m.last_look_end = 0.0
    m.out_of_time = False
    m.obstacles, m.target = [], None
    m.tasks_done = m.explore_done = False
    m.last_xy = None
    m.start_time, m.time_limit, m.exit_margin = 0.0, 0.0, 0.0
    m.exit_xy, m.waiting_for_exit, m.nav = None, False, None
    return m


def run_to_investigate(m):
    m.tick()
    assert m.state == 'INVESTIGATE'
    m.look_pub.publish.assert_not_called()               # settling first, robot still
    m.now_s.t += 1.2
    m.tick()
    m.look_pub.publish.assert_called_once_with(Bool(data=True))


def test_scene_answer_stops_settles_looks_and_relooks_once_on_a_stale_photo():
    m = controller()
    m.on_room(String(data='kitchen'))
    run_to_investigate(m)
    assert m.look_cell == (1, 0)
    # vlm_watcher was busy: answer after 2 s with latency 6 s -> photo before the look
    m.now_s.t += 2.0
    m.on_scene(String(data=json.dumps({'scene_type': 'kitchen', 'latency_s': 6.0})))
    m.on_look_done(Bool(data=True))
    m.tick()
    assert m.state == 'INVESTIGATE' and m.look_pub.publish.call_count == 2
    m.now_s.t += 5.0
    m.on_scene(String(data=json.dumps({'scene_type': 'kitchen', 'latency_s': 4.5})))
    m.on_look_done(Bool(data=True))
    m.tick()
    assert m.state == 'EXPLORE'
    # same cell, new scene answer: no second stop
    m.on_room(String(data='kitchen'))
    assert not m.pending_look


def test_qr_look_waits_until_the_code_was_placed():
    m = controller()
    m.on_event(String(data='qr'))
    run_to_investigate(m)
    m.on_look_done(Bool(data=True))
    m.tick()
    assert m.state == 'INVESTIGATE'                      # QR not placed yet
    m.now_s.t += 1.0
    m.on_found(String(data=json.dumps({'task_type': 'qr', 'message': 'HELLO'})))
    m.tick()
    assert m.state == 'EXPLORE'


def test_dead_desktop_never_blocks_the_robot():
    m = controller()
    m.on_room(String(data='clinic'))
    run_to_investigate(m)
    m.now_s.t += 21.0                                    # no look_done ever
    m.tick()
    assert m.state == 'EXPLORE'


def test_room_view_waits_for_look_done_then_relooks_stale_once():
    from lite3_explore_pose.explorer_node import FrontierExplorer
    e = object.__new__(FrontierExplorer)
    e.P = {'room_dwell_s': 2.5, 'room_look_wait_done': True, 'room_look_timeout_s': 15.0,
           'room_max_relooks': 1}
    e.room = Mock(cell=(2, 1), views=[('room', 0.0)])
    e.room_idx = 0
    e.now_s = Clock()
    e.get_logger = Mock()
    e.publish_cmd = Mock()
    e.look_pub = Mock()
    e.look_done_t = e.scene_latency = None
    e.start_room_dwell()
    assert not e.room_view_finished(102.6)              # dwell over, no answer yet
    e.on_scene(String(data=json.dumps({'latency_s': 7.0})))   # arrives at t=100 (clock)
    e.now_s.t = 103.0
    e.on_scene(String(data=json.dumps({'latency_s': 7.0})))
    e.on_look_done(Bool(data=True))
    assert not e.room_view_finished(103.0)              # photo ~96 < look 100: look again
    assert e.look_pub.publish.call_count == 2
    e.now_s.t = 109.0
    e.on_scene(String(data=json.dumps({'latency_s': 4.0})))
    e.on_look_done(Bool(data=True))
    assert e.room_view_finished(109.0)
