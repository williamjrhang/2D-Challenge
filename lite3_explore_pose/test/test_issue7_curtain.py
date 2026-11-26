"""Issue 7: curtain approach square to the doorway, passable-curtain record, far side
re-opened, crossing back when this side is done, curtain sightings judged by wrinkle."""
import json
import math
from unittest.mock import Mock

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker

from lite3_explore_pose.crossing import approach_pose, far_side
from lite3_explore_pose.explorer_node import FrontierExplorer
from lite3_explore_pose.frontier_record import FrontierRecord
from lite3_explore_pose.cells import CellGrid
from lite3_explore_pose.mission_config import Arena
from lite3_explore_pose.mission_controller import MissionController

ARENA = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0, 0],
                         'start_heading_deg': 0}})


def test_curtain_approach_is_square_to_the_doorway():
    # curtain in the doorway between cells (1,0) and (2,0) at x=1.53, seen from an angle
    x, y, yaw = approach_pose((1.55, 0.18), (0.2, -0.6), 0.5, ARENA, square=True)
    assert math.isclose(x, 1.05, abs_tol=1e-6) and math.isclose(y, 0.0, abs_tol=1e-6)
    assert math.isclose(yaw, 0.0, abs_tol=1e-6)
    # rubble keeps the old 'face it from where it was seen' rule
    x, y, yaw = approach_pose((2.0, 1.0), (0.0, 1.0), 0.6)
    assert (round(x, 2), round(y, 2), round(yaw, 2)) == (1.4, 1.0, 0.0)


def test_far_side():
    assert far_side((3.0, 0.0), (1.53, 0.0), (1.0, 0.0), (2.5, 0.0))
    assert not far_side((0.0, 0.0), (1.53, 0.0), (1.0, 0.0), (2.5, 0.0))


def test_explorer_reopens_goals_behind_a_crossed_curtain():
    e = object.__new__(FrontierExplorer)
    e.blacklist = [(3.0, 0.0), (0.0, 1.0)]
    e.deferred_goals = {(2.5, 0.5): 999.0}
    e.frontier_rec = FrontierRecord()
    e.frontier_rec.add((3.0, 0.0), (3, 0), 1.0)
    e.frontier_rec.failed((3.0, 0.0), 2.0, 'Nav2 ABORTED')
    e.cells = CellGrid(ARENA)
    e.cell_attempts = {(3, 0): 2, (0, 1): 2}
    e.room_status = {(3, 1): {'attempts': 2, 'done': False}}
    e.get_logger = Mock()
    e.on_portal(String(data=json.dumps({'xy': [1.53, 0.0], 'from': [1.0, 0.0],
                                        'to': [2.5, 0.0]})))
    assert e.blacklist == [(0.0, 1.0)] and e.deferred_goals == {}
    assert e.frontier_rec.entries[0]['status'] == 'pending'
    assert e.cell_attempts == {(3, 0): 0, (0, 1): 2}
    assert e.room_status[(3, 1)]['attempts'] == 0
    assert not e.final_retry_done


def controller():
    m = object.__new__(MissionController)
    m.P = {'approach_merge_m': 0.8, 'portal_return': True, 'portal_max_crossings': 3,
           'approach_min_sightings': 3, 'curtain_min_sightings': 4,
           'approach_kinds': ['rubble', 'curtain'], 'global_frame': 'map'}
    m.arena = ARENA
    m.portals, m.obstacles = [], []
    m.room_priority = False
    m.portal_pub = Mock()
    m.curtain_marker_pub = Mock()
    m.get_clock = Mock()
    m.get_clock().now().to_msg.return_value = Time()
    m.get_logger = Mock()
    m.nav = Mock()
    m.enter = Mock(side_effect=lambda st: setattr(m, 'state', st))
    m.localization_ready = Mock(return_value=True)
    return m


def test_crossed_curtain_is_recorded_and_crossed_back_when_this_side_is_done():
    m = controller()
    m.target = {'kind': 'curtain', 'x': 1.53, 'y': 0.0}
    m.traverse_xy = (1.0, 0.0)
    m.robot_xy = Mock(return_value=(2.6, 0.0))
    m.record_portal()
    sent = json.loads(m.portal_pub.publish.call_args.args[0].data)
    assert sent['from'] == [1.0, 0.0] and sent['to'] == [2.6, 0.0] and sent['crossings'] == 1
    m.robot_xy.return_value = (3.5, 0.5)                 # explored everything behind it
    m.explore_done = True
    assert m.portal_back() and m.state == 'APPROACH'
    assert m.target['portal'] == 0 and (m.target['rx'], m.target['ry']) == (3.5, 0.5)
    m.robot_xy.return_value = (0.2, 0.0)                 # on the near side: nothing to do
    m.state = 'EXPLORE'
    assert not m.portal_back()


def test_curtain_location_is_saved_before_approach():
    m = controller()
    m.out_of_time = False
    m.robot_xy = Mock(return_value=(0.0, 0.0))

    def sighting(x, y):
        return String(data=json.dumps({'kind': 'curtain', 'x': x, 'y': y,
                                       'rx': 0.0, 'ry': 0.0}))

    def pins(msg):
        return [marker for marker in msg.markers if marker.action == Marker.ADD]

    m.on_obstacle_seen(sighting(2.0, 0.0))
    log = m.get_logger().info
    assert any(call.args[0].startswith('CURTAIN: candidate location recorded')
               for call in log.call_args_list)
    m.on_obstacle_seen(sighting(2.0, 0.0))
    m.on_obstacle_seen(sighting(2.0, 0.0))
    assert not any(call.args[0].startswith('CURTAIN: approach location saved')
                   for call in log.call_args_list)
    assert all(not pins(call.args[0])
               for call in m.curtain_marker_pub.publish.call_args_list)
    assert m.pick_target() is None

    m.on_obstacle_seen(sighting(2.0, 0.0))
    assert any(call.args[0].startswith('CURTAIN: approach location saved')
               for call in log.call_args_list)
    m.enter.assert_not_called()
    (saved,) = pins(m.curtain_marker_pub.publish.call_args.args[0])
    assert saved.header.frame_id == 'map'
    assert (saved.pose.position.x, saved.pose.position.y) == (2.0, 0.0)

    m.on_obstacle_seen(sighting(2.5, 0.5))
    (updated,) = pins(m.curtain_marker_pub.publish.call_args.args[0])
    assert (updated.ns, updated.id) == (saved.ns, saved.id)
    assert (updated.pose.position.x, updated.pose.position.y) == (2.1, 0.1)
    m.portal_pub.publish.assert_not_called()

    m.do_explore()
    m.enter.assert_called_once_with('APPROACH')
    assert (m.target['kind'], m.target['x'], m.target['y']) == ('curtain', 2.1, 0.1)


def test_room_priority_defers_a_rubble_detour_until_the_room_is_clear():
    m = controller()
    m.tasks_done = m.explore_done = False
    m.P['look_every_s'] = 0.0
    rubble = {'kind': 'rubble', 'x': 1.02, 'y': 0.0}
    m.pick_target = Mock(return_value=rubble)

    m.on_room_priority(Bool(data=True))
    m.do_explore()
    m.pick_target.assert_not_called()
    m.enter.assert_not_called()

    m.on_room_priority(Bool(data=False))
    m.do_explore()
    assert m.target is rubble
    m.enter.assert_called_once_with('APPROACH')


def test_confirmed_crossing_can_interrupt_a_room_visit():
    m = controller()
    m.state = 'EXPLORE'
    m.P['pose_jump_m'] = 0.0
    m.time_limit = 0.0
    m.now_s = Mock(return_value=10.0)
    m.start_time = 0.0
    m.pending_special = 'rubble'
    m.cancel_nav = Mock()
    m.on_room_priority(Bool(data=True))

    m.tick()

    assert m.current_special == 'rubble'
    m.cancel_nav.assert_called_once_with()
    m.enter.assert_called_once_with('TRAVERSE')


def test_close_curtain_event_requires_completed_alignment():
    m = controller()
    m.P['global_frame'] = 'map'
    m.now_s = Mock(return_value=20.0)
    m.last_special_end = float('-inf')
    m.pending_special = None
    m.state = 'EXPLORE'
    m.target = None
    m.arrived_at = None
    m.approach_yaw = math.radians(179.0)
    transform = TransformStamped()
    m.tf_buffer = Mock()
    m.tf_buffer.lookup_transform.return_value = transform
    event = String(data='curtain_ahead')

    m.on_event(event)
    assert m.pending_special is None

    m.state = 'APPROACH'
    m.target = {'kind': 'curtain'}
    m.on_event(event)
    assert m.pending_special is None
    m.tf_buffer.lookup_transform.assert_not_called()

    m.arrived_at = 19.0
    off_axis_yaw = math.radians(-146.0)  # 35 degrees away across the +/-180 boundary
    transform.transform.rotation.z = math.sin(off_axis_yaw / 2.0)
    transform.transform.rotation.w = math.cos(off_axis_yaw / 2.0)
    m.on_event(event)
    assert m.pending_special is None

    aligned_yaw = math.radians(-179.0)
    transform.transform.rotation.z = math.sin(aligned_yaw / 2.0)
    transform.transform.rotation.w = math.cos(aligned_yaw / 2.0)
    m.target = {'kind': 'rubble'}
    m.on_event(event)
    assert m.pending_special is None

    m.target = {'kind': 'curtain'}
    m.on_event(event)
    assert m.pending_special == 'curtain'


def test_curtain_approach_waits_for_new_explorer_pause():
    m = controller()
    m.P.update(global_frame='map', approach_timeout_s=30.0, curtain_standoff_m=0.5)
    m.state = 'APPROACH'
    m.state_since = 10.0
    m.now_s = Mock(return_value=12.0)
    m.explorer_paused_at = 9.0
    m.target = {'kind': 'curtain', 'x': 1.55, 'y': 0.18, 'rx': 0.2, 'ry': -0.6}
    m.approach_sent = False
    m.nav.get_clock().now().to_msg.return_value = Time()
    m.nav.goToPose.return_value = True
    m.nav.isTaskComplete.return_value = False

    m.do_approach()
    assert not m.approach_sent
    m.nav.goToPose.assert_not_called()

    m.explorer_paused_at = 11.0
    m.do_approach()
    m.nav.goToPose.assert_not_called()

    m.nav.isTaskComplete.return_value = True
    m.do_approach()
    assert m.approach_sent
    goal = m.nav.goToPose.call_args.args[0]
    assert goal.header.frame_id == 'map'
    assert math.isclose(goal.pose.position.x, 1.05)
    assert goal.pose.position.y == 0.0
    assert goal.pose.orientation.w == 1.0


def test_explorer_acknowledges_pause_after_nav2_action_finishes():
    e = object.__new__(FrontierExplorer)
    e.P = {'coverage_enabled': False, 'dry_run': False}
    e.robot_pose = Mock(return_value=None)
    e.localization_pub = Mock()
    e.room_priority_pub = Mock()
    e.grid = None
    e.now_s = Mock(return_value=11.0)
    e.deferred_goals = {}
    e.update_cells = Mock()
    e.enabled = False
    e.pose_retry_pending = e.pose_cancel_pending = False
    e.state = 'NAVIGATE'
    e.nav = Mock()
    e.nav.isTaskComplete.return_value = False
    e.get_logger = Mock()
    e.warn_idle = Mock()
    e.publish_cmd = Mock()
    e.status_pub = Mock()

    e.tick()
    e.nav.cancelTask.assert_called_once()
    assert all(call.args[0].data != 'PAUSED' for call in e.status_pub.publish.call_args_list)

    e.nav.isTaskComplete.return_value = True
    e.tick()
    assert e.state == 'PAUSED'
    assert e.status_pub.publish.call_args.args[0].data == 'PAUSED'


def test_curtain_confirmation_repeats_during_approach_only(monkeypatch):
    from lite3_explore_pose import obstacle_detector

    monkeypatch.setattr(obstacle_detector, 'time', Mock(monotonic=Mock(return_value=105.0)))
    d = object.__new__(obstacle_detector.ObstacleDetector)
    d.P = {'cooldown_s': 15.0, 'publish_events': True}
    d.last_event = {'curtain': 100.0, 'rubble': 100.0}
    d.get_logger = Mock()
    d.event_pub = Mock()
    dets = [{'kind': 'curtain', 'conf': 0.4}, {'kind': 'rubble', 'conf': 0.9}]

    d.mission_state = 'EXPLORE'
    d.send('curtain', dets)
    d.mission_state = 'APPROACH'
    d.send('rubble', dets)
    d.event_pub.publish.assert_not_called()

    d.send('curtain', dets)
    d.event_pub.publish.assert_called_once_with(String(data='curtain_ahead'))


def test_curtain_sighting_flat_wall_rejected_fabric_reported_far_by_sightings_only():
    from lite3_explore_pose.obstacle_detector import ObstacleDetector
    rng = np.random.default_rng(1)
    floor = -0.35

    def surface(x0, folds, left=0.3, n=20000):
        y = rng.uniform(left - 0.45, left + 0.45, n)
        z = rng.uniform(floor, floor + 0.9, n)
        x = x0 + (0.015 * np.sin(2 * np.pi * y / 0.12) if folds else 0 * y) \
            + rng.normal(0, 0.0015, n)
        return np.c_[x, y, z]
    d = object.__new__(ObstacleDetector)
    d.P = {'curtain_sighting_range_m': 2.5, 'curtain_wrinkle_range_m': 1.0,
           'curtain_flat_check': True, 'curtain_min_wrinkle_mm': 2.5}
    d.floor_z = floor
    d.get_logger = Mock()
    d.report_sighting = Mock()
    d.base_points = Mock(return_value=surface(0.8, folds=False))
    d.curtain_sighting({}, 0.8, 0.3, 0.4)
    d.report_sighting.assert_not_called()                 # flat at 0.8 m: a wall
    d.base_points.return_value = surface(0.8, folds=True)
    d.curtain_sighting({}, 0.8, 0.3, 0.4)
    assert d.report_sighting.call_args.args[:3] == ('curtain', 0.8, 0.3)
    d.report_sighting.reset_mock()
    d.curtain_sighting({}, 2.2, 0.0, 0.4)                 # far: no wrinkle judgement
    assert d.report_sighting.call_args.args[3]['wrinkle_mm'] is None
    d.report_sighting.reset_mock()
    d.curtain_sighting({}, 3.0, 0.0, 0.4)                 # beyond range
    d.report_sighting.assert_not_called()
