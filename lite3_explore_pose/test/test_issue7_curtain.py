"""Issue 7: curtain approach square to the doorway, passable-curtain record, far side
re-opened, crossing back when this side is done, curtain sightings judged by wrinkle."""
import json
import math
from unittest.mock import Mock

import numpy as np
from std_msgs.msg import String

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
           'approach_kinds': ['rubble', 'curtain']}
    m.arena = ARENA
    m.portals, m.obstacles = [], []
    m.portal_pub = Mock()
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


def test_curtain_needs_more_sightings_than_rubble():
    m = controller()
    m.out_of_time = False
    m.robot_xy = Mock(return_value=(0.0, 0.0))
    for _ in range(3):
        m.on_obstacle_seen(String(data=json.dumps({'kind': 'curtain', 'x': 2.0, 'y': 0.0,
                                                   'rx': 0.0, 'ry': 0.0})))
    assert m.pick_target() is None
    m.on_obstacle_seen(String(data=json.dumps({'kind': 'curtain', 'x': 2.0, 'y': 0.0,
                                               'rx': 0.0, 'ry': 0.0})))
    assert m.pick_target()['kind'] == 'curtain'


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
