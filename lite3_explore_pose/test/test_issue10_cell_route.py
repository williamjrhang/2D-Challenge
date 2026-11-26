"""Issue 10: BFS over the cell graph, routes through doorway midpoints and cell centres,
centring rule, grid-alignment estimate on synthetic maps."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from nav2_simple_commander.robot_navigator import TaskResult

from lite3_explore_pose.cell_route import (bfs, grid_alignment, needs_centring, neighbours,
                                           route_poses)
from lite3_explore_pose.cells import CellGrid
from lite3_explore_pose.frontiers import MapInfo
from lite3_explore_pose.mission_config import Arena

ARENA = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0, 0],
                         'start_heading_deg': 0}})
G = CellGrid(ARENA)

# 3x2 maze:  (0,1) (1,1) (2,1)
#            (0,0) (1,0) (2,0)    wall between (0,0)-(1,0) and between (1,1)-(2,1)
WALLS = {(0, 0): {'-x', '-y', '+x'}, (1, 0): {'-x', '-y'}, (2, 0): {'+x', '-y'},
         (0, 1): {'-x', '+y'}, (1, 1): {'+y', '+x'}, (2, 1): {'-x', '+x', '+y'}}


def test_neighbours_respect_walls_on_either_side():
    assert sorted(neighbours((0, 0), WALLS)) == [(0, 1)]
    assert sorted(neighbours((1, 0), WALLS)) == [(1, 1), (2, 0)]


def test_bfs_goes_round_the_walls():
    assert bfs((0, 0), (2, 1), WALLS) == [(0, 0), (0, 1), (1, 1), (1, 0), (2, 0), (2, 1)]
    assert bfs((0, 0), (9, 9), WALLS) is None


def test_route_through_doorway_midpoints_and_centres():
    path = [(0, 0), (0, 1), (1, 1)]
    pts = route_poses(path, G.centre, goal_xy=(1.1, 1.0))
    assert [tuple(round(v, 2) for v in p[:2]) for p in pts] == [
        (0.0, 0.51), (0.0, 1.02), (0.51, 1.02), (1.1, 1.0)]
    assert math.isclose(pts[0][2], math.pi / 2) and math.isclose(pts[2][2], 0.0)


def test_centring_rule():
    assert not needs_centring((0.05, 0.05, 0.1), (0.0, 0.0), 0.0)
    assert needs_centring((0.2, 0.0, 0.0), (0.0, 0.0), 0.0)            # 20 cm off
    assert needs_centring((0.0, 0.0, math.radians(20)), (0.0, 0.0), 0.0)   # 20 deg off


def walls_map(rot_deg=0.0, dx=0.0, dy=0.0):
    res = 0.05
    info = MapInfo(res, -3.0, -3.0, int(6 / res), int(6 / res))
    grid = np.zeros((info.height, info.width), dtype=np.int16)
    t = math.radians(rot_deg)
    for k in range(-2, 3):
        b = (k + 0.5) * 1.02
        for s in np.arange(-2.5, 2.5, res / 3):
            for (x, y) in ((b + dx, s + dy), (s + dx, b + dy)):
                xr, yr = math.cos(t) * x - math.sin(t) * y, math.sin(t) * x + math.cos(t) * y
                r, c = info.world_to_cell(xr, yr)
                if info.in_bounds(r, c):
                    grid[r, c] = 100
    return grid, info


def test_grid_alignment_square_start():
    est = grid_alignment(*walls_map(), ARENA)
    assert abs(est['rot_deg']) <= 0.25 and abs(est['dx_m']) < 0.03 and abs(est['dy_m']) < 0.03


def test_grid_alignment_detects_offset_and_rotation():
    est = grid_alignment(*walls_map(rot_deg=4.0, dx=0.08, dy=-0.06), ARENA)
    assert abs(est['rot_deg'] - 4.0) <= 0.5                  # walls turned +4 deg (CCW)
    est = grid_alignment(*walls_map(dx=0.08, dy=-0.06), ARENA)
    assert abs(est['dx_m'] - 0.08) < 0.02 and abs(est['dy_m'] + 0.06) < 0.02


def make_explorer(walls=WALLS, *, grid_explore=False):
    from lite3_explore_pose.explorer_node import FrontierExplorer
    e = object.__new__(FrontierExplorer)
    e.P = {'cell_routing': True, 'route_kinds': ['cell'], 'centre_tolerance_m': 0.15,
           'centre_heading_deg': 15.0, 'global_frame': 'map', 'time_limit_base_s': 20.0,
           'time_limit_per_m_s': 8.0, 'no_backup_after_crossing_s': 0.0,
           'grid_explore': grid_explore, 'dry_run': False, 'grid_max_attempts': 2,
           'blacklist_radius_m': 0.30, 'cell_visit_edge_margin_m': 0.10,
           'pose_goal_tolerance_m': 0.35, 'pose_result_wait_s': 2.0}
    e.cells = G
    e.cell_info = {
        cell: SimpleNamespace(state='free', walls=list(sides), centre=G.centre(cell),
                              goal=G.centre(cell))
        for cell, sides in walls.items()
    }
    e.visited_cells, e.wall_memory = set(), {}
    e.cell_attempts, e.deferred_goals = {}, {}
    e.nav = Mock()
    e.nav.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    e.nav.goThroughPoses.return_value = True
    e.nav.goToPose.return_value = True
    e.pose_health = Mock(pose_stamp=1, odom_stamp=1)
    e.now_s = Mock(return_value=10.0)
    e.get_logger = Mock()
    e.set_state = Mock()
    e.last_pose = (0.0, 0.0, math.pi / 2)            # centred, facing the (0,1) doorway
    return e


def test_explorer_routes_through_cells_and_centres_first():
    e = make_explorer()
    assert e.send_goal(G.centre((1, 1)), (0.0, 0.0), is_frontier=False, kind='cell')
    poses = e.nav.goThroughPoses.call_args.args[0]
    assert [(round(p.pose.position.x, 2), round(p.pose.position.y, 2)) for p in poses] == [
        (0.0, 0.51), (0.0, 1.02), (0.51, 1.02), (1.02, 1.02)]
    e.nav.goToPose.assert_not_called()
    # off-centre: centre first, then the route once that is done
    e.nav.reset_mock()
    e.last_pose = (0.3, -0.1, 0.0)
    e.send_goal(G.centre((1, 1)), (0.3, -0.1), is_frontier=False, kind='cell')
    centre = e.nav.goToPose.call_args.args[0].pose
    assert (centre.position.x, centre.position.y) == (0.0, 0.0)
    e.nav.goThroughPoses.assert_not_called()
    e.nav.isTaskComplete.return_value = True
    e.do_navigate((0.0, 0.0, math.pi / 2))
    assert len(e.nav.goThroughPoses.call_args.args[0]) == 4 and e.route_pending is None
    # unknown goal cell: direct goal
    e.nav.reset_mock()
    e.send_goal((5.0, 5.0), (0.0, 0.0), is_frontier=False, kind='cell')
    e.nav.goToPose.assert_called_once()
    e.nav.goThroughPoses.assert_not_called()


def test_grid_uses_visited_hop_and_skips_disconnected_cell():
    walls = {(0, 0): {'+x'}, (0, 1): set(), (1, 1): {'-y'}, (1, 0): {'-x', '+y'}}
    e = make_explorer(walls, grid_explore=True)
    e.visited_cells = {(0, 0), (0, 1)}
    e.send_goal = Mock(return_value=True)

    assert e.start_cell(e.last_pose)

    assert e.send_goal.call_args.args[0] == G.centre((0, 1))
    e.set_state.assert_called_once_with('NAVIGATE')


def test_grid_prefers_shorter_open_route_over_geometrically_nearer_cell():
    walls = {(0, 0): {'+x'}, (0, 1): set(), (1, 1): set(), (1, 0): {'-x'},
             (0, -1): set(), (0, -2): set()}
    e = make_explorer(walls, grid_explore=True)
    e.visited_cells = {(0, 0), (0, 1), (1, 1), (0, -1)}
    e.send_goal = Mock(return_value=True)

    assert e.start_cell(e.last_pose)

    assert e.send_goal.call_args.args[0] == G.centre((0, -1))


def test_grid_dispatches_only_the_adjacent_cell():
    e = make_explorer(grid_explore=True)

    assert e.send_goal(G.centre((0, 1)), e.last_pose[:2], is_frontier=False, kind='cell')

    e.nav.goToPose.assert_called_once()
    goal = e.nav.goToPose.call_args.args[0].pose
    assert (goal.position.x, goal.position.y) == G.centre((0, 1))
    e.nav.goThroughPoses.assert_not_called()


def test_grid_rejects_a_direct_goal_without_an_open_shared_side():
    e = make_explorer(grid_explore=True)

    for cell in [(1, 0), (1, 1)]:
        assert not e.send_goal(G.centre(cell), e.last_pose[:2], is_frontier=False, kind='cell')

    e.nav.goToPose.assert_not_called()
    e.nav.goThroughPoses.assert_not_called()


@pytest.mark.parametrize('result', [TaskResult.SUCCEEDED, TaskResult.FAILED])
def test_grid_waits_for_successful_centring_before_leaving(result):
    e = make_explorer(grid_explore=True)
    e.last_pose = (0.3, -0.1, 0.0)
    e.start_recovery = Mock()

    assert e.send_goal(G.centre((0, 1)), e.last_pose[:2], is_frontier=False, kind='cell')
    e.nav.goToPose.assert_called_once()
    centre = e.nav.goToPose.call_args.args[0].pose
    assert (centre.position.x, centre.position.y) == G.centre((0, 0))
    assert math.isclose(centre.orientation.z, math.sin(math.pi / 4))

    e.nav.isTaskComplete.return_value = True
    e.nav.getResult.return_value = result
    e.pose_health.pose_stamp = e.pose_health.odom_stamp = 2
    e.do_navigate((0.0, 0.0, math.pi / 2))

    if result == TaskResult.SUCCEEDED:
        assert e.nav.goToPose.call_count == 2
        neighbour = e.nav.goToPose.call_args.args[0].pose
        assert (neighbour.position.x, neighbour.position.y) == G.centre((0, 1))
        e.start_recovery.assert_not_called()
    else:
        e.nav.goToPose.assert_called_once()
        e.start_recovery.assert_called_once()
    e.nav.goThroughPoses.assert_not_called()
