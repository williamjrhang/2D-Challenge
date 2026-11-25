"""Issue 10: BFS over the cell graph, routes through doorway midpoints and cell centres,
centring rule, grid-alignment estimate on synthetic maps."""
import math

import numpy as np

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


def test_explorer_routes_through_cells_and_centres_first():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from lite3_explore_pose.explorer_node import FrontierExplorer
    e = object.__new__(FrontierExplorer)
    e.P = {'cell_routing': True, 'route_kinds': ['cell'], 'centre_tolerance_m': 0.15,
           'centre_heading_deg': 15.0, 'global_frame': 'map', 'time_limit_base_s': 20.0,
           'time_limit_per_m_s': 8.0, 'no_backup_after_crossing_s': 0.0}
    e.cells = G
    e.cell_info = {c: SimpleNamespace(state='free', walls=list(w)) for c, w in WALLS.items()}
    e.visited_cells, e.wall_memory = set(), {}
    e.nav = Mock()
    from builtin_interfaces.msg import Time
    e.nav.get_clock.return_value.now.return_value.to_msg.return_value = Time()
    e.nav.goThroughPoses.return_value = True
    e.nav.goToPose.return_value = True
    e.pose_health = Mock(pose_stamp=1, odom_stamp=1)
    e.now_s = Mock(return_value=10.0)
    e.get_logger = Mock()
    e.last_pose = (0.0, 0.0, math.pi / 2)            # centred, facing the (0,1) doorway
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
