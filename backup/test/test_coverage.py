import math
import os

import numpy as np

from lite3_explore_pose.coverage import Coverage, MapInfo, dilate
from lite3_explore_pose.mission_config import Arena


def make_coverage(bounds=None, heading=0.0):
    return Coverage(Arena({'arena': {
        'cell_size_m': 1.02,
        'start_cell_centre': [2.04, 1.02],
        'start_heading_deg': heading,
    }}), bounds)


def test_fully_mapped_side_corridor_remains_a_goal_until_walked():
    info = MapInfo(0.05, -0.55, -0.55, 63, 43)
    grid = np.full((info.height, info.width), 100)
    grid[1:21, 1:62] = 0  # main corridor along y = 0
    grid[21:42, 21:42] = 0  # side corridor along x = 1.02 (fully mapped, nothing unknown)

    coverage = make_coverage()
    for x in (0.0, 1.02, 2.04):
        coverage.observe((x, 0.0))
    targets = coverage.targets(grid, info, (2.04, 0.0), 25, 0.3)
    assert [cell for cell, _ in targets] == [(1, 1)]
    assert math.dist(targets[0][1], (1.02, 1.02)) < info.resolution

    coverage.observe((1.03, 0.92))
    assert not coverage.targets(grid, info, (1.03, 0.92), 25, 0.3)


def test_rotated_arena_limits_targets_to_clear_unvisited_cells_inside_bounds():
    info = MapInfo(0.05, -1.55, -0.55, 63, 43)
    grid = np.zeros((info.height, info.width), dtype=np.int16)
    grid[21:, :] = -1
    coverage = make_coverage(bounds=(-1.45, -0.48, 1.46, 1.48), heading=90.0)
    coverage.observe((0.0, 0.0))
    coverage.observe((2.04, 0.0))  # outside the course

    targets = coverage.targets(grid, info, (0.0, 0.0), 25, 0.3, failed={(0, -1)})
    assert coverage.visited == {(0, 0)}
    assert [cell for cell, _ in targets] == [(0, 1)]
    assert math.dist(targets[0][1], (1.02, 0.0)) < info.resolution


def test_clearance_larger_than_initial_map_dimensions():
    info = MapInfo(0.05, 0.0, 0.0, 4, 1)
    grid = np.array([[0, 0, 0, 100]])
    assert not make_coverage().targets(grid, info, (0.0, 0.0), 25, 0.3)


def test_dilate_matches_square_neighbourhood():
    rng = np.random.default_rng(0)
    for shape in [(1, 4), (3, 3), (17, 23)]:
        mask = rng.random(shape) < 0.1
        for radius in range(0, 6):
            expected = np.zeros_like(mask)
            for r, c in zip(*np.nonzero(mask)):
                expected[max(0, r - radius):r + radius + 1, max(0, c - radius):c + radius + 1] = True
            assert (dilate(mask, radius) == expected).all()


# =====================================================================================
# Camera coverage (pose history -> seen mask -> COVER goals)
# =====================================================================================

from lite3_explore_pose.coverage import (  # noqa: E402
    PoseHistory, SeenTracker, coverage_fraction, default_coverage_path, find_coverage_goals,
    load, load_poses, newest_coverage_file, save, save_poses, seen_mask, visited_mask,
    within_bounds,
)

FOV, RANGE = 70.0, 2.5


def corridor_with_side_room():
    """10 cm cells, origin (0, 0). Corridor y 0.1..0.9 along x, wall at y 0.9..1.0
    with a doorway at x 1.8..2.2, side room above it (y 1.0..2.9, x 1.0..3.0)."""
    info = MapInfo(0.1, 0.0, 0.0, 40, 30)
    grid = np.full((info.height, info.width), 100, dtype=np.int16)
    grid[1:9, 1:39] = 0      # corridor
    grid[9, 18:22] = 0       # doorway
    grid[10:29, 10:30] = 0   # side room
    return grid, info


def test_side_room_not_seen_until_a_pose_looks_in():
    grid, info = corridor_with_side_room()
    room_centre = (19, 20)   # x = 2.05, y = 1.95
    # Walk the corridor looking straight ahead (+x): SLAM would map the room, the camera does not see it.
    corridor = [(x, 0.45, 0.0, 0.0) for x in np.arange(0.5, 3.6, 0.5)]
    seen = seen_mask(grid, info, corridor, FOV, RANGE)
    assert not seen[room_centre]
    assert seen[4, 30]                     # but the corridor ahead is seen
    # Stop in front of the doorway and turn towards the room.
    seen = seen_mask(grid, info, corridor + [(2.0, 0.45, math.pi / 2, 1.0)], FOV, RANGE)
    assert seen[room_centre]


def test_raycast_stops_at_walls_and_wall_counts_as_seen():
    info = MapInfo(0.1, 0.0, 0.0, 30, 11)
    grid = np.zeros((info.height, info.width), dtype=np.int16)
    grid[:, 15] = 100                      # wall at x = 1.5..1.6
    seen = seen_mask(grid, info, [(0.55, 0.55, 0.0, 0.0)], FOV, RANGE)
    assert seen[5, 10]                     # free cell in front of the wall
    assert seen[5, 15]                     # the wall itself (QR codes are on walls)
    assert not seen[:, 16:].any()          # nothing behind the wall


def test_raycast_stops_at_unknown():
    info = MapInfo(0.1, 0.0, 0.0, 30, 11)
    grid = np.zeros((info.height, info.width), dtype=np.int16)
    grid[:, 15] = -1
    seen = seen_mask(grid, info, [(0.55, 0.55, 0.0, 0.0)], FOV, RANGE)
    assert seen[5, 14] and not seen[5, 15] and not seen[:, 16:].any()


def test_poses_line_up_after_map_origin_shift_and_resize():
    grid, info = corridor_with_side_room()
    poses = [(1.0, 0.45, 0.0, 0.0), (2.0, 0.45, math.pi / 2, 1.0), (2.0, 1.5, 2.5, 2.0)]
    tracker = SeenTracker(FOV, RANGE)
    before = tracker.update(grid, info, poses).copy()

    # slam_toolbox grows the map by 1 m to the left and 0.5 m down: the origin moves.
    pad_r, pad_c = 5, 10
    big = np.full((info.height + pad_r + 7, info.width + pad_c + 3), -1, dtype=np.int16)
    big[pad_r:pad_r + info.height, pad_c:pad_c + info.width] = grid
    big_info = MapInfo(0.1, info.origin_x - pad_c * 0.1, info.origin_y - pad_r * 0.1,
                       big.shape[1], big.shape[0])
    after = tracker.update(big, big_info, poses)

    assert after.shape == big.shape
    assert (after[pad_r:pad_r + info.height, pad_c:pad_c + info.width] == before).all()
    assert after.sum() == before.sum()     # nothing appeared in the new border


def test_incremental_update_matches_full_recompute():
    grid, info = corridor_with_side_room()
    poses = [(x, 0.45, 0.3 * x, x) for x in np.arange(0.5, 3.6, 0.5)]
    tracker = SeenTracker(FOV, RANGE)
    for i in range(1, len(poses) + 1):
        tracker.update(grid, info, poses[:i])
    assert (tracker.mask == seen_mask(grid, info, poses, FOV, RANGE)).all()


def test_pose_history_only_adds_after_move_or_turn():
    h = PoseHistory(min_move_m=0.2, min_turn_deg=15.0)
    assert h.add(0.0, 0.0, 0.0, 0.0)
    assert not h.add(0.1, 0.0, 0.0, 1.0)                    # moved 0.1 m
    assert not h.add(0.0, 0.0, math.radians(10), 2.0)       # turned 10 deg
    assert h.add(0.0, 0.0, math.radians(20), 3.0)           # turned 20 deg
    assert h.add(0.25, 0.0, math.radians(20), 4.0)          # moved 0.25 m
    assert not h.add(0.25, 0.0, math.radians(20) + 2 * math.pi, 5.0)   # same angle, wrapped
    assert len(h.poses) == 3


def test_visited_mask_is_free_cells_near_poses():
    grid, info = corridor_with_side_room()
    visited = visited_mask(grid, info, [(1.05, 0.45, 0.0, 0.0)], 0.35)
    assert visited[4, 10] and visited[4, 13] and not visited[4, 15]
    assert not visited[0, 10]              # wall cells are never "visited"


def test_goals_respect_bounds_and_clearance():
    grid, info = corridor_with_side_room()
    corridor = [(x, 0.45, 0.0, 0.0) for x in np.arange(0.5, 3.6, 0.5)]
    seen = seen_mask(grid, info, corridor, FOV, RANGE)
    clearance = 0.3

    # Bounds cover only the corridor: the unseen room must not produce goals.
    corridor_bounds = (0.0, 0.0, 4.0, 0.95)
    for g in find_coverage_goals(grid, info, seen, corridor_bounds, clearance, 5):
        assert within_bounds(g.goal_xy, corridor_bounds)

    # Bounds include the room: one goal for it, inside bounds and clear of the walls.
    all_bounds = (0.0, 0.0, 4.0, 3.0)
    goals = find_coverage_goals(grid, info, seen, all_bounds, clearance, 5)
    assert goals
    walls = np.argwhere(grid != 0)
    for g in goals:
        assert within_bounds(g.goal_xy, all_bounds)
        r, c = g.goal_cell
        nearest_wall = np.min(np.hypot(walls[:, 0] - r, walls[:, 1] - c)) * info.resolution
        assert nearest_wall >= clearance
    room = [g for g in goals if g.goal_xy[1] > 1.0]
    assert room and math.dist(room[0].goal_xy, (2.0, 1.95)) < 0.5

    # Small clusters are ignored.
    assert not find_coverage_goals(grid, info, seen, all_bounds, clearance, 10 ** 6)


def test_blacklisted_goal_moves_instead_of_dropping_the_cluster():
    grid, info = corridor_with_side_room()
    seen = np.zeros_like(grid, dtype=bool)
    seen[:10, :] = True                    # corridor seen, room not
    bounds = (0.0, 0.0, 4.0, 3.0)
    first = find_coverage_goals(grid, info, seen, bounds, 0.3, 5)
    assert len(first) == 1
    used = first[0].goal_xy
    again = find_coverage_goals(grid, info, seen, bounds, 0.3, 5, avoid=[used], avoid_radius_m=0.6)
    assert len(again) == 1
    assert math.dist(again[0].goal_xy, used) >= 0.6
    assert again[0].goal_xy[1] > 1.0       # still inside the room


def test_coverage_fraction_counts_only_free_cells_inside_bounds():
    info = MapInfo(0.1, 0.0, 0.0, 10, 1)
    grid = np.array([[0, 0, 0, 0, 100, -1, 0, 0, 0, 0]], dtype=np.int16)
    seen = np.zeros_like(grid, dtype=bool)
    seen[0, :2] = True
    assert coverage_fraction(grid, info, seen, None) == 2 / 8
    assert coverage_fraction(grid, info, seen, (0.0, 0.0, 0.4, 0.1)) == 2 / 4
    assert coverage_fraction(np.full((1, 3), -1), MapInfo(0.1, 0, 0, 3, 1),
                             np.zeros((1, 3), bool), None) == 0.0


def test_save_load_round_trip_is_identical(tmp_path):
    poses = [(0.0, 0.0, 0.0, 100.0), (1.25, -0.5, 1.5707963, 101.5), (2.0, 3.0, -3.1, 103.25)]
    params = {'camera_fov_deg': 70.0, 'camera_range_m': 2.5, 'bounds': [0.0, -1.0, 5.0, 1.0]}
    path = str(tmp_path / 'sub' / 'coverage_test.npz')
    save_poses(path, poses, params)
    loaded_poses, loaded_params = load_poses(path)
    assert loaded_poses == poses
    assert loaded_params == params
    assert not (tmp_path / 'sub' / 'coverage_test.npz.tmp').exists()

    save_poses(path, [], params)           # empty history also round-trips
    assert load_poses(path) == ([], params)


def test_save_load_names_used_by_explorer(tmp_path):
    poses = [(0.5, 0.25, 0.1, 7.0)]
    meta = {'map_frame': 'map', 'resolution': 0.05}
    path = str(tmp_path / 'coverage_x.npz')
    save(path, poses, meta)
    assert load(path) == (poses, meta)


def test_default_path_and_newest_file(tmp_path):
    p = default_coverage_path(str(tmp_path), stamp='20261120_101500')
    assert p == str(tmp_path / 'coverage_20261120_101500.npz')
    assert default_coverage_path().endswith('.npz')
    assert '/challenge_ws/coverage/coverage_' in default_coverage_path()

    assert newest_coverage_file(str(tmp_path)) is None
    old = tmp_path / 'coverage_20261120_100000.npz'
    new = tmp_path / 'coverage_20261120_110000.npz'
    save(str(old), [], {})
    save(str(new), [], {})
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    (tmp_path / 'other.npz').write_bytes(b'')          # not a coverage file
    assert newest_coverage_file(str(tmp_path)) == str(new)


def test_goal_clearance_uses_occupied_min():
    # 0.1 m cells, free 9x9 room; a cell of value 50 (uncertain, below occupied_min)
    # next to the centre must not block the goal, a wall (>= occupied_min) must.
    info = MapInfo(0.1, 0.0, 0.0, 11, 11)
    grid = np.full((11, 11), 100, dtype=np.int16)
    grid[1:10, 1:10] = 0
    seen = np.zeros_like(grid, dtype=bool)
    bounds = (0.0, 0.0, 1.1, 1.1)
    grid[5, 6] = 50
    goals = find_coverage_goals(grid, info, seen, bounds, 0.15, 5, occupied_min=65)
    assert goals and goals[0].goal_cell == (5, 5)
    grid[5, 6] = 100
    goals = find_coverage_goals(grid, info, seen, bounds, 0.15, 5, occupied_min=65)
    assert goals and goals[0].goal_cell != (5, 5)
