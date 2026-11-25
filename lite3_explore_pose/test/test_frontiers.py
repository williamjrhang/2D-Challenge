"""Run with:  python3 -m pytest test/   (no ROS needed)"""

import numpy as np

from lite3_explore_pose.frontiers import (
    MapInfo, cluster_cells, find_frontiers, frontier_mask, score_frontiers,
    still_frontier,
)

# The example map from our slides:  ? unknown   # wall   . free   R robot (free)
SLIDE_MAP = [
    "????????",
    "?#####??",
    "?#.....?",
    "?#.R...?",
    "?#...##?",
    "?##.##??",
    "???.????",
]


def to_grid(rows):
    lut = {"?": -1, "#": 100, ".": 0, "R": 0}
    return np.array([[lut[ch] for ch in row] for row in rows], dtype=np.int16)


INFO = MapInfo(resolution=1.0, origin_x=0.0, origin_y=0.0, width=8, height=7)


def test_frontier_cells_match_slide():
    mask = frontier_mask(to_grid(SLIDE_MAP))
    assert set(zip(*map(list, np.nonzero(mask)))) == {(2, 6), (3, 6), (6, 3)}


def test_two_frontiers_gap_and_doorway():
    clusters = cluster_cells(frontier_mask(to_grid(SLIDE_MAP)))
    assert sorted(len(c) for c in clusters) == [1, 2]


def test_find_and_score():
    frontiers = find_frontiers(to_grid(SLIDE_MAP), INFO, min_size_cells=1, clearance_m=0.0)
    assert len(frontiers) == 2
    ranked = score_frontiers(frontiers, robot_xy=(3.5, 3.5), w_size=1.0, w_dist=0.1)
    # The 2-cell gap on the right beats the 1-cell doorway.
    assert ranked[0].length_m == 2.0


def test_bias_flips_choice():
    frontiers = find_frontiers(to_grid(SLIDE_MAP), INFO, min_size_cells=1, clearance_m=0.0)
    doorway_xy = INFO.cell_to_world(6, 3)
    ranked = score_frontiers(frontiers, robot_xy=(3.5, 3.5), w_size=1.0, w_dist=0.1,
                             w_bias=5.0, bias_xy=doorway_xy)
    assert ranked[0].goal_cell == (6, 3)


def test_blacklist_removes_frontier():
    frontiers = find_frontiers(to_grid(SLIDE_MAP), INFO, min_size_cells=1, clearance_m=0.0)
    gap_xy = next(f.goal_xy for f in frontiers if len(f.cells) == 2)
    ranked = score_frontiers(frontiers, robot_xy=(3.5, 3.5), blacklist=[gap_xy])
    assert len(ranked) == 1 and ranked[0].goal_cell == (6, 3)


def test_clearance_drops_goal_next_to_wall():
    # Every frontier cell here touches a wall, so a 1 m clearance removes them all.
    assert find_frontiers(to_grid(SLIDE_MAP), INFO, min_size_cells=1, clearance_m=1.0) == []


def test_still_frontier_after_explored():
    grid = to_grid(SLIDE_MAP)
    xy = INFO.cell_to_world(6, 3)
    assert still_frontier(grid, INFO, xy, radius_m=0.5)
    grid[6, :] = 100  # pretend the doorway area got mapped as wall
    assert not still_frontier(grid, INFO, xy, radius_m=0.5)
