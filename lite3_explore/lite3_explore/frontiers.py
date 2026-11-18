"""Frontier detection and scoring on an occupancy grid.

Pure numpy, no ROS imports, so it can be unit-tested on a laptop.

Occupancy grid values (nav_msgs/OccupancyGrid):
    -1      unknown
    0..100  probability of being occupied (0 = free, 100 = wall)

A frontier cell is a FREE cell that touches an UNKNOWN cell (4-neighbours).
Frontier cells that touch each other (8-neighbours) are grouped into one frontier.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

UNKNOWN = -1


@dataclass
class MapInfo:
    resolution: float  # metres per cell
    origin_x: float    # world x of cell (row 0, col 0) corner
    origin_y: float    # world y of cell (row 0, col 0) corner
    width: int         # number of columns
    height: int        # number of rows

    # NOTE: assumes the map origin has zero rotation (true for slam_toolbox).

    def cell_to_world(self, row, col):
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def world_to_cell(self, x, y):
        col = int(math.floor((x - self.origin_x) / self.resolution))
        row = int(math.floor((y - self.origin_y) / self.resolution))
        return row, col

    def in_bounds(self, row, col):
        return 0 <= row < self.height and 0 <= col < self.width


@dataclass
class Frontier:
    cells: list                 # [(row, col), ...]
    goal_cell: tuple            # (row, col) the robot is sent to
    goal_xy: tuple              # world (x, y) of goal_cell
    length_m: float             # frontier size in metres
    score: float = 0.0
    distance_m: float = field(default=0.0)


def _shift(mask, dr, dc):
    """Shift a boolean mask by (dr, dc), filling new cells with False."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    r0, r1 = max(dr, 0), h + min(dr, 0)
    c0, c1 = max(dc, 0), w + min(dc, 0)
    out[r0:r1, c0:c1] = mask[r0 - dr:r1 - dr, c0 - dc:c1 - dc]
    return out


def dilate(mask, radius_cells):
    """Grow True regions by radius_cells (square neighbourhood)."""
    if radius_cells <= 0:
        return mask.copy()
    out = mask.copy()
    for dr in range(-radius_cells, radius_cells + 1):
        for dc in range(-radius_cells, radius_cells + 1):
            out |= _shift(mask, dr, dc)
    return out


def frontier_mask(grid, free_max=25):
    """Boolean mask of free cells that have an unknown 4-neighbour."""
    free = (grid >= 0) & (grid <= free_max)
    unknown = grid == UNKNOWN
    touches_unknown = (
        _shift(unknown, 1, 0) | _shift(unknown, -1, 0)
        | _shift(unknown, 0, 1) | _shift(unknown, 0, -1)
    )
    return free & touches_unknown


def cluster_cells(mask):
    """Group True cells into 8-connected clusters. Returns list of [(r, c), ...]."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    clusters = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        cluster = []
        queue = deque([(r, c)])
        seen[r, c] = True
        while queue:
            cr, cc = queue.popleft()
            cluster.append((int(cr), int(cc)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        clusters.append(cluster)
    return clusters


def find_frontiers(grid, info, min_size_cells=5, free_max=25,
                   occupied_min=65, clearance_m=0.25):
    """Find frontiers and pick a safe goal cell for each.

    The goal is the frontier cell closest to the cluster's centre that is at
    least clearance_m away from any wall. Clusters with no safe cell are dropped.
    """
    mask = frontier_mask(grid, free_max)
    occupied = grid >= occupied_min
    near_wall = dilate(occupied, int(math.ceil(clearance_m / info.resolution)))

    frontiers = []
    for cells in cluster_cells(mask):
        if len(cells) < min_size_cells:
            continue
        safe = [rc for rc in cells if not near_wall[rc]]
        if not safe:
            continue
        mr = sum(r for r, _ in cells) / len(cells)
        mc = sum(c for _, c in cells) / len(cells)
        goal = min(safe, key=lambda rc: (rc[0] - mr) ** 2 + (rc[1] - mc) ** 2)
        frontiers.append(Frontier(
            cells=cells,
            goal_cell=goal,
            goal_xy=info.cell_to_world(*goal),
            length_m=len(cells) * info.resolution,
        ))
    return frontiers


def is_blacklisted(xy, blacklist, radius_m):
    return any(math.hypot(xy[0] - bx, xy[1] - by) < radius_m for bx, by in blacklist)


def score_frontiers(frontiers, robot_xy, w_size=1.0, w_dist=1.0, w_bias=0.0,
                    bias_xy=None, blacklist=(), blacklist_radius_m=0.5):
    """Score and sort frontiers, best first. Blacklisted ones are removed.

    score = w_size * length_m - w_dist * distance_m + w_bias * bias_term
    bias_term is 1 at the bias point and falls off with distance from it.
    """
    kept = []
    for f in frontiers:
        if is_blacklisted(f.goal_xy, blacklist, blacklist_radius_m):
            continue
        f.distance_m = math.hypot(f.goal_xy[0] - robot_xy[0], f.goal_xy[1] - robot_xy[1])
        bias_term = 0.0
        if bias_xy is not None:
            d_bias = math.hypot(f.goal_xy[0] - bias_xy[0], f.goal_xy[1] - bias_xy[1])
            bias_term = 1.0 / (1.0 + d_bias)
        f.score = w_size * f.length_m - w_dist * f.distance_m + w_bias * bias_term
        kept.append(f)
    kept.sort(key=lambda f: f.score, reverse=True)
    return kept


def still_frontier(grid, info, xy, radius_m=0.3, free_max=25):
    """True if there is still a frontier cell within radius_m of xy."""
    row, col = info.world_to_cell(*xy)
    rad = max(1, int(math.ceil(radius_m / info.resolution)))
    r0, r1 = max(0, row - rad), min(info.height, row + rad + 1)
    c0, c1 = max(0, col - rad), min(info.width, col + rad + 1)
    if r0 >= r1 or c0 >= c1:
        return False
    return bool(frontier_mask(grid, free_max)[r0:r1, c0:c1].any())
