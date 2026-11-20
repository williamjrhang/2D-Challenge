"""Coverage on the current SLAM map. Pure numpy, no ROS imports.

Two kinds of coverage live here:

1. Coverage (class): which 1.02 m ARENA CELLS the robot has walked through.
   (Not used by explorer_node at the moment.)

2. Camera coverage (functions below the class): which free map cells the forward
   CAMERA has looked at. SLAM maps a room from its doorway, but the QR codes are on
   the side walls, so "mapped" is not "seen". When no frontiers are left, the
   explorer's COVER state drives to places the camera has not looked at yet.

   History is stored as robot POSES (x, y, yaw, t) in the map frame, not as a grid:
   slam_toolbox grows the map and moves its origin, so we rasterize the poses onto
   whatever the current map is (see SeenTracker).
"""

import glob
import json
import math
import os
import time

import numpy as np

# MapInfo is shared with frontiers.py (same fields, plus world_to_cell / in_bounds).
from lite3_explore.frontiers import Frontier, MapInfo, cluster_cells

UNKNOWN = -1


def dilate(mask, radius_cells):
    """Grow True regions by radius_cells (square neighbourhood).

    A square dilation is separable, so rows then columns costs O(radius) array
    passes instead of O(radius^2).
    """
    radius_cells = max(0, int(radius_cells))
    out = mask.copy()
    for k in range(1, radius_cells + 1):
        out[k:, :] |= mask[:-k, :]
        out[:-k, :] |= mask[k:, :]
    rows = out.copy()
    for k in range(1, radius_cells + 1):
        out[:, k:] |= rows[:, :-k]
        out[:, :-k] |= rows[:, k:]
    return out


def within_bounds(xy, bounds):
    return bounds is None or (bounds[0] <= xy[0] <= bounds[2]
                              and bounds[1] <= xy[1] <= bounds[3])


class Coverage:
    """Arena cells the robot has WALKED through (used by SELECT / PLAN_COVERAGE)."""

    def __init__(self, arena, bounds):
        self.arena = arena
        self.bounds = bounds
        self.visited = set()

    def cell_at(self, xy):
        return self.arena.cell_index(*self.arena.map_to_arena(*xy))

    def cell_position(self, cell):
        i, j = cell
        x, y = self.arena.start
        return self.arena.arena_to_map(x + i * self.arena.cell, y + j * self.arena.cell)

    def observe(self, xy):
        if within_bounds(xy, self.bounds):
            self.visited.add(self.cell_at(xy))

    def targets(self, grid, info, robot_xy, free_max, clearance_m, failed=()):
        """One clear point near the middle of each unvisited cell, nearest first."""
        free = (grid >= 0) & (grid <= free_max)
        near_obstacle = dilate(~free, math.ceil(clearance_m / info.resolution))
        best = {}
        for row, col in zip(*np.nonzero(free & ~near_obstacle)):
            xy = info.cell_to_world(row, col)
            if not within_bounds(xy, self.bounds):
                continue
            cell = self.cell_at(xy)
            if cell in self.visited or cell in failed:
                continue
            # Prefer interior points so that Nav2's goal tolerance still puts the
            # robot inside the cell. An offset point can serve a partially open cell.
            centre_distance = math.dist(xy, self.cell_position(cell))
            if cell not in best or centre_distance < best[cell][0]:
                best[cell] = centre_distance, xy
        return sorted(((cell, xy) for cell, (_, xy) in best.items()),
                      key=lambda target: math.dist(robot_xy, target[1]))


# =====================================================================================
# Camera coverage
# =====================================================================================

class PoseHistory:
    """Robot poses (x, y, yaw, t) in the map frame.

    A pose is only stored if the robot moved more than min_move_m or turned more than
    min_turn_deg since the last stored pose, so standing still adds nothing.
    """

    def __init__(self, min_move_m=0.2, min_turn_deg=15.0):
        self.min_move_m = min_move_m
        self.min_turn_rad = math.radians(min_turn_deg)
        self.poses = []

    def add(self, x, y, yaw, t):
        """Store the pose if it is new enough. Returns True if it was stored."""
        if self.poses:
            lx, ly, lyaw, _ = self.poses[-1]
            turned = abs(math.atan2(math.sin(yaw - lyaw), math.cos(yaw - lyaw)))
            if math.hypot(x - lx, y - ly) <= self.min_move_m and turned <= self.min_turn_rad:
                return False
        self.poses.append((float(x), float(y), float(yaw), float(t)))
        return True

    def clear(self):
        self.poses = []


def bounds_mask(info, bounds):
    """Cells whose centre is inside bounds [xmin, ymin, xmax, ymax] (all cells if None)."""
    if bounds is None:
        return np.ones((info.height, info.width), dtype=bool)
    xs = info.origin_x + (np.arange(info.width) + 0.5) * info.resolution
    ys = info.origin_y + (np.arange(info.height) + 0.5) * info.resolution
    cols_ok = (xs >= bounds[0]) & (xs <= bounds[2])
    rows_ok = (ys >= bounds[1]) & (ys <= bounds[3])
    return rows_ok[:, None] & cols_ok[None, :]


def visited_mask(grid, info, poses, robot_radius_m, free_max=25):
    """Free cells within robot_radius_m of any pose (where the robot's body has been)."""
    mask = np.zeros((info.height, info.width), dtype=bool)
    rad = max(1, int(math.ceil(robot_radius_m / info.resolution)))
    for x, y, _, _ in poses:
        row, col = info.world_to_cell(x, y)
        r0, r1 = max(0, row - rad), min(info.height, row + rad + 1)
        c0, c1 = max(0, col - rad), min(info.width, col + rad + 1)
        if r0 >= r1 or c0 >= c1:
            continue  # pose is outside the current map
        rr, cc = np.mgrid[r0:r1, c0:c1]
        cx, cy = info.cell_to_world(rr, cc)
        mask[r0:r1, c0:c1] |= np.hypot(cx - x, cy - y) <= robot_radius_m
    return mask & (grid >= 0) & (grid <= free_max)


def mark_wedge(seen, grid, info, pose, fov_deg, range_m, occupied_min=65):
    """Mark the cells the camera sees from one pose (in place).

    Rays are cast across the field of view. Each ray stops at the first occupied cell
    (that wall cell IS marked seen: the camera sees the wall and its QR code) or at
    the first unknown cell (we can't claim to have seen past something not mapped yet).
    """
    x, y, yaw = pose[0], pose[1], pose[2]
    res = info.resolution
    fov = math.radians(fov_deg)
    # Enough rays that neighbouring rays are < 1 cell apart at full range.
    n_rays = max(2, int(math.ceil(fov * range_m / res)) + 1)
    n_steps = max(1, int(math.ceil(range_m / (0.5 * res))))   # half-cell steps
    angles = yaw + np.linspace(-fov / 2.0, fov / 2.0, n_rays)
    dists = np.arange(1, n_steps + 1) * (range_m / n_steps)

    # (n_rays, n_steps) world points -> cells
    px = x + np.cos(angles)[:, None] * dists[None, :]
    py = y + np.sin(angles)[:, None] * dists[None, :]
    cols = np.floor((px - info.origin_x) / res).astype(int)
    rows = np.floor((py - info.origin_y) / res).astype(int)
    inside = (rows >= 0) & (rows < info.height) & (cols >= 0) & (cols < info.width)
    rows_c = np.clip(rows, 0, info.height - 1)
    cols_c = np.clip(cols, 0, info.width - 1)
    values = grid[rows_c, cols_c]

    # A sample blocks the ray if it is a wall, unknown, or off the map.
    blocks = (values >= occupied_min) | (values == UNKNOWN) | ~inside
    # Index of the first blocking sample on each ray (n_steps if none).
    first_block = np.where(blocks.any(axis=1), blocks.argmax(axis=1), n_steps)
    step_idx = np.arange(n_steps)[None, :]
    visible = step_idx < first_block[:, None]
    # Include the blocking cell itself if it is a wall (seen), not if unknown/off-map.
    hit_wall = (step_idx == first_block[:, None]) & (values >= occupied_min) & inside
    visible |= hit_wall

    # The robot's own cell is always seen.
    r0, c0 = info.world_to_cell(x, y)
    if info.in_bounds(r0, c0):
        seen[r0, c0] = True
    seen[rows[visible], cols[visible]] = True


def seen_mask(grid, info, poses, fov_deg, range_m, occupied_min=65):
    """Cells the camera has seen from any of the poses (full recompute)."""
    seen = np.zeros((info.height, info.width), dtype=bool)
    for pose in poses:
        mark_wedge(seen, grid, info, pose, fov_deg, range_m, occupied_min)
    return seen


class SeenTracker:
    """Keeps the seen mask up to date without recomputing it every time.

    New poses are added incrementally. The whole mask is recomputed only when the map
    size, origin or resolution changes (slam_toolbox grew the map), or when the pose
    list was reset/reloaded.
    """

    def __init__(self, fov_deg, range_m, occupied_min=65):
        self.fov_deg = fov_deg
        self.range_m = range_m
        self.occupied_min = occupied_min
        self.reset()

    def reset(self):
        self.mask = None
        self.map_key = None
        self.n_done = 0      # poses already drawn into self.mask

    def update(self, grid, info, poses):
        key = (info.resolution, info.origin_x, info.origin_y, info.width, info.height)
        if key != self.map_key or len(poses) < self.n_done:
            self.mask = seen_mask(grid, info, poses, self.fov_deg, self.range_m,
                                  self.occupied_min)
            self.map_key = key
        else:
            for pose in poses[self.n_done:]:
                mark_wedge(self.mask, grid, info, pose, self.fov_deg, self.range_m,
                           self.occupied_min)
        self.n_done = len(poses)
        return self.mask


def unseen_free_mask(grid, info, seen, bounds, free_max=25):
    """Free cells inside bounds that the camera has not seen."""
    free = (grid >= 0) & (grid <= free_max)
    return free & ~seen & bounds_mask(info, bounds)


def coverage_fraction(grid, info, seen, bounds, free_max=25):
    """Share (0..1) of the free cells inside bounds that the camera has seen."""
    free = (grid >= 0) & (grid <= free_max) & bounds_mask(info, bounds)
    total = int(free.sum())
    if total == 0:
        return 0.0
    return float((free & seen).sum()) / total


def find_coverage_goals(grid, info, seen, bounds, clearance_m, min_cluster_cells,
                        free_max=25, occupied_min=65, max_goal_dist_m=2.5,
                        avoid=(), avoid_radius_m=0.0):
    """Group unseen free cells into clusters and pick a goal for each.

    The goal is picked like find_frontiers does: the safe cell (clearance_m from
    walls (>= occupied_min) AND unknown, inside bounds) closest to the cluster's centre. Unseen strips
    along walls often have no safe cell inside them, so we then take the nearest safe
    cell anywhere, as long as it is within max_goal_dist_m (camera range) of the
    cluster centre. Clusters with no such cell are dropped.

    avoid: world (x, y) points (the explorer's blacklist). Cells within avoid_radius_m
    of them are not used as goals, so a big cluster that is still partly unseen after
    the robot went to its centre gets a goal somewhere else instead of being dropped.

    Returns Frontier objects so the explorer can reuse score_frontiers (and its
    blacklist). length_m = cluster size in cells * resolution, as for frontiers.
    """
    unseen = unseen_free_mask(grid, info, seen, bounds, free_max)
    free = (grid >= 0) & (grid <= free_max)
    obstacle = (grid >= occupied_min) | (grid == UNKNOWN)
    near_obstacle = dilate(obstacle, int(math.ceil(clearance_m / info.resolution)))
    safe = free & ~near_obstacle & bounds_mask(info, bounds)
    if avoid_radius_m > 0:
        rad = int(math.ceil(avoid_radius_m / info.resolution))
        for ax, ay in avoid:
            row, col = info.world_to_cell(ax, ay)
            r0, r1 = max(0, row - rad), min(info.height, row + rad + 1)
            c0, c1 = max(0, col - rad), min(info.width, col + rad + 1)
            if r0 >= r1 or c0 >= c1:
                continue
            rr, cc = np.mgrid[r0:r1, c0:c1]
            cx, cy = info.cell_to_world(rr, cc)
            safe[r0:r1, c0:c1] &= np.hypot(cx - ax, cy - ay) >= avoid_radius_m
    safe_rows, safe_cols = np.nonzero(safe)

    goals = []
    for cells in cluster_cells(unseen):
        if len(cells) < min_cluster_cells:
            continue
        mr = sum(r for r, _ in cells) / len(cells)
        mc = sum(c for _, c in cells) / len(cells)
        inside = [rc for rc in cells if safe[rc]]
        if inside:
            goal = min(inside, key=lambda rc: (rc[0] - mr) ** 2 + (rc[1] - mc) ** 2)
        elif len(safe_rows):
            d2 = (safe_rows - mr) ** 2 + (safe_cols - mc) ** 2
            i = int(np.argmin(d2))
            if math.sqrt(d2[i]) * info.resolution > max_goal_dist_m:
                continue
            goal = (int(safe_rows[i]), int(safe_cols[i]))
        else:
            continue
        goals.append(Frontier(
            cells=cells,
            goal_cell=goal,
            goal_xy=info.cell_to_world(*goal),
            length_m=len(cells) * info.resolution,
        ))
    return goals


def cells_seen_fraction(seen, cells):
    """Share of the given (row, col) cells that are seen. Cells off the map count unseen."""
    if not cells:
        return 1.0
    h, w = seen.shape
    hits = sum(1 for r, c in cells if 0 <= r < h and 0 <= c < w and seen[r, c])
    return hits / len(cells)


# ---------------------------------------------------------------- save / load

def save_poses(path, poses, params):
    """Save poses + params to an .npz file atomically (temp file, then rename).

    A crash half-way through writing leaves the old file intact.
    """
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    arr = np.array(poses, dtype=np.float64).reshape(-1, 4)
    with open(tmp, 'wb') as f:   # file object: np.savez must not append '.npz'
        np.savez(f, poses=arr, params=np.array(json.dumps(params)))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_poses(path):
    """Load (poses, params) written by save_poses."""
    with np.load(os.path.expanduser(path), allow_pickle=False) as data:
        poses = [tuple(float(v) for v in row) for row in data['poses']]
        params = json.loads(str(data['params']))
    return poses, params


def save(path, poses, meta):
    """Same as save_poses (name used by the explorer)."""
    save_poses(path, poses, meta)


def load(path):
    """Same as load_poses: returns (poses, meta)."""
    return load_poses(path)


COVERAGE_DIR = '~/challenge_ws/coverage'


def default_coverage_path(directory=COVERAGE_DIR, stamp=None):
    """<directory>/coverage_<YYYYmmdd_HHMMSS>.npz (stamp defaults to now)."""
    stamp = stamp or time.strftime('%Y%m%d_%H%M%S')
    return os.path.join(os.path.expanduser(directory), f'coverage_{stamp}.npz')


def newest_coverage_file(directory=COVERAGE_DIR):
    """Most recently modified coverage_*.npz in directory, or None."""
    files = glob.glob(os.path.join(os.path.expanduser(directory), 'coverage_*.npz'))
    return max(files, key=os.path.getmtime) if files else None
