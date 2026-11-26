"""Frontier goals that lead INTO an arena cell. Pure numpy, no ROS.

Field problem (issue 1): the 360 deg lidar maps the space behind a doorway from metres
away, so the old rule "frontier gone -> cancel the goal" turned the robot away before it
ever walked in, and the forward camera never saw the room's side walls.

New rule: a frontier goal is done only when the robot has physically entered the cell the
frontier leads into (or the camera seen-mask already covers that cell). A frontier that
merely becomes mapped turns into a "go into that cell" goal instead of being cancelled.
"""
import math

import numpy as np

UNKNOWN = -1


def unknown_side_xy(grid, info, cells):
    """Mean world position of the UNKNOWN 4-neighbours of a frontier's cells, i.e. the
    side the frontier opens onto. None if it has none (already mapped)."""
    h, w = grid.shape
    pts = []
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == UNKNOWN:
                pts.append(info.cell_to_world(nr, nc))
    if not pts:
        return None
    a = np.asarray(pts, dtype=float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def target_cell(cell_of, goal_xy, robot_xy, unknown_xy=None, lead_m=0.5):
    """Arena cell the frontier leads into.

    cell_of(x, y) -> (i, j). Uses the unknown side of the frontier when known, else a
    point lead_m beyond the goal on the robot -> goal line. If that lands in the robot's
    own cell (frontier right next to it), the goal's cell is used."""
    if unknown_xy is not None:
        dx, dy = unknown_xy[0] - goal_xy[0], unknown_xy[1] - goal_xy[1]
    else:
        dx, dy = goal_xy[0] - robot_xy[0], goal_xy[1] - robot_xy[1]
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return cell_of(*goal_xy)
    px, py = goal_xy[0] + lead_m * dx / n, goal_xy[1] + lead_m * dy / n
    cell = cell_of(px, py)
    if cell == cell_of(*robot_xy):
        cell = cell_of(*goal_xy)
    return cell


def frontier_step(still_frontier, robot_cell, target, seen_frac, seen_done_frac=0.9,
                  converted=False):
    """What to do with a running frontier goal on a recheck.

    robot_cell: cell whose interior the robot is in now (None = on a border / unknown).
    seen_frac:  share of the target cell's free interior the camera has seen (None = n/a).
    Returns 'done' | 'convert' | 'continue':
      done      robot entered the target cell, or the camera already saw it
      convert   frontier got mapped before arrival: keep going, now INTO the target cell
      continue  nothing to change
    Never 'cancel': a mapped frontier is not a reason to turn away."""
    if target is not None and robot_cell == target:
        return 'done'
    if target is not None and seen_frac is not None and seen_frac >= seen_done_frac:
        return 'done'
    if not still_frontier and not converted and target is not None:
        return 'convert'
    return 'continue'


def cell_seen_fraction(seen, grid, info, centre_xy, half_m, free_max=25):
    """Share of the free map cells within +/- half_m (axis-aligned box in the map frame,
    good enough for a 'did the camera look in there' test) of centre_xy that are seen."""
    if seen is None or grid is None:
        return None
    r0, c0 = info.world_to_cell(centre_xy[0] - half_m, centre_xy[1] - half_m)
    r1, c1 = info.world_to_cell(centre_xy[0] + half_m, centre_xy[1] + half_m)
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(info.height - 1, r1), min(info.width - 1, c1)
    if r0 > r1 or c0 > c1:
        return None
    g = grid[r0:r1 + 1, c0:c1 + 1]
    free = (g >= 0) & (g <= free_max)
    n = int(free.sum())
    if n == 0:
        return None
    return float((seen[r0:r1 + 1, c0:c1 + 1] & free).sum()) / n
