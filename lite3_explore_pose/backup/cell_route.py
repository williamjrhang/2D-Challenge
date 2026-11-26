"""Routes between arena cells through cell centres and doorway midpoints (issue 10), and a
check of how well the arena grid matches the walls on the map. Pure numpy, no ROS.

Field problem: Nav2 cut corners through 1.02 m doorways, the robot brushed the walls and
goals failed. Two cells are connected when they share a side and neither has a wall on it
(CellGrid.analyze); a breadth-first search over that graph gives the cells to pass, and
the robot is routed centre -> doorway midpoint -> centre -> ... -> goal (Nav2
navigateThroughPoses), so it goes straight through each doorway.
"""
import math
from collections import deque

import numpy as np

STEPS = {'+x': (1, 0), '-x': (-1, 0), '+y': (0, 1), '-y': (0, -1)}
OPP = {'+x': '-x', '-x': '+x', '+y': '-y', '-y': '+y'}


def neighbours(cell, walls):
    """Cells reachable from cell through an open side. walls: cell -> set of wall sides."""
    if cell not in walls:
        return []
    out = []
    for side, (di, dj) in STEPS.items():
        nb = (cell[0] + di, cell[1] + dj)
        if nb in walls and side not in walls[cell] and OPP[side] not in walls[nb]:
            out.append(nb)
    return out


def bfs(start, goal, walls):
    """Shortest list of cells start..goal (both included), or None."""
    if start not in walls or goal not in walls:
        return None
    prev = {start: None}
    q = deque([start])
    while q:
        c = q.popleft()
        if c == goal:
            path = []
            while c is not None:
                path.append(c)
                c = prev[c]
            return path[::-1]
        for nb in neighbours(c, walls):
            if nb not in prev:
                prev[nb] = c
                q.append(nb)
    return None


def route_poses(path, centre, goal_xy, goal_yaw=None):
    """[(x, y, yaw)]: doorway midpoint, next cell centre, ..., then the goal itself.
    The start cell's centre is NOT included (centring is a separate step, see
    needs_centring). yaw = travel direction (the goal keeps goal_yaw if given)."""
    pts = []
    for a, b in zip(path, path[1:]):
        ca, cb = centre(a), centre(b)
        yaw = math.atan2(cb[1] - ca[1], cb[0] - ca[0])
        pts.append(((ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0, yaw))
        if b != path[-1]:
            pts.append((cb[0], cb[1], yaw))
    last_yaw = pts[-1][2] if pts else 0.0
    if goal_yaw is None:
        p = pts[-1] if pts else None
        goal_yaw = (math.atan2(goal_xy[1] - p[1], goal_xy[0] - p[0])
                    if p is not None and math.dist(p[:2], goal_xy) > 0.05 else last_yaw)
    pts.append((goal_xy[0], goal_xy[1], goal_yaw))
    return pts


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def needs_centring(pose, cell_centre, first_dir_yaw, tol_m=0.15, tol_deg=15.0):
    """Robot more than tol_m from its cell centre, or heading more than tol_deg off the
    direction of the first doorway: go to the centre and turn first."""
    off = math.dist(pose[:2], cell_centre) > tol_m
    turn = abs(wrap(first_dir_yaw - pose[2])) > math.radians(tol_deg)
    return off or turn


# ================= grid alignment check (report only) =================

def grid_alignment(grid, info, arena, occupied_min=65, max_deg=8.0, step_deg=0.25,
                   min_points=200):
    """Estimate how the walls on the map sit against the arena grid from mission.yaml
    (start_cell_centre, start_heading_deg). Walls should lie on cell BORDERS:
    start + (k + 0.5) * cell along both arena axes.

    Rotation: the angle that makes wall points line up best on both axes (circular
    concentration of x and y modulo the cell size). Offset: where those lines are vs the
    expected borders. Returns dict(rot_deg, dx_m, dy_m, strength, n) or None:
      rot_deg  walls are turned this much (counter-clockwise +) against the grid
      dx_m/dy_m  walls sit this far from the expected cell borders (arena x / y)
      strength 0..1, how clearly the walls form a grid (low = too little map to judge)"""
    rows, cols = np.nonzero(grid >= occupied_min)
    if len(rows) < min_points:
        return None
    xs = info.origin_x + (cols + 0.5) * info.resolution
    ys = info.origin_y + (rows + 0.5) * info.resolution
    c = arena.cell
    sx, sy = arena.start
    # map -> arena frame (same as Arena.map_to_arena)
    cy, sy_ = math.cos(arena.yaw), math.sin(arena.yaw)
    ax = sx + cy * xs - sy_ * ys
    ay = sy + sy_ * xs + cy * ys
    best = None
    for d in np.arange(-max_deg, max_deg + 1e-9, step_deg):
        t = math.radians(d)
        # rotate about the start cell centre
        rx = sx + math.cos(t) * (ax - sx) - math.sin(t) * (ay - sy)
        ry = sy + math.sin(t) * (ax - sx) + math.cos(t) * (ay - sy)
        zx = np.exp(2j * np.pi * (rx - sx) / c).mean()
        zy = np.exp(2j * np.pi * (ry - sy) / c).mean()
        score = abs(zx) + abs(zy)
        if best is None or score > best[0]:
            best = (score, d, zx, zy)
    score, d, zx, zy = best

    def offset(z):
        # expected border phase: (k + 0.5) * c -> angle pi
        ph = wrap(np.angle(z) - math.pi)
        return ph / (2 * math.pi) * c
    # the grid is rotated by -d relative to the walls
    return {'rot_deg': round(-float(d), 2), 'dx_m': round(float(offset(zx)), 3),
            'dy_m': round(float(offset(zy)), 3), 'strength': round(float(score) / 2, 3),
            'n': int(len(xs))}
