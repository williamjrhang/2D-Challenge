"""Arena cells, walls and rooms from the live SLAM map. Pure numpy, no ROS.

Booklet v6.1:
  - cells are 1.02 m apart, origin (0, 0) = centre of the start cell, +x = start forward,
    +y = start left (same Arena as the task reports, mission_config.py)
  - QR codes are only on the LEFT or RIGHT walls of rooms (never in corridors), 10-15 cm
  - no pre-mapping: everything here comes from the map built during the run

Wall test adapted from lite3_maze_mission (maze_grid.CellAnalyzer): a cell side is a wall
when at least wall_min_frac of the edge has an occupied map cell within +/- wall_band_m.
A ROOM is a free cell with exactly three walls; its open side is the doorway.
"""
import math
from dataclasses import dataclass, field

import numpy as np

# side name -> unit normal in the ARENA frame (outwards from the cell centre)
SIDES = {'+x': (1.0, 0.0), '-x': (-1.0, 0.0), '+y': (0.0, 1.0), '-y': (0.0, -1.0)}
OPPOSITE = {'+x': '-x', '-x': '+x', '+y': '-y', '-y': '+y'}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class CellInfo:
    cell: tuple
    centre: tuple                  # map frame
    state: str                     # free | occupied | unknown | partial
    free_frac: float
    walls: list = field(default_factory=list)
    goal: tuple = None             # map (x, y): free point in the cell's middle, clear of walls


@dataclass
class Room:
    cell: tuple
    open_side: str
    side_walls: list               # the two walls left / right of the doorway (QR walls)
    standpoint: tuple              # map (x, y) just outside the doorway
    views: list                    # [(name, yaw)]: into the room, then each side wall


class CellGrid:

    def __init__(self, arena, half_m=0.30, wall_band_m=0.15, wall_min_frac=0.5,
                 min_free_frac=0.55, max_occupied_frac=0.25, range_cells=6, step_m=0.05,
                 bounds_cells=None, clearance_m=0.30):
        self.arena = arena
        self.size = arena.cell
        self.half = half_m
        self.band = wall_band_m
        self.wall_min = wall_min_frac
        self.min_free = min_free_frac
        self.max_occ = max_occupied_frac
        self.range = int(range_cells)
        self.step = step_m
        # [i_min, j_min, i_max, j_max] arena cells taking part (None = range_cells box)
        self.bounds_cells = [int(v) for v in bounds_cells] if bounds_cells else None
        self.clearance = clearance_m
        g = np.arange(-math.floor(half_m / step_m), math.floor(half_m / step_m) + 1) * step_m
        u, v = np.meshgrid(g, g)
        uv = np.stack([u.ravel(), v.ravel()], 1)
        self._visit_uv = uv[np.argsort((uv ** 2).sum(1), kind='stable')]   # centre first
        along = np.arange(-0.35, 0.35 + 1e-9, step_m)
        across = np.arange(self.size / 2 - wall_band_m, self.size / 2 + wall_band_m + 1e-9, step_m)
        self._side_uv = {}
        for s, (nu, nv) in SIDES.items():
            tu, tv = -nv, nu
            a, c = np.meshgrid(along, across, indexing='ij')
            self._side_uv[s] = np.stack([a * tu + c * nu, a * tv + c * nv], -1)

    # ---------- frames ----------
    def vec_to_map(self, u, v):
        """Arena-frame direction -> map frame (rotation only)."""
        c, s = math.cos(self.arena.yaw), math.sin(self.arena.yaw)
        return c * u + s * v, -s * u + c * v

    def centre(self, cell):
        sx, sy = self.arena.start
        return self.arena.arena_to_map(sx + cell[0] * self.size, sy + cell[1] * self.size)

    def cell_of(self, x, y):
        return self.arena.cell_index(*self.arena.map_to_arena(x, y))

    def in_visit_region(self, cell, x, y):
        ax, ay = self.arena.map_to_arena(x, y)
        sx, sy = self.arena.start
        return (abs(ax - (sx + cell[0] * self.size)) <= self.half
                and abs(ay - (sy + cell[1] * self.size)) <= self.half)

    def side_yaw(self, side):
        mx, my = self.vec_to_map(*SIDES[side])
        return math.atan2(my, mx)

    def _to_map(self, cell, uv):
        cx, cy = self.centre(cell)
        mx, my = self.vec_to_map(1.0, 0.0)
        nx, ny = self.vec_to_map(0.0, 1.0)
        return cx + uv[..., 0] * mx + uv[..., 1] * nx, cy + uv[..., 0] * my + uv[..., 1] * ny

    # ---------- map lookups ----------
    @staticmethod
    def _cats(grid, info, xs, ys, free_max, occupied_min):
        """-1 unknown / outside, 0 free, 1 occupied for world points."""
        col = np.floor((xs - info.origin_x) / info.resolution).astype(int)
        row = np.floor((ys - info.origin_y) / info.resolution).astype(int)
        inside = (row >= 0) & (row < info.height) & (col >= 0) & (col < info.width)
        out = np.full(xs.shape, -1, dtype=np.int8)
        vals = grid[row[inside], col[inside]]
        cat = np.where(vals >= occupied_min, 1, np.where((vals >= 0) & (vals <= free_max), 0, -1))
        out[inside] = cat
        return out

    def in_bounds(self, cell):
        b = self.bounds_cells
        return b is None or (b[0] <= cell[0] <= b[2] and b[1] <= cell[1] <= b[3])

    def map_box(self, margin_m=0.3):
        """Map-frame [xmin, ymin, xmax, ymax] around the bounds cells (None if no bounds)."""
        b = self.bounds_cells
        if b is None:
            return None
        pts = [self.centre((i, j)) for i in (b[0], b[2]) for j in (b[1], b[3])]
        r = (self.size / 2.0 + margin_m) * math.sqrt(2.0)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r]

    def candidate_cells(self, info):
        """Cells (inside bounds_cells, else within range_cells) whose centre is on the map."""
        x0, y0 = info.origin_x, info.origin_y
        x1, y1 = x0 + info.width * info.resolution, y0 + info.height * info.resolution
        b = self.bounds_cells or [-self.range, -self.range, self.range, self.range]
        out = []
        for i in range(b[0], b[2] + 1):
            for j in range(b[1], b[3] + 1):
                cx, cy = self.centre((i, j))
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    out.append((i, j))
        return out

    def _crop(self, grid, info, cells):
        """Only the part of the map around the cells: after a SLAM jump /map can have
        >10 M cells, and dilating all of it every update cost a full CPU core (25 Nov)."""
        from types import SimpleNamespace
        m = self.size / 2.0 + self.band + self.clearance + 0.2
        xs = [self.centre(c)[0] for c in cells] or [0.0]
        ys = [self.centre(c)[1] for c in cells] or [0.0]
        res = info.resolution
        c0 = max(0, int(math.floor((min(xs) - m * 1.5 - info.origin_x) / res)))
        r0 = max(0, int(math.floor((min(ys) - m * 1.5 - info.origin_y) / res)))
        c1 = min(info.width, int(math.ceil((max(xs) + m * 1.5 - info.origin_x) / res)))
        r1 = min(info.height, int(math.ceil((max(ys) + m * 1.5 - info.origin_y) / res)))
        sub = grid[r0:max(r0, r1), c0:max(c0, c1)]
        return sub, SimpleNamespace(resolution=res, origin_x=info.origin_x + c0 * res,
                                    origin_y=info.origin_y + r0 * res,
                                    width=sub.shape[1], height=sub.shape[0])

    def analyze(self, grid, info, free_max=25, occupied_min=65):
        from lite3_explore.coverage import dilate
        cells = self.candidate_cells(info)
        grid, info = self._crop(grid, info, cells)
        occ_near = dilate(grid >= occupied_min, int(math.ceil(self.clearance / info.resolution)))
        out = {}
        for cell in cells:
            xs, ys = self._to_map(cell, self._visit_uv)
            k = self._cats(grid, info, xs, ys, free_max, occupied_min)
            n = float(k.size)
            free = float((k == 0).sum()) / n
            occ = float((k == 1).sum()) / n
            unk = 1.0 - free - occ
            if occ > self.max_occ:
                state = 'occupied'
            elif free >= self.min_free:
                state = 'free'
            elif unk >= 0.5:
                state = 'unknown'
            else:
                state = 'partial'
            walls, goal = [], None
            if state == 'free':
                col = np.floor((xs - info.origin_x) / info.resolution).astype(int)
                row = np.floor((ys - info.origin_y) / info.resolution).astype(int)
                ok = (k == 0)
                inside = (row >= 0) & (row < info.height) & (col >= 0) & (col < info.width)
                ok &= inside
                ok[inside] &= ~occ_near[row[inside], col[inside]]
                idx = np.flatnonzero(ok)            # samples are sorted centre first
                if len(idx):
                    goal = (float(xs[idx[0]]), float(ys[idx[0]]))
                for side, uv in self._side_uv.items():
                    sx, sy = self._to_map(cell, uv)
                    ks = self._cats(grid, info, sx, sy, free_max, occupied_min)
                    if (ks == 1).any(axis=1).mean() >= self.wall_min:
                        walls.append(side)
            out[cell] = CellInfo(cell, self.centre(cell), state, round(free, 3), walls, goal)
        return out

    # ---------- rooms ----------
    def room_of(self, info_cell, standoff_m=0.15):
        """Room (3 walls, 1 doorway) -> where to stand and where to look; else None.
        Standpoint: base_link standoff_m outside the doorway line, so the camera (0.25 m
        ahead of base_link) is ~0.6 m from each side wall's centre, viewed at ~40 deg."""
        if info_cell.state != 'free' or len(info_cell.walls) != 3:
            return None
        open_side = next(s for s in SIDES if s not in info_cell.walls)
        ou, ov = SIDES[open_side]
        side_walls = [s for s in info_cell.walls if s != OPPOSITE[open_side]]
        h = self.size / 2.0
        cx, cy = info_cell.centre
        mx, my = self.vec_to_map(ou * (h + standoff_m), ov * (h + standoff_m))
        sp = (cx + mx, cy + my)
        views = [('room', math.atan2(cy - sp[1], cx - sp[0]))]
        for s in side_walls:
            nu, nv = SIDES[s]
            wx, wy = self.vec_to_map(nu * h, nv * h)
            views.append((s, math.atan2(cy + wy - sp[1], cx + wx - sp[0])))
        return Room(info_cell.cell, open_side, side_walls, sp, views)

    def rooms(self, cells):
        return [r for r in (self.room_of(c) for c in cells.values()) if r is not None]
