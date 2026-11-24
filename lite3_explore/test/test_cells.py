"""cells.py: cells, walls and rooms from a synthetic map. Run: python3 -m pytest test/"""
import math

import numpy as np

from lite3_explore.cells import CellGrid
from lite3_explore.frontiers import MapInfo
from lite3_explore.mission_config import Arena

RES = 0.05


def arena(heading=0.0):
    return Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0.0, 0.0],
                            'start_heading_deg': heading, 'final_cell_centre': None}})


def blank(x0=-1.0, y0=-2.0, w=6.0, h=4.0, fill=0):
    info = MapInfo(resolution=RES, origin_x=x0, origin_y=y0,
                   width=int(w / RES), height=int(h / RES))
    return np.full((info.height, info.width), fill, dtype=np.int16), info


def wall(grid, info, x0, y0, x1, y1):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) / (RES / 2)) + 1
    for t in np.linspace(0, 1, n):
        r, c = info.world_to_cell(x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        grid[r, c] = 100


def room_map():
    """Room at cell (2, 0): walls on +x, +y, -y, doorway on -x (towards the start)."""
    g, info = blank()
    wall(g, info, 2.55, -0.51, 2.55, 0.51)     # +x
    wall(g, info, 1.53, 0.51, 2.55, 0.51)      # +y
    wall(g, info, 1.53, -0.51, 2.55, -0.51)    # -y
    return g, info


def test_walls_and_room_found():
    g, info = room_map()
    cg = CellGrid(arena())
    cells = cg.analyze(g, info)
    assert cells[(2, 0)].state == 'free'
    assert sorted(cells[(2, 0)].walls) == ['+x', '+y', '-y']
    assert cells[(1, 0)].walls == ['+y', '-y'] or set(cells[(1, 0)].walls) <= {'+y', '-y'}
    rooms = cg.rooms(cells)
    assert [r.cell for r in rooms] == [(2, 0)]
    r = rooms[0]
    assert r.open_side == '-x' and sorted(r.side_walls) == ['+y', '-y']
    assert math.isclose(r.standpoint[0], 2.04 - 0.66, abs_tol=1e-6)
    assert math.isclose(r.standpoint[1], 0.0, abs_tol=1e-6)
    names = [v[0] for v in r.views]
    assert names[0] == 'room' and sorted(names[1:]) == ['+y', '-y']
    yaws = dict(r.views)
    assert math.isclose(yaws['room'], 0.0, abs_tol=1e-6)
    assert math.isclose(yaws['+y'], math.atan2(0.51, 0.66), abs_tol=1e-6)
    assert math.isclose(yaws['-y'], -math.atan2(0.51, 0.66), abs_tol=1e-6)


def test_corridor_is_not_a_room():
    g, info = blank()
    wall(g, info, 0.51, 0.51, 3.57, 0.51)
    wall(g, info, 0.51, -0.51, 3.57, -0.51)
    cg = CellGrid(arena())
    assert cg.rooms(cg.analyze(g, info)) == []


def test_unknown_cells_have_no_walls():
    g, info = blank(fill=-1)
    cells = CellGrid(arena()).analyze(g, info)
    assert all(c.state == 'unknown' and not c.walls for c in cells.values())


def test_rotated_start_heading():
    """Robot started facing arena +y: the same physical room is cell (0, 2) in the arena."""
    g, info = room_map()
    cg = CellGrid(arena(heading=90.0))
    rooms = cg.rooms(cg.analyze(g, info))
    assert [r.cell for r in rooms] == [(0, 2)]
    assert math.isclose(rooms[0].standpoint[0], 2.04 - 0.66, abs_tol=1e-6)


def test_visit_region():
    cg = CellGrid(arena())
    assert cg.in_visit_region((1, 0), 1.2, 0.1)
    assert not cg.in_visit_region((1, 0), 1.5, 0.0)
    assert cg.cell_of(1.2, -0.9) == (1, -1)


def test_bounds_and_safe_goal():
    g, info = room_map()
    cg = CellGrid(arena(), bounds_cells=[0, -2, 5, 2])
    cells = cg.analyze(g, info)
    assert all(cg.in_bounds(c) for c in cells)
    assert (6, 0) not in cells and (0, 3) not in cells
    gx, gy = cells[(1, 0)].goal
    assert math.isclose(gx, 1.02, abs_tol=0.06) and abs(gy) < 0.06     # centre is clear
    box = cg.map_box()
    assert box[0] < -0.5 and box[2] > 5.6 and box[1] < -2.5 and box[3] > 2.5
