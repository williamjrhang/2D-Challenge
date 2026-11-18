"""Shared helpers: load config/mission.yaml and convert map <-> arena coordinates.

No ROS imports here, so it can be unit tested on its own.
"""

import math
import os

import yaml


def default_mission_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('lite3_explore'), 'config', 'mission.yaml')
    except Exception:
        return os.path.join(os.path.dirname(__file__), '..', 'config', 'mission.yaml')


def load_mission(path=''):
    with open(path or default_mission_file(), encoding='utf-8') as f:
        return yaml.safe_load(f)


class Arena:
    """map frame (origin = robot start, x = robot forward) <-> arena frame (organisers' grid)."""

    def __init__(self, cfg):
        a = cfg['arena']
        self.cell = float(a['cell_size_m'])
        self.start = (float(a['start_cell_centre'][0]), float(a['start_cell_centre'][1]))
        self.yaw = math.radians(float(a['start_heading_deg']))
        self.final = (float(a['final_cell_centre'][0]), float(a['final_cell_centre'][1]))

    def map_to_arena(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.start[0] + c * x - s * y, self.start[1] + s * x + c * y

    def arena_to_map(self, ax, ay):
        dx, dy = ax - self.start[0], ay - self.start[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def cell_centre(self, ax, ay):
        """Snap an arena point to the centre of its grid cell."""
        return ((math.floor(ax / self.cell) + 0.5) * self.cell,
                (math.floor(ay / self.cell) + 0.5) * self.cell)

    def map_to_cell_centre(self, x, y):
        cx, cy = self.cell_centre(*self.map_to_arena(x, y))
        return round(cx, 2), round(cy, 2)
