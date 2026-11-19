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
        fc = a.get('final_cell_centre')     # null = unknown: no drive to a final cell
        self.final = (float(fc[0]), float(fc[1])) if fc else None

    def map_to_arena(self, x, y):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.start[0] + c * x - s * y, self.start[1] + s * x + c * y

    def arena_to_map(self, ax, ay):
        dx, dy = ax - self.start[0], ay - self.start[1]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return c * dx + s * dy, -s * dx + c * dy

    def cell_centre(self, ax, ay):
        """Snap an arena point to the centre of its grid cell. Cell centres sit at
        start_cell_centre + k * cell_size (booklet: start = (0, 0), spacing 1.02 m)."""
        return (self.start[0] + round((ax - self.start[0]) / self.cell) * self.cell,
                self.start[1] + round((ay - self.start[1]) / self.cell) * self.cell)

    def map_to_cell_centre(self, x, y):
        cx, cy = self.cell_centre(*self.map_to_arena(x, y))
        return round(cx, 2), round(cy, 2)


# ---------------- Task reports (booklet section F) ----------------
# A submission is only VALID if every field uses a permitted value. Fields a task type does
# not use must be "" (strings) or null (pos). Evaluated fields per task_type:
#   object / hazard: label, scene_type, pos      survivor: scene_type, pos      qr: message
NEEDS_SCENE = ('object', 'hazard', 'survivor')


def make_report(cfg, item, pos):
    """Build the JSON body for /task_report from a /perception/found item.
    Returns (report_without_task_id, problem). problem is a reason string when the item
    cannot score (e.g. unknown room for a survivor) -> log it, do not use up a task_id."""
    t = item.get('task_type')
    scenes = set(cfg.get('scene_types') or [])
    labels = set(cfg.get('hazard_labels') or []) | set(cfg.get('object_labels') or [])
    scene = item.get('scene_type') or ''
    scene = scene if scene in scenes else ''
    label = item.get('label') or ''
    label = label if (t in ('object', 'hazard') and label in labels) else ''
    report = {
        'group_id': cfg['group_id'],
        'task_type': t,
        'scene_type': scene,
        'label': label,
        'pos': None if t == 'qr' else [round(float(pos[0]), 2), round(float(pos[1]), 2)],
        'message': (item.get('message') or '') if t == 'qr' else '',
    }
    problem = None
    if t in NEEDS_SCENE and not scene:
        problem = 'room type unknown'
    elif t in ('object', 'hazard') and not label:
        problem = f'label {item.get("label")!r} not in the allowed list'
    elif t == 'qr' and not report['message']:
        problem = 'empty QR message'
    return report, problem


def pick_task_id(task_ids, report, used):
    """Most specific key first: 'type:label@scene', 'type:label', 'type@scene', 'type'.
    A value is one id or a list of ids (next unused one is taken). Each id is sent ONCE
    (the server keeps only the latest submission, so a later wrong guess must not
    overwrite a good one). Returns None if nothing matches or all ids are used."""
    t, label, scene = report['task_type'], report['label'], report['scene_type']
    keys = [f'{t}:{label}@{scene}', f'{t}:{label}', f'{t}@{scene}', t]
    for key in keys:
        value = (task_ids or {}).get(key)
        if not value:
            continue
        for tid in (value if isinstance(value, list) else [value]):
            if tid not in used:
                return tid
    return None
