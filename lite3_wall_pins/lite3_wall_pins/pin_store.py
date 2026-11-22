"""Pins on the map + JSON persistence. Pure Python, no ROS.

Matching rule (checked BEFORE a pin is created):
  * QR:      same payload AND within merge_radius_m  -> same pin.
             Different payloads are NEVER merged, however close.
             Same payload farther away than the radius -> a separate pin (logged).
  * picture: any picture pin within picture_merge_radius_m -> same pin.
A matched observation only refines the existing pin (running mean, capped weight) and
bumps its counters; it never creates a second marker.

The wall position and the robot's viewing pose are stored separately. The viewing pose
kept is the one from the *closest* observation (best place to stand and read the target).
"""

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

FORMAT_VERSION = 1


@dataclass
class ViewPose:
    x: float
    y: float
    yaw: float                   # robot heading (rad, map frame) when it saw the target
    yaw_to_target: float         # heading that points the robot's x-axis at the target
    range_m: float               # camera -> target distance at that time


@dataclass
class Pin:
    id: str
    kind: str                    # 'qr' | 'picture'
    payload: str                 # QR text; '' for pictures
    x: float                     # wall position, map frame
    y: float
    z: float
    observations: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    view: Optional[ViewPose] = None
    experimental: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def label(self):
        if self.kind == 'qr':
            return f'QR: {self.payload}'
        return 'picture (experimental)' if self.experimental else 'picture'

    def to_json(self):
        d = asdict(self)
        d['label'] = self.label
        d['wall'] = {'x': d.pop('x'), 'y': d.pop('y'), 'z': d.pop('z')}
        d['view_pose'] = d.pop('view')
        return d

    @staticmethod
    def from_json(d):
        w = d['wall']
        v = d.get('view_pose')
        return Pin(id=d['id'], kind=d['kind'], payload=d.get('payload', ''),
                   x=float(w['x']), y=float(w['y']), z=float(w.get('z', 0.0)),
                   observations=int(d.get('observations', 1)),
                   first_seen=float(d.get('first_seen', 0.0)),
                   last_seen=float(d.get('last_seen', 0.0)),
                   view=ViewPose(**v) if v else None,
                   experimental=bool(d.get('experimental', False)),
                   extra=dict(d.get('extra', {})))


class PinStore:

    def __init__(self, path, map_id, frame_id='map', merge_radius_m=0.6,
                 picture_merge_radius_m=0.8, max_avg_weight=20):
        self.path = os.path.expanduser(path) if path else ''
        self.map_id = map_id
        self.frame_id = frame_id
        self.merge_radius_m = float(merge_radius_m)
        self.picture_merge_radius_m = float(picture_merge_radius_m)
        self.max_avg_weight = int(max_avg_weight)
        self.pins = []
        self.dirty = False

    # ------------------------------------------------------------ persistence

    def load(self):
        """Load pins for this map. Returns a human-readable status line."""
        if not self.path or not os.path.isfile(self.path):
            return f'no saved pins at {self.path or "(no file)"} - starting empty'
        with open(self.path) as f:
            data = json.load(f)
        if data.get('map_id') != self.map_id:
            raise ValueError(f'{self.path} belongs to map_id "{data.get("map_id")}", '
                             f'not "{self.map_id}"; refusing to mix maps')
        if data.get('frame_id', 'map') != self.frame_id:
            raise ValueError(f'{self.path} is in frame "{data.get("frame_id")}", '
                             f'not "{self.frame_id}"')
        self.pins = [Pin.from_json(p) for p in data.get('pins', [])]
        self.dirty = False
        return f'loaded {len(self.pins)} pins for map "{self.map_id}" from {self.path}'

    def save(self):
        if not self.path:
            return
        d = os.path.dirname(self.path)
        os.makedirs(d or '.', exist_ok=True)
        data = {'format_version': FORMAT_VERSION, 'map_id': self.map_id,
                'frame_id': self.frame_id, 'saved_at': time.time(),
                'pins': [p.to_json() for p in self.pins]}
        fd, tmp = tempfile.mkstemp(prefix='.pins_', suffix='.json', dir=d or '.')
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)                 # atomic: never a half-written file
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.dirty = False

    # ------------------------------------------------------------ matching

    def find_match(self, kind, payload, x, y):
        """Nearest existing pin this observation belongs to, or None."""
        radius = self.merge_radius_m if kind == 'qr' else self.picture_merge_radius_m
        best, best_d = None, radius
        for p in self.pins:
            if p.kind != kind or (kind == 'qr' and p.payload != payload):
                continue
            d = math.hypot(p.x - x, p.y - y)
            if d <= best_d:
                best, best_d = p, d
        return best

    def same_payload_elsewhere(self, payload, x, y):
        """QR pins with this payload that are OUTSIDE the merge radius (for logging)."""
        return [p for p in self.pins if p.kind == 'qr' and p.payload == payload
                and math.hypot(p.x - x, p.y - y) > self.merge_radius_m]

    def observe(self, kind, payload, xyz, view=None, experimental=False, extra=None,
                stamp=None):
        """Record one located observation. Returns (pin, created: bool)."""
        x, y, z = (float(v) for v in xyz)
        now = time.time() if stamp is None else stamp
        pin = self.find_match(kind, payload, x, y)
        if pin is None:
            n = sum(1 for p in self.pins if p.kind == kind) + 1
            pin = Pin(id=f'{kind}_{n}', kind=kind, payload=payload if kind == 'qr' else '',
                      x=x, y=y, z=z, first_seen=now, last_seen=now, view=view,
                      experimental=experimental, extra=dict(extra or {}))
            while any(p.id == pin.id for p in self.pins):     # ids stay unique on reload
                n += 1
                pin.id = f'{kind}_{n}'
            self.pins.append(pin)
            self.dirty = True
            return pin, True
        w = min(pin.observations, self.max_avg_weight)
        pin.x = (pin.x * w + x) / (w + 1)
        pin.y = (pin.y * w + y) / (w + 1)
        pin.z = (pin.z * w + z) / (w + 1)
        pin.observations += 1
        pin.last_seen = now
        if view is not None and (pin.view is None or view.range_m < pin.view.range_m):
            pin.view = view
        if extra:
            pin.extra.update(extra)
        self.dirty = True
        return pin, False


def default_pins_path(map_id):
    safe = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in map_id) or 'default'
    return os.path.join('~', '.ros', 'lite3_wall_pins', f'{safe}.json')
