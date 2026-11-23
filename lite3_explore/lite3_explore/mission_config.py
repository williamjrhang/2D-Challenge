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

    def cell_index(self, ax, ay):
        return (round((ax - self.start[0]) / self.cell),
                round((ay - self.start[1]) / self.cell))

    def cell_centre(self, ax, ay):
        """Snap an arena point to the centre of its grid cell. Cell centres sit at
        start_cell_centre + k * cell_size (booklet: start = (0, 0), spacing 1.02 m)."""
        i, j = self.cell_index(ax, ay)
        return self.start[0] + i * self.cell, self.start[1] + j * self.cell

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


# ================= v6.1: missions from text, per-cell evidence =================
import re
from collections import defaultdict

# Words in mission / QR texts -> vocabulary entry. Longest phrase wins. Add more in
# mission.yaml under `synonyms:` (phrase: canonical).
SYNONYMS = {
    # scene types
    'clothes shop': 'clothes_shop', 'clothing shop': 'clothes_shop', 'clothing store': 'clothes_shop',
    'clothes store': 'clothes_shop', 'clothes': 'clothes_shop', 'clothing': 'clothes_shop',
    'boutique': 'clothes_shop', 'apparel': 'clothes_shop', 'fashion': 'clothes_shop',
    'fitness centre': 'fitness_centre', 'fitness center': 'fitness_centre', 'fitness': 'fitness_centre',
    'gym': 'fitness_centre', 'gymnasium': 'fitness_centre', 'exercise': 'fitness_centre',
    'workout': 'fitness_centre',
    'kitchen': 'kitchen', 'food preparation': 'kitchen', 'prepare food': 'kitchen',
    'preparing food': 'kitchen', 'cooking': 'kitchen', 'cook': 'kitchen',
    'market': 'market', 'grocery': 'market', 'groceries': 'market', 'marketplace': 'market',
    'stall': 'market', 'produce': 'market',
    'workshop': 'workshop', 'repair': 'workshop', 'garage': 'workshop',
    'clinic': 'clinic', 'hospital': 'clinic', 'doctor': 'clinic', 'infirmary': 'clinic',
    'sick bay': 'clinic', 'treatment room': 'clinic',
    'cleaning room': 'cleaning_room', 'cleaning': 'cleaning_room', 'janitor': 'cleaning_room',
    'cleaning supplies': 'cleaning_room', 'storeroom for cleaning': 'cleaning_room',
    'furniture shop': 'furniture_shop', 'furniture store': 'furniture_shop',
    'furniture': 'furniture_shop',
    # hazards
    'high voltage': 'high_voltage', 'electrical': 'high_voltage', 'electric': 'high_voltage',
    'electricity': 'high_voltage', 'voltage': 'high_voltage', 'electrocution': 'high_voltage',
    'radioactive': 'radioactive', 'radioactivity': 'radioactive', 'nuclear': 'radioactive',
    'ionising radiation': 'radioactive', 'ionizing radiation': 'radioactive',
    'chemical': 'chemical', 'chemicals': 'chemical', 'corrosive': 'chemical', 'toxic': 'chemical',
    'biological': 'biological', 'biohazard': 'biological', 'infectious': 'biological',
    'virus': 'biological', 'bio hazard': 'biological',
    'flammable': 'flammable', 'fire': 'flammable', 'flame': 'flammable', 'inflammable': 'flammable',
    # objects
    'medical kit': 'medical_kit', 'medical equipment': 'medical_kit',
    'medical supplies': 'medical_kit', 'first aid': 'medical_kit', 'first-aid': 'medical_kit',
    'medkit': 'medical_kit', 'med kit': 'medical_kit',
    'hand tools': 'hand_tools', 'hand tool': 'hand_tools', 'tools': 'hand_tools', 'tool': 'hand_tools',
    'food': 'food', 'water': 'water', 'drinking water': 'water',
    'utensils': 'utensils', 'utensil': 'utensils', 'cutlery': 'utensils',
}
SURVIVOR_WORDS = r'\b(survivors?|victims?|persons?|people|friends?|casualty|casualties)\b'
OTHER_WORDS = r'\b(other|another|second|friend|mentioned|named|indicated|from_qr)\b'
QR_WORDS = r'\b(qr|message|code|information|info|clue|cue|share|decode)\b'


def _norm(text):
    return ' ' + re.sub(r'[^a-z0-9]+', ' ', (text or '').lower().replace('_', ' ')) + ' '


def vocab_kind(cfg):
    kinds = {}
    for s in cfg.get('scene_types') or []:
        kinds[s] = 'scene'
    for s in cfg.get('hazard_labels') or []:
        kinds[s] = 'hazard'
    for s in cfg.get('object_labels') or []:
        kinds[s] = 'object'
    return kinds


def find_terms(text, cfg):
    """Vocabulary entries named in a free text, in order: [(canonical, kind), ...]."""
    kinds = vocab_kind(cfg)
    table = dict(SYNONYMS)
    table.update({k.replace('_', ' '): k for k in kinds})
    table.update({str(k).lower(): v for k, v in (cfg.get('synonyms') or {}).items()})
    t = _norm(text)
    found = []
    for phrase in sorted(table, key=len, reverse=True):
        canon = table[phrase]
        if canon not in kinds:
            continue
        pat = ' ' + _norm(phrase).strip() + ' '
        while pat in t:
            i = t.index(pat)
            found.append((i, canon, kinds[canon]))
            t = t[:i] + ' ' + '#' * (len(pat) - 2) + ' ' + t[i + len(pat):]   # consume
    return [(c, k) for _, c, k in sorted(found)]


def room_from_text(text, cfg):
    rooms = [c for c, k in find_terms(text, cfg) if k == 'scene']
    return rooms[0] if rooms else None


def parse_mission(text, cfg):
    """Mission text -> {'task_type', 'target'}; target = label (object / hazard), room
    (survivor), 'from_qr' (survivor in the room the QR names), None (any). Canonical form:
      'object medical_kit' | 'hazard high_voltage' | 'survivor kitchen' | 'qr' | 'survivor from_qr'
    Free text also works ('find the survivor in the room for food preparation'); always
    check with tools/check_missions.py before the run."""
    t = _norm(text)
    terms = find_terms(text, cfg)
    rooms = [c for c, k in terms if k == 'scene']
    hazards = [c for c, k in terms if k == 'hazard']
    objects = [c for c, k in terms if k == 'object']
    survivor = re.search(SURVIVOR_WORDS, t) or t.strip().startswith('survivor')
    if 'from qr' in t:
        return {'task_type': 'survivor', 'target': 'from_qr'}
    if re.search(r'^\s*qr\b', t) or (re.search(QR_WORDS, t) and not rooms and not hazards
                                     and not objects and not re.search(OTHER_WORDS, t)):
        return {'task_type': 'qr', 'target': None}
    if survivor and rooms:
        return {'task_type': 'survivor', 'target': rooms[0]}
    if hazards and (not objects or re.search(r'\b(hazard|sign|warning)\b', t)):
        return {'task_type': 'hazard', 'target': hazards[0]}
    if objects:
        return {'task_type': 'object', 'target': objects[0]}
    if survivor:
        return {'task_type': 'survivor', 'target': 'from_qr' if re.search(OTHER_WORDS, t) else None}
    if re.search(QR_WORDS, t):
        return {'task_type': 'qr', 'target': None}
    return {'task_type': None, 'target': None}


def load_missions(cfg, override_file=''):
    """{task_id: text} from mission.yaml `missions:`, replaced line by line by the
    override file if it exists ('M03: survivor kitchen' per line, # comments)."""
    missions = {str(k): str(v) for k, v in (cfg.get('missions') or {}).items()}
    path = os.path.expanduser(override_file or '')
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if ':' in line:
                    k, v = line.split(':', 1)
                    if k.strip() and v.strip():
                        missions[k.strip()] = v.strip()
    return missions


EVIDENCE_DEFAULTS = {
    'scene_confirm': 1.0,      # scene of a cell: best vote sum minus second best
    'item_confirm': 1.2,       # item in a cell: sum of best confidence per photo (~2 sightings)
    'scene_weak': 0.3,         # at the end of the run, send best guesses above these
    'item_weak': 0.4,
    'resubmit_margin': 0.5,    # resend a task only if the new answer is this much stronger
    'qr_survivor_bonus': 0.6,  # a QR read in the same cell supports a survivor there
    'item_scene_weight': 0.5,  # an item's photo also votes its scene into the item's cell
}


def range_weight(r):
    if r is None:
        return 0.5
    return 1.0 if r < 2.0 else 0.6 if r < 3.5 else 0.25


class Evidence:
    """Everything seen, per arena cell. Each item gets the room of ITS OWN cell, decided by
    votes from many photos (not the scene of the one photo it was seen in)."""

    def __init__(self, cfg, arena):
        self.cfg, self.arena = cfg, arena
        self.p = dict(EVIDENCE_DEFAULTS, **(cfg.get('evidence') or {}))
        self.scene_votes = defaultdict(lambda: defaultdict(float))   # cell -> scene -> sum
        self.items = defaultdict(lambda: defaultdict(dict))  # (type, label) -> cell -> photo -> w
        self.qrs = {}                                        # text -> {'n', 'cells', 'first'}
        self.order = 0

    def cell(self, map_xy):
        return self.arena.cell_index(*self.arena.map_to_arena(*map_xy))

    def cell_pos(self, cell):
        return [round(self.arena.start[0] + cell[0] * self.arena.cell, 2),
                round(self.arena.start[1] + cell[1] * self.arena.cell, 2)]

    def add_scene(self, cell, scene, weight):
        if scene in (self.cfg.get('scene_types') or []) and weight > 0:
            self.scene_votes[cell][scene] += weight

    def add_item(self, task_type, label, cell, weight, photo):
        d = self.items[(task_type, label)][cell]
        d[photo] = max(d.get(photo, 0.0), weight)

    def add_qr(self, text, cell=None):
        self.order += 1
        q = self.qrs.setdefault(text, {'n': 0, 'cells': defaultdict(int), 'first': self.order})
        q['n'] += 1
        if cell is not None:
            q['cells'][cell] += 1

    def scene_of(self, cell):
        votes = sorted(self.scene_votes.get(cell, {}).items(), key=lambda kv: -kv[1])
        if not votes:
            return '', 0.0
        second = votes[1][1] if len(votes) > 1 else 0.0
        return votes[0][0], round(votes[0][1] - second, 3)

    def qr_cells(self):
        return {c for q in self.qrs.values() for c in q['cells']}

    def candidates(self, task_type, label):
        """[(cell, item_strength, scene, scene_strength)] for one (type, label)."""
        out = []
        for cell, photos in self.items.get((task_type, label), {}).items():
            s = sum(photos.values())
            if task_type == 'survivor' and cell in self.qr_cells():
                s += self.p['qr_survivor_bonus']
            scene, ss = self.scene_of(cell)
            out.append((cell, round(s, 3), scene, ss))
        return out

    def best_qr(self):
        if not self.qrs:
            return None
        return max(self.qrs, key=lambda t: (self.qrs[t]['n'], -self.qrs[t]['first']))

    def decide(self, missions, final=False):
        """missions = {task_id: {'task_type', 'target'}} -> {task_id: answer} where answer =
        {'task_type', 'label', 'scene_type', 'pos', 'message', 'score', 'confirmed'}.
        Only answers that are confirmed (or, if final, above the weak limits) are returned."""
        p = self.p
        out, used_survivor_cells = {}, set()

        def ok(item_s, scene_s):
            if item_s >= p['item_confirm'] and scene_s >= p['scene_confirm']:
                return True, True
            return final and item_s >= p['item_weak'] and scene_s >= p['scene_weak'], False

        def answer(t, label, cell, scene, score, confirmed):
            return {'task_type': t, 'label': label if t in ('object', 'hazard') else '',
                    'scene_type': scene, 'pos': self.cell_pos(cell), 'message': '',
                    'score': round(score, 3), 'confirmed': confirmed, 'cell': list(cell)}

        qr_text = self.best_qr()
        qr_room = room_from_text(qr_text, self.cfg) if qr_text else None
        # explicit rooms first, then 'any', then 'from_qr' (gets a survivor not used above)
        order = sorted(missions.items(), key=lambda kv: (
            kv[1].get('task_type') == 'survivor' and kv[1].get('target') in (None, 'from_qr'),
            kv[1].get('target') == 'from_qr'))
        for tid, m in order:
            t, target = m.get('task_type'), m.get('target')
            if t == 'qr':
                if qr_text:
                    out[tid] = {'task_type': 'qr', 'label': '', 'scene_type': '', 'pos': None,
                                'message': qr_text, 'score': float(self.qrs[qr_text]['n']),
                                'confirmed': True, 'cell': None}
                continue
            if t in ('object', 'hazard'):
                cands = self.candidates(t, target)
                want_room = None
            elif t == 'survivor':
                cands = [c for c in self.candidates('survivor', 'survivor')
                         if c[0] not in used_survivor_cells]
                want_room = target
                if target == 'from_qr':
                    want_room = qr_room
                    if qr_text is None and not final:
                        continue              # wait for the QR before guessing this one
            else:
                continue
            best = None
            for cell, s, scene, ss in cands:
                if not scene or (want_room and scene != want_room):
                    continue
                good, confirmed = ok(s, ss)
                if not good:
                    continue
                score = min(s, ss) + (10 if confirmed else 0)
                if best is None or score > best[0]:
                    best = (score, cell, scene, confirmed, min(s, ss))
            if best:
                out[tid] = answer(t, target, best[1], best[2], best[4], best[3])
                if t == 'survivor':
                    used_survivor_cells.add(best[1])
        return out
