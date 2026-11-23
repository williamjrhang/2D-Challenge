"""Booklet v6.1 scene understanding + reporting. Run:  python3 -m pytest test/  (no ROS)."""
import math
import os

import cv2
import numpy as np
import pytest
import yaml

from lite3_explore.mission_config import (Arena, Evidence, find_terms, load_missions,
                                          parse_mission, room_from_text)
from lite3_explore.reporter import should_send
from lite3_explore.vlm_watcher import build_prompt, clean_answer
from lite3_explore.qr_reader import QrDecoder
from lite3_explore.locator import Locator

REAL = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'config', 'mission.yaml')))
ARENA = Arena(REAL)
C = REAL['arena']['cell_size_m']


def mission(text):
    return parse_mission(text, REAL)


# ---------------- mission text ----------------

def test_canonical_missions():
    assert mission('object medical_kit') == {'task_type': 'object', 'target': 'medical_kit'}
    assert mission('hazard high_voltage') == {'task_type': 'hazard', 'target': 'high_voltage'}
    assert mission('survivor kitchen') == {'task_type': 'survivor', 'target': 'kitchen'}
    assert mission('qr') == {'task_type': 'qr', 'target': None}
    assert mission('survivor from_qr') == {'task_type': 'survivor', 'target': 'from_qr'}
    assert mission('survivor furniture_shop')['target'] == 'furniture_shop'
    assert mission('survivor fitness_centre')['target'] == 'fitness_centre'


def test_free_text_missions():
    assert mission('Find where the medical equipment is') == \
        {'task_type': 'object', 'target': 'medical_kit'}
    assert mission('A damaged electrical installation: find the warning sign') == \
        {'task_type': 'hazard', 'target': 'high_voltage'}
    assert mission('Find the survivor in the room for food preparation') == \
        {'task_type': 'survivor', 'target': 'kitchen'}
    assert mission("Read the first survivor's QR message") == {'task_type': 'qr', 'target': None}
    assert mission('Find the other survivor')['target'] == 'from_qr'
    assert mission('Find the survivor in the gym')['target'] == 'fitness_centre'
    assert mission('nonsense words') == {'task_type': None, 'target': None}


def test_room_from_qr_text():
    assert room_from_text('My friend is in the market.', REAL) == 'market'
    assert room_from_text('Help! My brother is trapped in the furniture store', REAL) == \
        'furniture_shop'
    assert room_from_text('I hurt my leg', REAL) is None
    # 'food preparation' is one phrase (kitchen), not the object 'food'
    assert find_terms('room for food preparation', REAL) == [('kitchen', 'scene')]


def test_missions_file_overrides(tmp_path):
    f = tmp_path / 'm.txt'
    f.write_text('# today\nM03: survivor clinic\nM06: hazard flammable\n')
    m = load_missions(REAL, str(f))
    assert m['M03'] == 'survivor clinic' and m['M06'] == 'hazard flammable'
    assert m['M01'] == REAL['missions']['M01']
    assert load_missions(REAL, str(tmp_path / 'missing.txt'))['M03'] == 'survivor kitchen'


# ---------------- VLM answer ----------------

def test_prompt_has_v61_vocabulary():
    p = build_prompt(REAL)
    for s in REAL['scene_types'] + REAL['hazard_labels'] + REAL['object_labels']:
        assert s in p
    assert 'explosion' in p and 'magnet' in p and 'Do NOT count the actual living people' in p
    assert '<' not in p.replace('<=', '')          # every placeholder filled


def test_clean_answer_drops_off_vocabulary():
    a = {'scene_type': 'gym', 'scene_confidence': 0.9,
         'hazards': [{'label': 'explosion', 'box': [0.1, 0.1, 0.2, 0.2]},
                     {'label': 'flammable', 'box': [0.1, 0.1, 0.3, 0.3], 'confidence': 0.8}],
         'survivors': [{'box': [0.5, 0.2, 0.4, 0.6]}],            # x2 < x1 -> no box
         'objects': [{'label': 'hammer', 'box': [0, 0, 1, 1]},
                     {'label': 'water', 'box': [0.2, 0.5, 0.3, 1.4], 'confidence': 'x'}]}
    scene, conf, items = clean_answer(a, REAL)
    assert scene == 'unknown' and conf == 0.0
    assert items == [('hazard', 'flammable', [0.1, 0.1, 0.3, 0.3], 0.8),
                     ('survivor', 'survivor', None, 0.7),
                     ('object', 'water', [0.2, 0.5, 0.3, 1.0], 0.7)]


def test_clean_answer_old_format():
    a = {'scene_type': 'clinic', 'scene_confidence': 0.8, 'hazard_label': 'biological',
         'hazard_box': [0.1, 0.1, 0.2, 0.2], 'survivor_visible': False, 'objects': []}
    scene, conf, items = clean_answer(a, REAL)
    assert scene == 'clinic' and items[0][:2] == ('hazard', 'biological')


# ---------------- evidence per cell ----------------

def xy(i, j, dx=0.0, dy=0.0):
    return ARENA.arena_to_map(i * C + dx, j * C + dy)


def ev_with(scene_cells, items):
    ev = Evidence(REAL, ARENA)
    for cell, scene, n in scene_cells:
        for _ in range(n):
            ev.add_scene(cell, scene, 0.8)
    for photo, (t, label, cell, w) in enumerate(items):
        ev.add_item(t, label, cell, w, photo)
    return ev


MISSIONS = {k: mission(v) for k, v in REAL['missions'].items()}


def test_cell_from_map_point():
    ev = Evidence(REAL, ARENA)
    assert ev.cell(xy(2, 1, 0.3, -0.4)) == (2, 1)
    assert ev.cell_pos((2, 1)) == [2.04, 1.02]


def test_one_sighting_is_not_enough_but_two_are():
    ev = ev_with([((1, 0), 'clinic', 2)], [('object', 'medical_kit', (1, 0), 0.9)])
    assert 'M01' not in ev.decide(MISSIONS)
    ev.add_item('object', 'medical_kit', (1, 0), 0.8, 99)
    a = ev.decide(MISSIONS)['M01']
    assert a['confirmed'] and a['scene_type'] == 'clinic' and a['pos'] == [1.02, 0.0]
    assert a['label'] == 'medical_kit'


def test_item_takes_room_of_its_own_cell():
    # photo said "kitchen" (the room behind), but the kit's cell is voted clinic
    ev = ev_with([((1, 0), 'clinic', 3), ((2, 0), 'kitchen', 3)],
                 [('object', 'medical_kit', (1, 0), 0.9), ('object', 'medical_kit', (1, 0), 0.9)])
    assert ev.decide(MISSIONS)['M01']['scene_type'] == 'clinic'


def test_unclear_room_waits_then_best_guess_at_the_end():
    ev = ev_with([((1, 0), 'clinic', 1), ((1, 0), 'kitchen', 1)],
                 [('hazard', 'high_voltage', (1, 0), 0.9), ('hazard', 'high_voltage', (1, 0), 0.9)])
    assert 'M02' not in ev.decide(MISSIONS)                     # tie: room not clear
    ev.add_scene((1, 0), 'kitchen', 0.5)
    a = ev.decide(MISSIONS, final=True)['M02']
    assert not a['confirmed'] and a['scene_type'] == 'kitchen'


def test_survivors_m03_and_m05_from_qr():
    ev = ev_with([((0, 1), 'kitchen', 2), ((2, 2), 'market', 2), ((1, 2), 'clinic', 2)],
                 [('survivor', 'survivor', (0, 1), 0.9), ('survivor', 'survivor', (0, 1), 0.9),
                  ('survivor', 'survivor', (2, 2), 0.9), ('survivor', 'survivor', (2, 2), 0.9),
                  ('survivor', 'survivor', (1, 2), 0.9), ('survivor', 'survivor', (1, 2), 0.9)])
    d = ev.decide(MISSIONS)
    assert d['M03']['scene_type'] == 'kitchen' and d['M03']['pos'] == [0.0, 1.02]
    assert 'M05' not in d                                     # no QR read yet
    ev.add_qr('My friend is in the market.', (0, 1))
    d = ev.decide(MISSIONS)
    assert d['M04']['message'] == 'My friend is in the market.' and d['M04']['pos'] is None
    assert d['M05']['scene_type'] == 'market' and d['M05']['pos'] == [2.04, 2.04]


def test_qr_without_room_gives_the_other_survivor():
    ev = ev_with([((0, 1), 'kitchen', 2), ((1, 2), 'clinic', 2)],
                 [('survivor', 'survivor', (0, 1), 0.9), ('survivor', 'survivor', (0, 1), 0.9),
                  ('survivor', 'survivor', (1, 2), 0.9), ('survivor', 'survivor', (1, 2), 0.9)])
    ev.add_qr('Please help my friend', (0, 1))
    d = ev.decide(MISSIONS)
    assert d['M03']['pos'] == [0.0, 1.02] and d['M05']['pos'] == [1.02, 2.04]


def test_same_photo_counts_once():
    ev = ev_with([((1, 0), 'clinic', 2)], [])
    for _ in range(5):
        ev.add_item('object', 'medical_kit', (1, 0), 0.9, 'photo-7')
    assert 'M01' not in ev.decide(MISSIONS)


# ---------------- resubmission ----------------

def ans(scene, score, confirmed, pos=(1.02, 0.0)):
    return {'task_type': 'survivor', 'label': '', 'scene_type': scene, 'pos': list(pos),
            'message': '', 'score': score, 'confirmed': confirmed}


def test_resubmit_only_when_stronger():
    first = ans('kitchen', 0.5, False)
    assert should_send(first, None, 0.5)
    assert not should_send(ans('kitchen', 2.0, True), ans('kitchen', 0.5, False), 0.5)  # same
    assert should_send(ans('market', 1.2, True), first, 0.5)          # guess -> confirmed
    assert not should_send(ans('market', 3.0, False), ans('kitchen', 1.0, True), 0.5)
    assert not should_send(ans('market', 1.3, True), ans('kitchen', 1.0, True), 0.5)
    assert should_send(ans('market', 1.6, True), ans('kitchen', 1.0, True), 0.5)


# ---------------- QR ----------------

def qr_image(size, text='My friend is in the furniture shop.', blur=False):
    q = cv2.QRCodeEncoder.create().encode(text)
    q = cv2.resize(q, (size, size), interpolation=cv2.INTER_AREA)
    im = np.full((480, 640), 200, np.uint8)
    pad = cv2.copyMakeBorder(q, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
    im[150:150 + pad.shape[0], 40:40 + pad.shape[1]] = pad
    return cv2.GaussianBlur(im, (3, 3), 0) if blur else im


@pytest.mark.skipif(not hasattr(cv2, 'QRCodeEncoder'), reason='needs OpenCV QR encoder')
@pytest.mark.parametrize('size', [120, 70, 55])
def test_small_side_wall_qr_is_read(size):
    d = QrDecoder()
    text = ''
    for _ in range(2):                         # left/right third alternate
        text, pts = d.read(qr_image(size))
        if text:
            break
    assert text == 'My friend is in the furniture shop.'
    assert 30 < pts[:, 0].mean() < 60 + size and 140 < pts[:, 1].mean() < 175 + size


@pytest.mark.skipif(not hasattr(cv2, 'QRCodeEncoder'), reason='needs OpenCV QR encoder')
def test_plain_opencv_misses_small_code():         # why the crop+upscale is needed
    assert cv2.QRCodeDetector().detectAndDecode(qr_image(55))[0] == ''


# ---------------- locator geometry (no ROS) ----------------

class FakeLoc(Locator):
    def __init__(self):                                   # no subscriptions
        self.K = np.array([[460.0, 0, 320], [0, 460.0, 240], [0, 0, 1]])
        self.grid = None
        self.floor_z = -0.356


def camera_snap(x=0.0, y=0.0, z=0.0, yaw=0.0):
    # optical frame (z forward, x right, y down) -> map (x forward, y left, z up)
    R0 = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return {'cloud': None, 'R_mc': Rz @ R0, 't_mc': np.array([x, y, z]), 'pose': (x, y, yaw)}


def test_wall_item_is_pulled_into_the_viewers_cell():
    loc = FakeLoc()
    # wall along x = 1.53 (border between cells 1 and 2), 0.05 m map cells
    data = np.zeros((100, 100), np.int8)
    data[:, int((1.53 + 2.5) / 0.05)] = 100
    loc.grid = {'res': 0.05, 'ox': -2.5, 'oy': -2.5, 'w': 100, 'h': 100, 'data': data}
    w = loc.place([300, 200, 340, 280], 640, 480, camera_snap(0.2, 0.0, 0.0), on='wall')
    assert w['how'] == 'ray_wall' and 1.5 < w['xy'][0] < 1.6
    assert ARENA.cell_index(*w['cell_xy']) == (1, 0)            # viewer's side of the wall
    assert ARENA.cell_index(*w['xy']) in ((1, 0), (2, 0))


def test_floor_object_without_depth_uses_floor_plane():
    loc = FakeLoc()
    # bottom of the box 100 px below centre: ray down at atan(100/460) -> floor at ~1.64 m
    w = loc.place([300, 300, 340, 340], 640, 480, camera_snap(), on='floor', pull_m=0.1)
    assert w['how'] == 'ray_floor' and abs(w['xy'][0] - 0.356 * 460 / 100) < 0.02


def test_nothing_known_gives_none_not_a_guess():
    loc = FakeLoc()
    assert loc.place([300, 100, 340, 140], 640, 480, camera_snap(), on='wall') is None


def test_scene_box_places_the_room_vote():
    from lite3_explore.vlm_watcher import scene_box
    assert scene_box({'scene_box': [0.548, 0.092, 0.945, 0.373]}) == [0.548, 0.092, 0.945, 0.373]
    assert scene_box({'scene_box': None}) is None and scene_box({}) is None
    assert 'scene_box' in build_prompt(REAL)
