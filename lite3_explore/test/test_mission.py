"""Run with:  python3 -m pytest test/   (no ROS needed)"""
import json
import math

import cv2
import numpy as np
import pytest

from lite3_explore.mission_config import Arena
from lite3_explore.vlm_watcher import encode_multipart, parse_result

CFG = {'arena': {'cell_size_m': 1.0, 'start_cell_centre': [0.5, 0.5],
                 'start_heading_deg': 90.0, 'final_cell_centre': [0.5, 5.5]}}


def test_start_pose_is_start_cell():
    a = Arena(CFG)
    assert a.map_to_arena(0.0, 0.0) == (0.5, 0.5)


def test_robot_forward_follows_start_heading():
    a = Arena(CFG)            # robot starts facing arena +y
    ax, ay = a.map_to_arena(2.0, 0.0)
    assert math.isclose(ax, 0.5, abs_tol=1e-9) and math.isclose(ay, 2.5)


def test_round_trip_and_final_cell():
    a = Arena(CFG)
    mx, my = a.arena_to_map(*a.final)
    assert math.isclose(mx, 5.0) and math.isclose(my, 0.0, abs_tol=1e-9)
    ax, ay = a.map_to_arena(mx, my)
    assert math.isclose(ax, 0.5) and math.isclose(ay, 5.5)


def test_snaps_to_cell_centre():
    a = Arena(CFG)
    assert a.map_to_cell_centre(1.2, -0.3) == (0.5, 1.5)   # arena (0.8, 1.7) -> cell centre


def test_parse_result_plain_and_fenced():
    assert parse_result('{"a": 1}') == {'a': 1}
    assert parse_result('```json\n{"a": true}\n```') == {'a': True}


def test_multipart_has_fields_and_image():
    body, ctype = encode_multipart({'group_id': 'g', 'prompt': 'p'}, 'image', 'f.jpg', b'\xff\xd8', 'image/jpeg')
    boundary = ctype.split('boundary=')[1]
    assert body.count(boundary.encode()) == 4
    assert b'name="group_id"\r\n\r\ng\r\n' in body and b'filename="f.jpg"' in body


@pytest.mark.skipif(not hasattr(cv2, 'QRCodeEncoder'), reason='OpenCV 4.5 has no QR encoder')
def test_opencv_reads_a_qr_code():
    enc = cv2.QRCodeEncoder.create()
    img = enc.encode('Survivor hidden in kitchen')
    img = cv2.resize(img, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    text, pts, _ = cv2.QRCodeDetector().detectAndDecode(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    assert text == 'Survivor hidden in kitchen'
