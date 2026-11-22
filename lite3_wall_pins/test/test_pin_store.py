import json
import math

import pytest

from lite3_wall_pins.pin_store import PinStore, ViewPose


def store(tmp_path, map_id='lab'):
    return PinStore(str(tmp_path / 'pins.json'), map_id, merge_radius_m=0.6,
                    picture_merge_radius_m=0.8)


def view(x, y, rng):
    return ViewPose(x=x, y=y, yaw=0.0, yaw_to_target=0.0, range_m=rng)


def test_repeated_observations_make_one_pin(tmp_path):
    s = store(tmp_path)
    p1, new1 = s.observe('qr', 'ROOM-A', (2.0, 1.0, 0.5))
    p2, new2 = s.observe('qr', 'ROOM-A', (2.1, 1.05, 0.52))
    p3, new3 = s.observe('qr', 'ROOM-A', (1.95, 0.98, 0.49))
    assert new1 and not new2 and not new3
    assert p1 is p2 is p3 and len(s.pins) == 1 and p1.observations == 3
    assert abs(p1.x - 2.0167) < 1e-3


def test_distinct_payloads_close_together_stay_separate(tmp_path):
    s = store(tmp_path)
    s.observe('qr', 'A', (1.0, 1.0, 0.5))
    _, new = s.observe('qr', 'B', (1.05, 1.0, 0.5))      # 5 cm apart
    assert new and len(s.pins) == 2
    assert {p.payload for p in s.pins} == {'A', 'B'}


def test_same_payload_far_away_is_separate_and_reported(tmp_path):
    s = store(tmp_path)
    s.observe('qr', 'A', (0.0, 0.0, 0.5))
    assert [p.id for p in s.same_payload_elsewhere('A', 5.0, 0.0)] == ['qr_1']
    _, new = s.observe('qr', 'A', (5.0, 0.0, 0.5))
    assert new and len(s.pins) == 2


def test_pictures_merge_by_radius_only(tmp_path):
    s = store(tmp_path)
    s.observe('picture', '', (3.0, 0.0, 1.0), experimental=True)
    _, new = s.observe('picture', '', (3.5, 0.3, 1.0), experimental=True)   # 0.58 m
    assert not new
    _, new = s.observe('picture', '', (5.0, 0.0, 1.0), experimental=True)
    assert new and len(s.pins) == 2
    # a QR next to a picture is never merged into it
    _, new = s.observe('qr', 'X', (3.0, 0.0, 1.0))
    assert new and len(s.pins) == 3


def test_closest_view_pose_is_kept(tmp_path):
    s = store(tmp_path)
    p, _ = s.observe('qr', 'A', (2, 0, 0.5), view(0.0, 0.0, 2.0))
    s.observe('qr', 'A', (2, 0, 0.5), view(1.2, 0.0, 0.8))
    s.observe('qr', 'A', (2, 0, 0.5), view(-1.0, 0.0, 3.0))
    assert p.view.x == 1.2 and p.view.range_m == 0.8
    s.observe('qr', 'A', (2, 0, 0.5), None)                 # missing pose never erases
    assert p.view is not None


def test_save_and_reload_then_dedup(tmp_path):
    s = store(tmp_path)
    s.observe('qr', 'A', (1.0, 2.0, 0.4), view(0.2, 2.0, 0.8))
    s.observe('qr', 'B', (1.0, 2.2, 0.4))
    s.save()
    data = json.loads((tmp_path / 'pins.json').read_text())
    a = [p for p in data['pins'] if p['payload'] == 'A'][0]
    assert a['wall'] == {'x': 1.0, 'y': 2.0, 'z': 0.4}
    assert a['view_pose']['x'] == 0.2 and a['label'] == 'QR: A'

    s2 = store(tmp_path)
    assert 'loaded 2' in s2.load()
    _, new = s2.observe('qr', 'A', (1.1, 2.0, 0.4))           # after "restart"
    assert not new and len(s2.pins) == 2
    _, new = s2.observe('qr', 'C', (4, 4, 0.4))
    assert new and s2.pins[-1].id == 'qr_3'


def test_other_map_is_refused(tmp_path):
    s = store(tmp_path, 'lab')
    s.observe('qr', 'A', (0, 0, 0))
    s.save()
    with pytest.raises(ValueError):
        store(tmp_path, 'arena').load()


def test_missing_file_is_empty(tmp_path):
    s = store(tmp_path)
    assert 'starting empty' in s.load() and s.pins == []


def test_yaw_to_target_math():
    # robot at origin, target at (0, 2): face +90 deg
    assert abs(math.atan2(2 - 0, 0 - 0) - math.pi / 2) < 1e-12
