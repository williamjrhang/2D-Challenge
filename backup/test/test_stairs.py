"""stairs_profile on synthetic depth clouds (base_link, floor at z = FZ)."""
import math

import numpy as np
import pytest

from lite3_explore_pose.stairs import stairs_profile

FZ = -0.356
RNG = np.random.default_rng(0)


def floor(x0=0.3, x1=2.5):
    x, y = np.meshgrid(np.arange(x0, x1, 0.02), np.arange(-0.6, 0.6, 0.02))
    return np.stack([x.ravel(), y.ravel(), np.full(x.size, FZ)], 1)


def stairs(x0=0.9, n=4, rise=0.17, tread=0.28, top_len=0.8):
    pts = [floor(0.3, x0)]
    ys = np.arange(-0.6, 0.6, 0.02)
    for k in range(1, n + 1):
        xs = x0 + (k - 1) * tread
        zs = np.arange((k - 1) * rise, k * rise, 0.02)
        y, z = np.meshgrid(ys, zs)                       # riser face
        pts.append(np.stack([np.full(y.size, xs), y.ravel(), FZ + z.ravel()], 1))
        length = tread if k < n else top_len
        x, y = np.meshgrid(np.arange(xs, xs + length, 0.02), ys)   # tread
        pts.append(np.stack([x.ravel(), y.ravel(), np.full(x.size, FZ + k * rise)], 1))
    return np.concatenate(pts)


def wall(x=1.0, height=0.54):
    pts = [floor(0.3, x)]
    y, z = np.meshgrid(np.arange(-0.6, 0.6, 0.02), np.arange(0, height, 0.02))
    pts.append(np.stack([np.full(y.size, x), y.ravel(), FZ + z.ravel()], 1))
    return np.concatenate(pts)


def rubble():
    p = floor()
    for _ in range(12):                                   # random low blocks / slats
        cx, cy = RNG.uniform(0.5, 2.0), RNG.uniform(-0.4, 0.4)
        h = RNG.uniform(0.03, 0.09)
        m = (np.abs(p[:, 0] - cx) < 0.08) & (np.abs(p[:, 1] - cy) < 0.15)
        p[m, 2] = FZ + h
    return p


def kerb(x0=1.0, h=0.15):
    p = floor()
    p[p[:, 0] >= x0, 2] = FZ + h
    return p


def realistic(p, drop=0.4, noise=0.008, yaw_deg=0.0):
    p = p[RNG.random(len(p)) > drop].copy()
    p += RNG.normal(0, noise, p.shape)
    if yaw_deg:
        a = math.radians(yaw_deg)
        c, s = math.cos(a), math.sin(a)
        x, y = p[:, 0].copy(), p[:, 1].copy()
        p[:, 0], p[:, 1] = c * x - s * y, s * x + c * y
    return p


def test_staircase_found_with_distance():
    r = stairs_profile(realistic(stairs(x0=0.9)), FZ)
    assert r['stairs'], r
    assert r['steps'] >= 2 and abs(r['first_step_m'] - 0.9) <= 0.06
    assert all(0.14 <= x <= 0.20 for x in r['rises'])
    assert r['yaw_deg'] is not None and abs(r['yaw_deg']) < 4


@pytest.mark.parametrize('rise,tread', [(0.10, 0.30), (0.20, 0.25), (0.15, 0.40)])
def test_generic_sizes_without_measuring(rise, tread):
    assert stairs_profile(realistic(stairs(rise=rise, tread=tread)), FZ)['stairs']


def test_many_missing_points_like_reflections():
    assert stairs_profile(realistic(stairs(), drop=0.6), FZ)['stairs']


def test_yaw_to_first_step_is_measured():
    r = stairs_profile(realistic(stairs(x0=1.0), yaw_deg=12.0), FZ)
    assert r['stairs'], r
    assert r['yaw_deg'] is not None and 8 <= abs(r['yaw_deg']) <= 16


def test_close_to_stairs_floor_out_of_view():
    r = stairs_profile(realistic(stairs(x0=0.2)), FZ)
    assert r['stairs'], r
    assert r['first_step_m'] <= 0.5 and r['floor_seen'] is False


@pytest.mark.parametrize('cloud,why', [
    (floor, 'no height change'),
    (rubble, ''),
    (wall, 'wall'),
    (kerb, '1 regular step'),
])
def test_not_stairs(cloud, why):
    r = stairs_profile(realistic(cloud()), FZ)
    assert not r['stairs'], r
    assert why in r['reason']


def test_no_points_or_floor():
    assert not stairs_profile(np.zeros((0, 3)), FZ)['stairs']
    assert not stairs_profile(realistic(stairs()), None)['stairs']


def test_result_is_json_serializable():
    """Detector publishes detections as JSON: numpy scalars broke whole frames (25/11)."""
    import json
    for cloud in (stairs(x0=0.9), stairs(x0=1.8), stairs(x0=0.2), wall(), floor()):
        json.dumps(stairs_profile(realistic(cloud), FZ))
