"""Run with:  python3 -m pytest test/   (no ROS needed)"""
import numpy as np

from lite3_explore.terrain import classify, estimate_floor

FLOOR = -0.32
rng = np.random.default_rng(0)


def patch(n=3000, x=(0.3, 1.6), y=(-0.4, 0.4), z=FLOOR, noise=0.005):
    p = np.column_stack([rng.uniform(*x, n), rng.uniform(*y, n), np.full(n, z)])
    p[:, 2] += rng.normal(0, noise, n)
    return p


def test_flat_floor_is_estimated_and_not_an_obstacle():
    pts = patch()
    assert abs(estimate_floor(pts) - FLOOR) < 0.01
    kind, _, _ = classify(pts, FLOOR)
    assert kind is None


def test_slats_ahead_are_rubble_with_distance():
    pts = patch()
    ahead = pts[:, 0] > 0.9                      # slats every 10 cm, first one at 1.0 m
    slat = ahead & ((pts[:, 0] * 10).astype(int) % 2 == 0)
    pts[slat, 2] = FLOOR + 0.08                  # 8 cm high slats
    kind, dist, _ = classify(pts, FLOOR)
    assert kind == 'rubble' and 0.95 < dist < 1.05
    assert abs(estimate_floor(pts) - FLOOR) < 0.01   # floor right at the feet is still flat
    far = pts[pts[:, 0] > 0.9]
    assert estimate_floor(far) is None               # only bumpy floor visible -> no recalibration


def test_holes_are_potholes():
    pts = patch()
    hole = (np.hypot(pts[:, 0] - 1.0, pts[:, 1]) < 0.15)
    pts[hole, 2] = FLOOR - 0.10
    kind, dist, _ = classify(pts, FLOOR)
    assert kind == 'potholes' and 0.8 < dist < 0.9


def test_wall_ahead_is_not_rubble():
    pts = patch()
    wall = pts[:, 0] > 1.0
    pts[wall, 2] = rng.uniform(FLOOR, FLOOR + 0.4, wall.sum())   # vertical wall face
    kind, _, _ = classify(pts, FLOOR)
    assert kind is None
