"""Wall vs curtain: flat panels must NOT count as a curtain (live 18/11 the robot walked into
pale wood walls after 'curtain_ahead'). Synthetic depth in base_link (x fwd, y left, z up)."""
import numpy as np

from lite3_explore.terrain import surface_wrinkle_mm

FLOOR = -0.35
RNG = np.random.default_rng(0)
LIMIT = 2.5          # default curtain_min_wrinkle_mm


def surface(depth_fn, noise_m=0.0015, n=20000):
    y = RNG.uniform(-0.4, 0.4, n)
    z = RNG.uniform(FLOOR, FLOOR + 0.9, n)
    x = depth_fn(y, z) + RNG.normal(0, noise_m, n)
    return np.c_[x, y, z]


def test_flat_wall_is_not_curtain():
    assert surface_wrinkle_mm(surface(lambda y, z: 0.40 + 0 * y), FLOOR) < LIMIT


def test_tilted_wall_is_not_curtain():
    assert surface_wrinkle_mm(surface(lambda y, z: 0.45 + 0.4 * y + 0.05 * z), FLOOR) < LIMIT


def test_wall_corner_is_not_curtain():
    assert surface_wrinkle_mm(surface(lambda y, z: 0.40 + 0.8 * np.abs(y)), FLOOR) < LIMIT


def test_noisy_far_wall_is_not_curtain():
    assert surface_wrinkle_mm(surface(lambda y, z: 0.8 + 0 * y, noise_m=0.004), FLOOR) < LIMIT


def test_curtain_with_folds_is_curtain():
    folds = lambda y, z: 0.40 + 0.015 * np.sin(2 * np.pi * y / 0.12)      # 1.5 cm folds
    assert surface_wrinkle_mm(surface(folds), FLOOR) > LIMIT


def test_gentle_curtain_is_curtain():
    folds = lambda y, z: 0.40 + 0.010 * np.sin(2 * np.pi * y / 0.15)      # 1 cm, wide folds
    assert surface_wrinkle_mm(surface(folds), FLOOR) > LIMIT


def test_no_depth_returns_none():
    assert surface_wrinkle_mm(np.zeros((0, 3)), FLOOR) is None
