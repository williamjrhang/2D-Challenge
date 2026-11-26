"""Run with:  python3 -m pytest test/   (no ROS needed)"""
import numpy as np

from lite3_explore_pose.terrain import classify, estimate_floor, shadow_check

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


def test_rubble_with_wall_behind_is_still_rubble():
    # Maze: rubble cell from 0.6 m, then a wall at 1.3 m (live: the wall vetoed the rubble).
    pts = patch()
    slat = (pts[:, 0] > 0.6) & ((pts[:, 0] * 10).astype(int) % 2 == 0)
    pts[slat, 2] = FLOOR + 0.06
    wall = pts[:, 0] > 1.3
    pts[wall, 2] = rng.uniform(FLOOR, FLOOR + 0.6, wall.sum())
    kind, dist, stats = classify(pts, FLOOR)
    assert kind == 'rubble' and 0.55 < dist < 0.7, stats


def test_wall_right_ahead_is_not_rubble():
    pts = patch()
    wall = pts[:, 0] > 0.5
    pts[wall, 2] = rng.uniform(FLOOR, FLOOR + 0.6, wall.sum())
    kind, _, _ = classify(pts, FLOOR)
    assert kind is None


def test_flat_floor_then_wall_is_not_rubble():
    pts = patch()
    wall = pts[:, 0] > 1.2
    pts[wall, 2] = rng.uniform(FLOOR, FLOOR + 0.6, wall.sum())
    kind, _, _ = classify(pts, FLOOR)
    assert kind is None


def floor_image(level=150, h=480, w=640, noise=8):
    return np.clip(rng.normal(level, noise, (h, w)), 0, 255)


def test_evenly_lit_floor_has_no_shadow():
    found, stats = shadow_check(floor_image())
    assert not found and stats['dark'] == 0.0


def test_dim_but_even_floor_has_no_shadow():
    found, _ = shadow_check(floor_image(level=70))
    assert not found


def test_shadow_patches_in_bottom_third_are_potholes():
    img = floor_image()
    img[360:440, 220:330] = 50                   # two dark hole shadows right ahead
    img[400:470, 380:450] = 40
    found, stats = shadow_check(img)
    assert found, stats


def test_shadow_above_bottom_third_is_ignored():
    img = floor_image()
    img[100:300, 160:480] = 30                   # far away / wall: not the floor ahead
    found, _ = shadow_check(img)
    assert not found


def test_shadow_outside_path_is_ignored():
    img = floor_image()
    img[330:, :150] = 30                         # left edge, outside the centre columns
    found, _ = shadow_check(img)
    assert not found


def test_too_dark_floor_is_not_trusted():
    img = floor_image(level=25, noise=3)
    img[360:440, 220:330] = 5
    found, _ = shadow_check(img)
    assert not found


def test_raised_board_is_not_taken_as_floor():
    # Live 18/11: floor_z became -0.262 (a raised board) instead of -0.356 -> false potholes.
    from lite3_explore_pose.terrain import update_floor
    ref = -0.356
    assert update_floor(ref, -0.262, ref) == ref                 # 9 cm off -> ignored
    assert abs(update_floor(ref, -0.346, ref) - (-0.354)) < 1e-6  # 1 cm drift -> follows
    assert update_floor(ref, None, ref) == ref


def test_board_ahead_is_rubble_when_floor_is_right():
    ref = -0.356
    pts = patch(z=ref)
    board = pts[:, 0] > 0.3
    pts[board, 2] = ref + 0.094                   # raised flat board, 9.4 cm
    gaps = board & ((pts[:, 0] * 10).astype(int) % 3 == 0)
    pts[gaps, 2] = ref                            # gaps between slats down to the floor
    kind, _, stats = classify(pts, ref)
    assert kind == 'rubble', stats
    kind_wrong, _, _ = classify(pts, ref + 0.094)  # old behaviour: floor = board top
    assert kind_wrong == 'potholes'


def test_wall_ahead_ignores_rubble_and_sees_wall():
    from lite3_explore_pose.terrain import wall_ahead_m
    ref = -0.356
    pts = patch(z=ref)
    board = (pts[:, 0] > 0.5) & (pts[:, 0] < 1.1)
    pts[board, 2] = ref + 0.094                   # rubble board: not a wall
    assert wall_ahead_m(pts, ref) is None
    wall = pts[:, 0] > 1.1
    pts[wall, 2] = rng.uniform(ref, ref + 0.6, wall.sum())
    d = wall_ahead_m(pts, ref)
    assert d is not None and 1.1 <= d < 1.2


def test_roller_across_the_path_is_rubble():
    # 22/11: a white roller (~15 cm) across the corridor: only ~15% of the region is bumps,
    # below rubble_frac, but it is a 'bar' -> rubble (high-step crossing).
    pts = patch()
    roller = (pts[:, 0] > 0.85) & (pts[:, 0] < 1.0)
    pts[roller, 2] = FLOOR + 0.12
    kind, dist, stats = classify(pts, FLOOR)
    assert kind == 'rubble' and 0.84 < dist < 0.9, stats


def test_scattered_small_bumps_are_not_a_bar():
    pts = patch()
    spots = rng.random(len(pts)) < 0.05           # 5% random bumps (noise, small stones)
    pts[spots, 2] = FLOOR + 0.05
    kind, _, stats = classify(pts, FLOOR)
    assert kind is None, stats
