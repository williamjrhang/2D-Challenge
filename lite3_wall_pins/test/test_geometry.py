import math

import numpy as np

from lite3_wall_pins import geometry as geo

# real Lite3 colour intrinsics (camera_info, 18/11)
K = np.array([[459.432, 0, 321.399], [0, 459.380, 239.076], [0, 0, 1.0]])


def wall_points(depth, half=1.5, step=0.01, noise=0.0, seed=0):
    """A wall facing the camera (optical frame: z forward) at `depth` m."""
    g = np.arange(-half, half, step)
    x, y = np.meshgrid(g, g)
    z = np.full_like(x, depth)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    if noise:
        pts[:, 2] += np.random.default_rng(seed).normal(0, noise, len(pts))
    return pts


def qr_polygon(center_xyz, side):
    cx, cy, cz = center_xyz
    h = side / 2
    corners = np.array([[cx - h, cy - h, cz], [cx + h, cy - h, cz],
                        [cx + h, cy + h, cz], [cx - h, cy + h, cz]])
    u, v = geo.project(corners, K)
    return np.stack([u, v], axis=1)


def test_quat_and_yaw():
    q = (0, 0, math.sin(0.4), math.cos(0.4))       # yaw 0.8 rad
    R = geo.quat_to_matrix(*q)
    assert abs(math.atan2(R[1, 0], R[0, 0]) - 0.8) < 1e-9
    assert abs(geo.yaw_from_quat(*q) - 0.8) < 1e-9


def test_point_in_polygon_both_windings():
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    u = np.array([5, 11, 0.5, -1.0])
    v = np.array([5, 5, 9.5, 5.0])
    assert list(geo.points_in_convex_polygon(u, v, sq)) == [True, False, True, False]
    assert list(geo.points_in_convex_polygon(u, v, sq[::-1])) == [True, False, True, False]


def test_qr_on_wall_located_on_the_wall():
    pts = wall_points(2.0, noise=0.01)
    poly = qr_polygon((0.3, -0.1, 2.0), 0.12)
    region = geo.shrink_polygon(poly, 0.8)
    inside = geo.region_points(pts, K, region, 0.2, 4.0)
    p, info = geo.robust_surface_point(inside, 10, 0.6, 0.06, 0.03)
    assert p is not None, info
    assert abs(p[2] - 2.0) < 0.02
    assert abs(p[0] - 0.3) < 0.02 and abs(p[1] + 0.1) < 0.02


def test_background_does_not_win_over_dominant_surface():
    # QR on a pillar at 1.5 m, wall behind at 3 m visible around it (region slightly too big)
    wall = wall_points(3.0)
    wall = wall[(np.abs(wall[:, 0]) > 0.16) | (np.abs(wall[:, 1]) > 0.16)]   # occluded
    pillar = wall_points(1.5, half=0.08, step=0.005)
    pts = np.vstack([wall, pillar])
    poly = qr_polygon((0, 0, 1.5), 0.15)
    inside = geo.region_points(pts, K, geo.shrink_polygon(poly, 0.8), 0.2, 4.0)
    p, info = geo.robust_surface_point(inside, 10, 0.6, 0.06, 0.03)
    assert p is not None, info
    assert abs(p[2] - 1.5) < 0.02


def test_no_depth_gives_reason_not_a_guess():
    poly = qr_polygon((0, 0, 2.0), 0.12)
    inside = geo.region_points(np.zeros((0, 3)), K, poly, 0.2, 4.0)
    p, reason = geo.robust_surface_point(inside, 10, 0.6, 0.06, 0.03)
    assert p is None and 'depth points' in reason


def test_split_depth_is_rejected():
    # half near, half far: no single surface -> refuse instead of averaging into thin air
    near = wall_points(1.0, half=0.2, step=0.004)
    near = near[near[:, 0] < 0]
    far = wall_points(3.0, half=0.6, step=0.012)
    far = far[far[:, 0] >= 0]
    poly = qr_polygon((0, 0, 1.0), 0.2)
    inside = geo.region_points(np.vstack([near, far]), K, poly, 0.2, 4.0)
    p, reason = geo.robust_surface_point(inside, 10, 0.6, 0.06, 0.03)
    assert p is None and 'one surface' in reason


def test_out_of_range_points_ignored():
    pts = wall_points(5.0)
    poly = qr_polygon((0, 0, 5.0), 0.3)
    assert len(geo.region_points(pts, K, poly, 0.2, 4.0)) == 0


def test_undistort_identity_without_distortion():
    poly = np.array([[100.0, 100.0], [200, 100], [200, 200], [100, 200]])
    assert np.allclose(geo.undistort_polygon(poly, K, np.zeros(5)), poly)
    D = np.array([0.0093, -0.048, 0.0005, -0.0003, 0.0327])       # real Lite3 D
    out = geo.undistort_polygon(poly, K, D)
    assert np.abs(out - poly).max() < 5.0                           # small but applied


def test_scale_polygon():
    poly = geo.scale_polygon([[10, 20], [30, 40]], (320, 240), (640, 480))
    assert np.allclose(poly, [[20, 40], [60, 80]])


def test_flatness_picture_vs_person():
    flat = wall_points(2.0, half=0.3, noise=0.008)
    _, frac, _ = geo.plane_fit(flat, 0.03)
    assert frac > 0.95
    # "person": body 30 cm in front of a wall, region covers both
    body = wall_points(1.7, half=0.15, step=0.005)
    rng = np.random.default_rng(1)
    body[:, 2] -= 0.12 * np.cos(body[:, 0] / 0.15 * 1.5)           # rounded torso
    wall = wall_points(2.0, half=0.3)
    wall = wall[(np.abs(wall[:, 0]) > 0.15)]
    _, frac2, _ = geo.plane_fit(np.vstack([body, wall]) + rng.normal(0, 0.005, (1, 3)), 0.03)
    assert frac2 < 0.85


def _box(pts, x1, y1, x2, y2):
    poly = geo.box_to_polygon(x1, y1, x2, y2)
    return geo.region_points(pts, K, poly, 0.2, 4.0)


def test_picture_core_on_wall_vs_person_box_mostly_wall():
    """The case that fooled a whole-box flatness test in simulation: the person's box is
    mostly wall, but the person (core of the box) stands 30 cm in front of it."""
    wall = wall_points(2.0, half=0.8, noise=0.008, seed=2)
    # printed picture: just the wall itself (paper adds ~0 depth)
    box = (220, 140, 420, 340)
    core = geo.shrink_polygon(geo.box_to_polygon(*box), 0.5)
    full_pts = _box(wall, *box)
    plane = geo.dominant_plane(full_pts, 0.03)
    core_pts = geo.region_points(wall, K, core, 0.2, 4.0)
    assert geo.on_plane_fraction(core_pts, plane, 0.03) > 0.95

    # real person: a 0.24 m wide body 0.3 m in front, occluding the wall behind it
    body = wall_points(1.7, half=0.12, step=0.004)
    hidden = (np.abs(wall[:, 0]) < 0.12 * 2.0 / 1.7) & (np.abs(wall[:, 1]) < 0.12 * 2.0 / 1.7)
    scene = np.vstack([wall[~hidden], body])
    full_pts = _box(scene, *box)
    plane = geo.dominant_plane(full_pts, 0.03)
    # the dominant plane is the WALL (2.0 m), not the body ...
    assert abs(plane[1][2] - 2.0) < 0.03 and geo.on_plane_fraction(full_pts, plane, 0.03) > 0.5
    # ... but the core is the body, off the wall plane -> rejected
    core_pts = geo.region_points(scene, K, core, 0.2, 4.0)
    assert geo.on_plane_fraction(core_pts, plane, 0.03) < 0.5


def test_dominant_plane_is_deterministic_and_vertical():
    wall = wall_points(2.0, half=0.5, noise=0.005, seed=3)
    p1 = geo.dominant_plane(wall, 0.03)
    p2 = geo.dominant_plane(wall, 0.03)
    assert np.allclose(p1[0], p2[0]) and abs(abs(p1[0][2]) - 1.0) < 0.01   # faces camera
