"""Image region + depth points -> 3D target point. Pure numpy/OpenCV, no ROS.

Pipeline used by the node:

  1. depth cloud (depth optical frame, time t_depth)
       --T_color<-depth-->  colour optical frame at the image time t_img
  2. project every point with the colour intrinsics K (pinhole, no distortion)
  3. undistort the detection polygon into the same pinhole pixel space
  4. keep points whose projection falls inside the (shrunk) polygon
  5. robust depth: median range, keep points near it, require enough of them
  6. the median of those inliers is the target point (on the wall surface)

Nothing here ever invents a point: every failure returns a reason string instead.
"""

import math

import cv2
import numpy as np


# ---------------------------------------------------------------- transforms

def quat_to_matrix(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError('zero quaternion')
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def make_transform(translation, quaternion):
    """4x4 homogeneous matrix from (tx, ty, tz) and (qx, qy, qz, qw)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*quaternion)
    T[:3, 3] = translation
    return T


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def apply_transform(T, pts):
    """pts: (N, 3) -> (N, 3)."""
    return pts @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------- image space

def scale_polygon(poly, image_size, info_size):
    """Rescale pixel coords from the decoded image size to the camera_info size."""
    (iw, ih), (cw, ch) = image_size, info_size
    poly = np.asarray(poly, dtype=float).reshape(-1, 2).copy()
    if (iw, ih) != (cw, ch):
        poly[:, 0] *= cw / float(iw)
        poly[:, 1] *= ch / float(ih)
    return poly


def undistort_polygon(poly, K, D):
    """Distorted pixel coords -> ideal pinhole pixel coords (same K)."""
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 1, 2)
    if D is None or len(D) == 0 or not np.any(np.asarray(D)):
        return poly.reshape(-1, 2)
    out = cv2.undistortPoints(poly, K, np.asarray(D, dtype=np.float64), P=K)
    return out.reshape(-1, 2)


def shrink_polygon(poly, factor):
    """Scale a polygon about its centroid (factor < 1 shrinks it)."""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    c = poly.mean(axis=0)
    return c + (poly - c) * float(factor)


def box_to_polygon(x1, y1, x2, y2):
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)


def polygon_area(poly):
    p = np.asarray(poly, dtype=float).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def points_in_convex_polygon(u, v, poly):
    """Vectorised inside test for a convex polygon (either winding)."""
    poly = np.asarray(poly, dtype=float).reshape(-1, 2)
    n = len(poly)
    pos = np.ones(u.shape, dtype=bool)
    neg = np.ones(u.shape, dtype=bool)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (v - y0) - (y1 - y0) * (u - x0)
        pos &= cross >= 0
        neg &= cross <= 0
    return pos | neg


def project(pts_cam, K):
    """Pinhole projection of points in an optical frame (z forward). Returns u, v."""
    z = pts_cam[:, 2]
    u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
    return u, v


# ---------------------------------------------------------------- estimation

def region_points(pts_cam, K, poly, min_range, max_range):
    """Points (colour optical frame) whose projection lies inside `poly`."""
    z = pts_cam[:, 2]
    ok = np.isfinite(pts_cam).all(axis=1) & (z > min_range) & (z < max_range)
    pts = pts_cam[ok]
    if len(pts) == 0:
        return pts
    # cheap bounding-box prefilter before the polygon test
    u, v = project(pts, K)
    poly = np.asarray(poly, dtype=float)
    bb = ((u >= poly[:, 0].min()) & (u <= poly[:, 0].max())
          & (v >= poly[:, 1].min()) & (v <= poly[:, 1].max()))
    pts, u, v = pts[bb], u[bb], v[bb]
    if len(pts) == 0:
        return pts
    return pts[points_in_convex_polygon(u, v, poly)]


def robust_surface_point(pts, min_points, min_inlier_frac, tol_abs, tol_rel):
    """Median point of the dominant depth layer inside a region.

    Returns (point (3,), info dict) or (None, reason str).
    """
    n = len(pts)
    if n < min_points:
        return None, f'only {n} valid depth points in region (need {min_points})'
    rng = np.linalg.norm(pts, axis=1)
    med = float(np.median(rng))
    tol = max(tol_abs, tol_rel * med)
    inl = np.abs(rng - med) <= tol
    k = int(inl.sum())
    if k < min_points:
        return None, f'only {k} depth points agree on one surface (need {min_points})'
    if k / n < min_inlier_frac:
        return None, (f'depth inside region is not one surface '
                      f'({100.0 * k / n:.0f}% within {tol * 100:.0f} cm, '
                      f'need {100 * min_inlier_frac:.0f}%)')
    p = np.median(pts[inl], axis=0)
    return p, {'points': n, 'inliers': k, 'range_m': float(np.linalg.norm(p))}


def plane_fit(pts, tol):
    """Least-squares plane. Returns (normal (3,), fraction of points within tol, rms)."""
    if len(pts) < 3:
        return None, 0.0, float('inf')
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    nrm = vt[-1]
    d = np.abs((pts - c) @ nrm)
    return nrm, float((d <= tol).mean()), float(np.sqrt((d ** 2).mean()))


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def dominant_plane(pts, tol, iters=80, seed=0, max_points=3000):
    """RANSAC plane of the dominant surface in `pts` (e.g. the wall around a picture).

    Returns (unit normal (3,), point on plane (3,)) or None. Deterministic (fixed seed).
    """
    if len(pts) < 10:
        return None
    rng = np.random.default_rng(seed)
    sub = pts if len(pts) <= max_points else pts[rng.choice(len(pts), max_points, replace=False)]
    best, best_k = None, -1
    for _ in range(iters):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        k = int((np.abs((sub - a) @ n) <= tol).sum())
        if k > best_k:
            best, best_k = (n, a), k
    if best is None:
        return None
    n, a = best
    inl = sub[np.abs((sub - a) @ n) <= tol]
    c = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
    return vt[-1], c


def on_plane_fraction(pts, plane, tol):
    if plane is None or len(pts) == 0:
        return 0.0
    n, c = plane
    return float((np.abs((pts - c) @ n) <= tol).mean())
