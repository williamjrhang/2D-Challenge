"""Floor-shape check from the depth camera: is the floor ahead flat, bumpy or holed?

Pure numpy (no ROS), so it can be unit tested. Points are in base_link
(x forward, y left, z up). floor_z = height of flat floor in base_link (negative).

    rubble   = many points a little ABOVE the floor (bumps), but not a wall
    potholes = points clearly BELOW the floor (the camera sees into the holes)
"""

import numpy as np


def floor_region(pts, x_min, x_max, half_width):
    m = (pts[:, 0] > x_min) & (pts[:, 0] < x_max) & (np.abs(pts[:, 1]) < half_width)
    return pts[m]


def estimate_floor(pts, bands=((0.3, 0.6), (0.5, 1.2)), half_width=0.25, max_spread=0.02,
                   min_points=200):
    """Floor height from the first flat band ahead (small height spread), else None.
    The band right in front of the feet is tried first: further ahead may already be the
    obstacle (live test: slats from 0.6 m, so the 0.5-1.2 m band was never flat)."""
    for x_min, x_max in bands:
        r = floor_region(pts, x_min, x_max, half_width)
        if len(r) < min_points:
            continue
        lo, med, hi = np.percentile(r[:, 2], [10, 50, 90])
        if hi - lo < 2 * max_spread:
            return float(med)
    return None


def classify(pts, floor_z, x_min=0.4, x_max=1.5, half_width=0.25,
             bump_min=0.03, wall_min=0.20, hole_min=0.04,
             rubble_frac=0.25, hole_frac=0.08, wall_frac=0.10, min_points=200):
    """Return (kind, nearest_x, stats). kind is 'rubble', 'potholes' or None."""
    r = floor_region(pts, x_min, x_max, half_width)
    stats = {'n': int(len(r))}
    if len(r) < min_points or floor_z is None:
        return None, None, stats
    h = r[:, 2] - floor_z
    bump = (h > bump_min) & (h < wall_min)
    wall = h >= wall_min
    hole = h < -hole_min
    stats.update(bump=round(float(bump.mean()), 2), wall=round(float(wall.mean()), 2),
                 hole=round(float(hole.mean()), 2))
    if hole.mean() > hole_frac:
        return 'potholes', float(r[hole, 0].min()), stats
    if bump.mean() > rubble_frac and wall.mean() < wall_frac:
        return 'rubble', float(r[bump, 0].min()), stats
    return None, None, stats
