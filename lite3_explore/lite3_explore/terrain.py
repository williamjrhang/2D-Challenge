"""Floor-shape check from the depth camera: is the floor ahead flat, bumpy or holed?

Pure numpy (no ROS), so it can be unit tested. Points are in base_link
(x forward, y left, z up). floor_z = height of flat floor in base_link (negative).

    rubble   = many points a little ABOVE the floor (bumps), but not a wall
    potholes = points clearly BELOW the floor (the camera sees into the holes)
    shadow_check = fallback for potholes from the colour image: dark shadow patches in
                   the bottom third of the frame (uneven ground casts shadows)
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


def update_floor(current, est, ref, tol=0.05, alpha=0.2):
    """Accept a new floor estimate only if it is within tol of the robot's own standing
    height ref (feet are on the floor, so the floor is always ~ref below base_link).
    Live 18/11: a flat raised board in front of the robot was taken as 'the floor'
    (-0.262 instead of -0.356), so the real floor looked like holes -> false potholes."""
    if est is None or abs(est - ref) > tol:
        return current
    return est if current is None else (1 - alpha) * current + alpha * est


def classify(pts, floor_z, x_min=0.4, x_max=1.5, half_width=0.25,
             bump_min=0.03, wall_min=0.20, hole_min=0.04,
             rubble_frac=0.25, hole_frac=0.08, wall_frac=0.10, min_points=200):
    """Return (kind, nearest_x, stats). kind is 'rubble', 'potholes' or None."""
    r = floor_region(pts, x_min, x_max, half_width)
    stats = {'n': int(len(r))}
    if len(r) < min_points or floor_z is None:
        return None, None, stats
    h = r[:, 2] - floor_z
    # A wall BEHIND the rubble (common in the maze: cell, then a wall / turn) must not veto
    # it. If the nearest wall is well past the start of the region, only look at the floor
    # in front of that wall. A wall right at the start still means "wall, not rubble".
    tall = h >= wall_min
    if tall.any():
        x_wall = float(np.percentile(r[tall, 0], 5))
        stats['wall_at'] = round(x_wall, 2)
        if x_wall > x_min + 0.3:
            keep = r[:, 0] < x_wall - 0.1
            if keep.sum() >= min_points:
                r, h = r[keep], h[keep]
                stats['n'] = int(len(r))
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


def surface_wrinkle_mm(pts, floor_z, x_max=1.4, half_width=0.3, layer_m=0.15,
                       min_height=0.12, max_height=0.6, bin_m=0.01, window_bins=11,
                       min_bins=25, min_points=300):
    """How wrinkled the nearest surface ahead is, in mm, or None if too little depth.

    Curtain = hanging fabric with vertical folds -> bumpy everywhere (several mm .. cm).
    Wall / wood board = flat                      -> ~1 mm (camera noise).
    Wall CORNER = two flat planes                  -> one kink only, so it stays low.

    How: keep the nearest layer of points (within layer_m of the closest), build the
    horizontal depth profile x(y) as the median per 1 cm strip, fit a straight line in
    every 10 cm window and take the MEDIAN of the windows' RMS residuals. A single kink
    (corner, door post) only affects a few windows; folds affect all of them."""
    floor = floor_z if floor_z is not None else -0.32
    r = pts[(pts[:, 0] > 0.1) & (pts[:, 0] < x_max) & (np.abs(pts[:, 1]) < half_width)
            & (pts[:, 2] > floor + min_height) & (pts[:, 2] < floor + max_height)]
    if len(r) < min_points:
        return None
    near = np.percentile(r[:, 0], 5)
    r = r[r[:, 0] < near + layer_m]
    if len(r) < min_points:
        return None
    idx = np.floor((r[:, 1] + half_width) / bin_m).astype(int)
    ys, xs = [], []
    for i in np.unique(idx):
        sel = r[idx == i, 0]
        if len(sel) >= 3:
            ys.append((i + 0.5) * bin_m - half_width)
            xs.append(np.median(sel))
    if len(ys) < min_bins:
        return None
    ys, xs = np.array(ys), np.array(xs)
    rms = []
    for k in range(len(ys) - window_bins + 1):
        y, x = ys[k:k + window_bins], xs[k:k + window_bins]
        if y[-1] - y[0] > 1.5 * window_bins * bin_m:     # gap in the profile: skip
            continue
        A = np.c_[y, np.ones(len(y))]
        coef, *_ = np.linalg.lstsq(A, x, rcond=None)
        rms.append(np.sqrt(np.mean((x - A @ coef) ** 2)))
    if len(rms) < 5:
        return None
    return float(np.median(rms) * 1000.0)


def shadow_check(gray, x_frac=(0.25, 0.75), grid=(4, 8), dark_ratio=0.6,
                 dark_cells_frac=0.15, min_median=40.0):
    """Potholes by lighting, for when the depth check finds nothing (holes too shallow /
    no depth). Uneven ground in front of the robot casts shadows: in the BOTTOM THIRD of
    the image (floor right ahead) some areas are much darker than the rest of the floor.

    gray = 2-D brightness image (0..255), e.g. HSV V channel or grayscale.
    Only the centre columns (x_frac, the robot's path) of the bottom third are used.
    That region is split into a grid of cells; the mean brightness of each cell is compared
    with the median cell. A cell darker than dark_ratio * median is a shadow cell; more than
    dark_cells_frac of the cells in shadow -> potholes. Comparing cells with each other (not
    with a fixed level) makes it work in both bright and dim venue light. The grid also
    averages away pixel noise and small dark specks.

    Returns (found, stats)."""
    g = np.asarray(gray, dtype=float)
    h, w = g.shape[:2]
    band = g[2 * h // 3:, int(x_frac[0] * w):int(x_frac[1] * w)]
    rows, cols = grid
    bh, bw = band.shape[0] // rows, band.shape[1] // cols
    stats = {}
    if bh < 2 or bw < 2:
        return False, stats
    cells = band[:bh * rows, :bw * cols].reshape(rows, bh, cols, bw).mean(axis=(1, 3))
    med = float(np.median(cells))
    stats['median'] = round(med, 1)
    if med < min_median:            # floor too dark overall: shadows are not measurable
        stats['dark'] = 0.0
        return False, stats
    dark = cells < dark_ratio * med
    stats['dark'] = round(float(dark.mean()), 2)
    stats['darkest'] = round(float(cells.min() / med), 2)
    return bool(dark.mean() > dark_cells_frac), stats


def wall_ahead_m(pts, floor_z, half_width=0.25, x_min=0.15, x_max=2.0, wall_min=0.20,
                 min_points=60):
    """Distance (base_link x) to the nearest WALL in the robot's path, or None.
    Wall = points at least wall_min above the floor (rubble boards / slats are lower).
    Used to stop rubble / pothole crossings before they walk into a wall (live 18/11:
    potholes crossing walks 1.6 m straight, wall seen at 1.11 m)."""
    if floor_z is None:
        return None
    r = pts[(pts[:, 0] > x_min) & (pts[:, 0] < x_max) & (np.abs(pts[:, 1]) < half_width)
            & (pts[:, 2] > floor_z + wall_min)]
    if len(r) < min_points:
        return None
    return float(np.percentile(r[:, 0], 5))
