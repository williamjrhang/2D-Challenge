"""Stairs from 3D shape (depth points in base_link). Pure numpy, no ROS.

Why: on photos of the arena stairs (shiny aluminium checker plate, closed risers, no
handrails) YOLO-World saw nothing / "plain wooden wall", and the trained terrain_v1.pt said
"rubble" 0.70-0.85. The SHAPE is unmistakable instead: >= 2 flat treads rising in regular
steps. Rubble is low and irregular, a wall is one tall jump.

stairs_profile(): points in a strip ahead of the robot -> height profile (top surface per
5 cm slice along x) -> flat plateaus -> rises between them. Generic limits, no measuring
needed (rise 8-25 cm, tread >= 12 cm). Missing slices (reflections on the metal) are
tolerated. Also estimates the robot's yaw to the first step edge (AI mode climbs straight).
"""
import math

import numpy as np


def height_profile(pts, floor_z, half_width=0.3, x_min=0.25, x_max=2.5, bin_m=0.05,
                   min_pts=6, top_pct=90.0):
    """(x centres, top height above floor per slice or nan)."""
    if pts is None or len(pts) == 0:
        return np.array([]), np.array([])
    p = np.asarray(pts, dtype=float)
    sel = (np.abs(p[:, 1]) <= half_width) & (p[:, 0] >= x_min) & (p[:, 0] < x_max)
    p = p[sel]
    edges = np.arange(x_min, x_max + 1e-9, bin_m)
    xc = (edges[:-1] + edges[1:]) / 2
    prof = np.full(len(xc), np.nan)
    if len(p) == 0:
        return xc, prof
    idx = np.clip(((p[:, 0] - x_min) / bin_m).astype(int), 0, len(xc) - 1)
    h = p[:, 2] - floor_z
    order = np.argsort(idx, kind='stable')
    idx, h = idx[order], h[order]
    starts = np.searchsorted(idx, np.arange(len(xc)))
    ends = np.searchsorted(idx, np.arange(len(xc)), side='right')
    for i, (a, b) in enumerate(zip(starts, ends)):
        if b - a >= min_pts:
            prof[i] = np.percentile(h[a:b], top_pct)
    return xc, prof


def plateaus(xc, prof, flat_tol=0.03, max_gap_bins=2, min_bins=2):
    """Runs of slices at the same height: [(x_start, x_end, level), ...] in x order.
    Gaps of up to max_gap_bins missing slices inside a run are bridged."""
    out = []
    run, gap = [], 0
    for x, v in zip(xc, prof):
        if np.isnan(v):
            gap += 1
            if run and gap > max_gap_bins:
                out.append(run)
                run = []
            continue
        if run and abs(v - np.median([r[1] for r in run])) > flat_tol:
            out.append(run)
            run = []
        run.append((x, v))
        gap = 0
    if run:
        out.append(run)
    half = (xc[1] - xc[0]) / 2 if len(xc) > 1 else 0.025
    return [(r[0][0] - half, r[-1][0] + half, float(np.median([v for _, v in r])))
            for r in out if len(r) >= min_bins]


def riser_yaw_deg(pts, floor_z, x_edge, rise, half_width=0.5, band_m=0.15, min_pts=20):
    """Robot yaw relative to the first step edge (0 = square on), from the first riser's
    points: fit x = a + b*y, yaw = atan(b). None when there are too few points."""
    p = np.asarray(pts, dtype=float)
    h = p[:, 2] - floor_z
    sel = ((np.abs(p[:, 1]) <= half_width) & (np.abs(p[:, 0] - x_edge) <= band_m)
           & (h > 0.2 * rise) & (h < 0.8 * rise))
    q = p[sel]
    if len(q) < min_pts or np.ptp(q[:, 1]) < 0.2:
        return None
    b = np.polyfit(q[:, 1], q[:, 0], 1)[0]
    return math.degrees(math.atan(b))


def stairs_profile(pts, floor_z, half_width=0.3, x_min=0.25, x_max=2.5, bin_m=0.05,
                   rise_min=0.08, rise_max=0.25, tread_min=0.12, min_steps=2,
                   wall_jump=0.40, flat_tol=0.03):
    """Returns dict: stairs (bool), first_step_m, steps, rises, reason, yaw_deg."""
    res = {'stairs': False, 'first_step_m': None, 'steps': 0, 'rises': [], 'yaw_deg': None,
           'reason': ''}
    if floor_z is None:
        res['reason'] = 'no floor height'
        return res
    xc, prof = height_profile(pts, floor_z, half_width, x_min, x_max, bin_m)
    if not len(xc) or np.isnan(prof).all():
        res['reason'] = 'no depth points ahead'
        return res
    pl = plateaus(xc, prof, flat_tol)
    if len(pl) < 2:
        res['reason'] = 'no height change ahead'
        return res
    best = []                               # longest run of regular rising steps
    cur = []
    for (a0, a1, l0), (b0, b1, l1) in zip(pl, pl[1:]):
        rise = l1 - l0
        tread_ok = (a1 - a0) >= tread_min or not cur   # first plateau may be the floor (cut)
        if rise > wall_jump and not cur:
            res['reason'] = f'wall: jump {rise:.2f} m at {b0:.2f} m'
            return res
        if rise_min <= rise <= rise_max and tread_ok:
            cur.append((b0, rise))
        else:
            if len(cur) > len(best):
                best = cur
            cur = []
    if len(cur) > len(best):
        best = cur
    res['steps'] = len(best)
    res['rises'] = [round(float(r), 3) for _, r in best]
    if len(best) < min_steps:
        res['reason'] = f'{len(best)} regular step(s) (need {min_steps})'
        return res
    rises = [r for _, r in best]
    if max(rises) - min(rises) > 0.5 * float(np.mean(rises)):
        res['reason'] = f'irregular rises {res["rises"]} (rubble?)'
        return res
    first_x, first_rise = best[0]
    k = next(i for i, (a0, _, _) in enumerate(pl) if abs(a0 - first_x) < 1e-9)
    below = pl[k - 1]                               # plateau under the first counted rise
    floor_seen = below[2] <= rise_min
    if not floor_seen:                              # already at the steps: floor out of view,
        first_x = below[0]                          # the nearest visible tread is a step too
    res['floor_seen'] = bool(floor_seen)
    res.update(stairs=True, first_step_m=round(float(first_x), 2),
               yaw_deg=riser_yaw_deg(pts, floor_z, first_x, first_rise),
               reason=f'{len(best)} steps, rises {res["rises"]} m')
    if res['yaw_deg'] is not None:
        res['yaw_deg'] = round(float(res['yaw_deg']), 1)
    return res
