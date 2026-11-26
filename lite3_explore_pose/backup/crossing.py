"""Crossing decisions for special_obstacles (issues 6 and 8). Pure Python, no ROS.

Kept out of the node so the rules that decide when the robot leaves AI mode, which gait
it walks off in and when a terrain crossing is really finished can be unit-tested.
"""
import math

TERRAIN = ('rubble', 'potholes')
STOPPED_ON_RUBBLE = 'stopped_on_rubble'  # /special/done outcome requiring manual recovery


def terrain_in(dets, ahead_m=1.2, kinds=TERRAIN):
    """True if a rubble / potholes detection is in the path (trigger, or 0 < forward <= ahead_m)."""
    for d in dets or ():
        if d.get('kind') in kinds and (d.get('trigger')
                                       or 0.0 < d.get('forward_m', 99.0) <= ahead_m):
            return True
    return False


def gait_after(dets, age_s, fresh_s=1.5, ahead_m=1.2, terrain_gait=None):
    """Gait to walk on in after the stairs (issue 6), from the latest detector frame.
    terrain_gait(kind) -> gait name. Returns (name, why).
      rubble / potholes ahead -> its terrain gait (high_step)
      flat floor ahead        -> flat
      no fresh data           -> flat"""
    if dets is None or age_s is None or age_s > fresh_s:
        return 'flat', 'no fresh detector data: flat'
    for d in dets:
        if d.get('kind') in TERRAIN and terrain_in([d], ahead_m):
            name = terrain_gait(d['kind']) if terrain_gait else 'high_step'
            if name:
                return name, f'{d["kind"]} ahead at {d.get("forward_m", "?")} m'
    return 'flat', 'flat floor ahead'


def stairs_exit(level_s, past_m, flights_done, level_hold_s=1.5, exit_after_m=0.35,
                exit_max_m=0.9):
    """When to leave AI mode after the stairs (organisers: exit at most 1 cell after them).

    level_s:      seconds the robot has been level (None = not level)
    past_m:       odometry distance walked since the last tilted (on-step) sample
    flights_done: True once the stairs are over (up AND down for stairs_up_and_down)
    Returns 'wait' | 'exit' | 'exit_cap'."""
    if not flights_done:
        return 'wait'
    if past_m is not None and past_m >= exit_max_m:
        return 'exit_cap'                      # rule: never more than ~1 cell after
    if level_s is not None and level_s >= level_hold_s and past_m is not None \
            and past_m >= exit_after_m:
        return 'exit'
    return 'wait'


def walked_enough(odo_m, timed_m, target_m, odom_fresh, time_factor=1.3, odo_agree=0.7):
    """Issue 8: is a crossing's distance done?
    Fresh odometry: ONLY odometry decides (odo >= target, with speed x time >= odo_agree x
    target as a sanity check against a jump). Never by time alone while odometry is fresh:
    22-25/11 the robot 'finished' on time while odometry showed it stuck on the rubble.
    No odometry: speed x time >= time_factor x target (old fallback)."""
    if odom_fresh and odo_m is not None:
        return odo_m >= target_m and timed_m >= odo_agree * target_m
    return timed_m >= time_factor * target_m


def odom_jumped(prev_m, now_m, dt_s, max_speed=0.6, slack_m=0.15):
    """Odometry distance grew faster than the robot can walk (25/11: 'floor ahead clear at
    11.30 m' 10 s into a rubble crossing)."""
    if prev_m is None or now_m is None or dt_s <= 0:
        return False
    return now_m - prev_m > max_speed * dt_s + slack_m


def verify_result(progress_m, need_m, max_tilt_deg, tilt_limit_deg):
    """Issue 8 VERIFY after switching back to flat: 'ok' | 'stuck' | 'tilted'."""
    if max_tilt_deg is not None and max_tilt_deg > tilt_limit_deg:
        return 'tilted'
    if progress_m is None or progress_m < need_m:
        return 'stuck'
    return 'ok'


def deg(rad):
    return math.degrees(rad)


# ================= curtain (issue 7) =================

def approach_pose(obs_xy, from_xy, standoff_m, arena=None, square=False):
    """Where to stand to face an obstacle: standoff_m before it, facing it.
    square=True (curtain): the direction is snapped to the nearest arena axis and the
    curtain's lateral position to the middle of its cell row, so the robot faces the
    doorway squarely (curtains hang in doorways between 1.02 m cells).
    Returns (x, y, yaw) in the map frame."""
    ox, oy = obs_xy
    fx, fy = from_xy
    if square and arena is not None:
        ax, ay = arena.map_to_arena(ox, oy)
        bx, by = arena.map_to_arena(fx, fy)
        dx, dy = ax - bx, ay - by
        sx, sy = arena.start
        c = arena.cell
        if abs(dx) >= abs(dy):
            ux, uy = (1.0 if dx >= 0 else -1.0), 0.0
            ay = sy + round((ay - sy) / c) * c           # middle of the cell row
        else:
            ux, uy = 0.0, (1.0 if dy >= 0 else -1.0)
            ax = sx + round((ax - sx) / c) * c
        gx, gy = arena.arena_to_map(ax - standoff_m * ux, ay - standoff_m * uy)
        tx, ty = arena.arena_to_map(ax, ay)
        return gx, gy, math.atan2(ty - gy, tx - gx)
    dx, dy = ox - fx, oy - fy
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    return ox - standoff_m * ux, oy - standoff_m * uy, math.atan2(uy, ux)


def far_side(point, portal_xy, from_xy, to_xy):
    """True if point is on the side of a crossed curtain the robot walked INTO."""
    dx, dy = to_xy[0] - from_xy[0], to_xy[1] - from_xy[1]
    return (point[0] - portal_xy[0]) * dx + (point[1] - portal_xy[1]) * dy > 0.0
