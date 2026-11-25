# Issue 3: noisy map during autonomous runs

Status: **proposal only.** Nothing in SLAM, the scan filter or Nav2 was changed (you said
"don't touch slam/nav2"). These files live in `~/ros2_ws/src/lite3_launcher/config/`, which
is not in git. Apply a row only after you OK it, then check it with `tools/diag_rates.py`.

## What feeds slam_toolbox

`/cloud_registered_body` (FAST-LIO, de-skewed, body frame) -> `pointcloud_to_laserscan`
(`pointcloud_to_scan.yaml`) -> `/scan` -> slam_toolbox (`nav2_params.yaml`, key `slam_toolbox`).

| Check | Today | Finding |
|---|---|---|
| Robot body / legs | `range_min: 0.4` | OK: the 0.61 x 0.37 m body's farthest corner is 0.36 m from base_link |
| Floor | `min_height: -0.18` (floor at -0.356 in base_link) | OK: cuts the floor up to 0.18 m above it |
| Ceiling / above walls | `max_height: 0.5` | **Too high.** Wall top = 0.54 m above the floor = +0.184 m in base_link. 0.5 keeps a 0.32 m band **above the walls**: people and objects outside the arena within `range_max` land in the map. This matches the streaks and blue frontiers outside the arena in your RViz screenshot. |
| Spectators | `range_max: 6.0`, no box filter | Nothing removes points outside the arena |

## Proposed changes (before -> after)

| File | Parameter | Before | After | Why |
|---|---|---|---|---|
| pointcloud_to_scan.yaml | `max_height` | 0.5 | **0.17** | just under the wall top (+0.184). 20/11: 0.13 kept only 11% of wall hits; 0.17 keeps most of the wall face the Mid-360 sees and drops everything above the walls. Check the wall hit ratio in RViz before keeping it. |
| pointcloud_to_scan.yaml | `range_max` | 6.0 | 5.5 | arena diagonal (5 x 1.02 cells + a corner) |
| nav2_params.yaml slam_toolbox | `do_loop_closing` | true (default) | **false** | the maze is repetitive (identical 1.02 m cells): a false loop closure moves the whole map by metres. The logs show 3.0-8.5 m jumps. FAST-LIO drift over a 10 min run in a 6 x 6 m arena is small. |
| " | `link_match_minimum_response_fine` | 0.1 (default) | 0.3 | reject weak scan matches (smeared scans while turning) |
| " | `minimum_travel_heading` | 0.3 | 0.2 | more scans during in-place turns, each less rotated |
| " | `transform_publish_period` | 0.02 | 0.05 | 20 Hz map->odom is enough; less TF traffic over the link |
| nav2_params.yaml | `behavior_server` (missing; the file has the old `recoveries_server` key, ignored in Humble) | spin at default `max_rotational_vel` 1.0 rad/s | add `behavior_server: max_rotational_vel: 0.5` | the explorer's `spin_speed: 0.5` only sets the time allowance; the real spin is 1.0 rad/s |
| nav2_params.yaml controller MPPI | `wz_max` | 1.9 | 0.8 | fast yaw smears scans on a legged robot |
| nav2_params.yaml | `velocity_smoother` (not configured) | max theta 2.5 rad/s (default) | `max_velocity: [0.5, 0.0, 0.8]`, `min_velocity: [-0.35, 0.0, -0.8]` | caps every yaw source the same way |

## Already in place (no change)

- turn_hold pauses 3 s after each Nav2 turn **only when Nav2 is started by
  `launch/robot.launch.py`**. `tools/auto_run.sh` started `lite3_launcher` directly, so it was
  off in those runs; fixed in issue 9 (`auto_run.sh` -> `robot.launch.py`).
- Every camera node now fetches depth on demand (`locator.py`), YOLO is limited to 2 threads,
  nice 10, 2 Hz.
- Without changing SLAM, the explorer can ignore the outside noise:
  `use_grid_bounds_box: true` in `explorer.yaml` limits frontiers to the `grid_bounds_cells` box.

## Measuring

On the robot and on the desktop during a run:

```bash
python3 ~/challenge_ws/lite3_explore_pose/tools/diag_rates.py --period 2
```

Healthy: `/livox/imu` >= 150 Hz (normally 200), `/fastlio/odometry` and `/odom` >= 8 Hz
(normally 10), `/map` ~0.5 Hz, no LOW lines, robot link rx well under 5 MB/s.
