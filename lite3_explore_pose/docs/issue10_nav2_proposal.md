# Issue 10: Nav2 values for 1.02 m cells (proposal only, not applied)

You said "don't touch slam/nav2", so these are numbers only. They live in
`~/ros2_ws/src/lite3_launcher/config/nav2_params.yaml` (not in git). The explorer now
routes through cell centres and doorway midpoints (`cell_routing`), which keeps paths
centred without any Nav2 change.

Robot footprint 0.61 x 0.37 m: inscribed radius 0.185 m, circumscribed 0.357 m. Centred in
a ~1.0 m doorway the robot is ~0.50 m from each side. Inflation cost:
`252 * exp(-k * (d - 0.185))` for d up to inflation_radius (lethal only inside 0.185 m).

| Parameter | Before | Cost at doorway centre (d=0.50) / near the wall (d=0.30) | Proposed | After |
|---|---|---|---|---|
| global `inflation_radius` | 0.55 | 98 / 178 | 0.60 | covers the whole half-corridor from both walls |
| global `cost_scaling_factor` | 3.0 | | 4.0 | 72 / 159: clearer valley in the middle |
| local `inflation_radius` | 1.0 | whole corridor ~100-180, almost flat | 0.60 | the controller sees a gradient again |
| local `cost_scaling_factor` | 3.0 | | 4.0 | same as global |
| planner `GridBased.tolerance` | 0.5 | goals inside walls are moved up to 0.5 m silently | 0.25 | |
| controller `goal_checker` (none -> Humble default) | xy 0.25, yaw 0.25 | | `general_goal_checker: SimpleGoalChecker, xy_goal_tolerance: 0.20, yaw_goal_tolerance: 0.25, stateful: true` | explicit, same as the explorer's centre_tolerance_m order |
| bt_navigator `default_bt_xml_filename` | not a Humble name (ignored) | | remove, or `default_nav_to_pose_bt_xml` | |

Doorways stay passable with every proposed value: lethal only within 0.185 m of a wall, the
doorway centre is 0.50 m from both sides.
