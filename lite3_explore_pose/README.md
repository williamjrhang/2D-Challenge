# lite3_explore_pose — Group 5 ground robot (Lite3)

This copy adds localization checks to the explorer’s Nav2 goals, including frontier,
camera coverage, grid, and room goals. Run one explorer at a time because both packages
use the same ROS topics and Nav2 action servers.

## Find where the pose goes stale

### 1. Isolate Nav2 and send one short goal

Start the robot, SLAM, and Nav2. Source the robot workspace in each terminal. For this
baseline, leave both exploration launches stopped. YOLO runs in the exploration launch,
so the baseline is also a test with YOLO off.

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch lite3_launcher lite3.launch.py slam:=True navigation:=True
```

In RViz set the fixed frame to `map` and check that the robot, scan, and map line up.
With the robot in open space, run `lite3_auto` and use RViz's **2D Goal Pose** tool
to place a goal about 1 m in front of the robot, with the arrow facing forward. Start the
monitors below before sending the goal. Note whether the physical robot and its RViz pose
both move and whether Nav2 reports success at the actual target.

### 2. Watch each TF link in its own terminal

```bash
ros2 run tf2_ros tf2_echo map odom
ros2 run tf2_ros tf2_echo odom base_link
```

Watch the **At time** values. They should keep advancing even while the robot is still.
A fixed timestamp, recurring lookup error, or growing delay identifies the link to
investigate. This setup normally has some SLAM delay. Compare the delay during the stall
with the baseline rather than inferring a fault from a single sample. For a delay summary
while the goal runs, use `ros2 run tf2_ros tf2_monitor map odom` or
`ros2 run tf2_ros tf2_monitor odom base_link`.

### 3. Trace the link that stalls

| Observation | Where to check |
| --- | --- |
| `odom → base_link` stops advancing | Compare the rates and header timestamps of `/fastlio/odometry` and `/odom`. If both stall or carry old timestamps, check Fast-LIO, its LiDAR and IMU inputs, and robot CPU load. |
| `/fastlio/odometry` is fresh but `/odom` stalls | Check `odom_relay` and the topic endpoints. The relay copies the source timestamps. |
| `/odom` is fresh but `odom → base_link` is stale | Check `odom_relay` TF publishing, the frame names, and other publishers on `/tf`. |
| `odom → base_link` stays fresh but `map → odom` stalls | Check `slam_toolbox` logs, the `/scan` rate and timestamps, and SLAM's scan-frame transform. |
| Both TF links stay fresh | Check `ros2 run tf2_ros tf2_echo map base_link`, the Nav2 result, and `/nav_cmd_vel` versus `/cmd_vel` with a goal active. If Nav2 reports success away from the physical target, compare the actual RViz pose and goal and check the Nav2 goal checker configuration. |

Run each of these in its own terminal as needed. `hz` measures messages received.
Header timestamps reveal old data that still arrives regularly. On a real robot, compare
the header stamps with the current ROS system time on the same synchronized machine.

```bash
ros2 topic hz /fastlio/odometry
ros2 topic hz /odom
ros2 topic echo /fastlio/odometry --field header --qos-reliability best_effort
ros2 topic echo /odom --field header --qos-reliability best_effort
ros2 topic hz /scan
ros2 topic hz /imu/data
```

### 4. Test the copied explorer

After the standalone Nav2 test has ended, build and start the copy with YOLO, QR and VLM
disabled to isolate the explorer. Keep the same TF monitors running.

```bash
cd ~/challenge_ws
colcon build --packages-select lite3_explore_pose
source install/setup.bash
ros2 launch lite3_explore_pose mission.launch.py live:=true use_yolo:=false use_qr:=false use_vlm:=false
```

Read `log/lite3_explore_pose_decisions.log` or echo `/exploration/status`.
`WAIT_POSE` means the explorer is waiting for localization. It logs the missing or old
TF link and cancels any active explorer action. With the default settings it waits for
one second of healthy TF and an advancing odometry stamp, then retries an interrupted
goal once if it still applies. It confirms a Nav2 success with newer odometry and a
newer map-frame pose within 0.35 m of the goal. A second localization-related failure
defers the goal for 15 seconds, after which it can be selected again. These thresholds are in
`config/explorer.yaml`. For a perception load comparison, repeat this run with
`use_yolo:=true` and leave `publish_obstacles` at its default of `false`.

## Pause after a Nav2 turn

Start the robot stack with the copied package's launch in place of the usual robot
bringup. It runs the same robot, SLAM, and Nav2 nodes and routes Nav2 velocity through
the turn hold. The joystick retains priority and special obstacle crossings can take over.

```bash
source ~/ros2_ws/install/setup.bash
cd ~/challenge_ws
colcon build --packages-select lite3_explore_pose
source install/setup.bash
ros2 launch lite3_explore_pose robot.launch.py slam:=True navigation:=True
```

Nav2 turns in place and holds still for three seconds after a substantial turn before
it can resume translation. The hold duration is the robot launch argument
`post_turn_hold_s`. The turn-hold node logs the start of each pause.

Use a separate terminal to start the mission with the obstacle detector and crossing
events enabled. Both are necessary for the robot to switch gait after detecting rubble.

```bash
source ~/ros2_ws/install/setup.bash
cd ~/challenge_ws
source install/setup.bash
lite3_auto
ros2 launch lite3_explore_pose mission.launch.py live:=true grid_explore:=true \
  use_yolo:=true publish_obstacles:=true use_qr:=false use_vlm:=false
```

For a bag, add `/nav_cmd_vel_raw`, `/navigation/turn_hold_active`, and
`/perception/event` alongside `/nav_cmd_vel`, `/cmd_vel`, and `/mission/state`.
This captures what Nav2 asked for, when translation was held, and whether an obstacle
event triggered a crossing.

### If the robot pose jumps or visited cells stay grey

If the explorer logs a jump in `odom → base_link` or `map → base_link`, restart the
robot bringup and explorer from the marked start so SLAM, the arena grid and visited
history share a frame. The mission waits for `/exploration/localization_ready` before
acting on detections or taking over motion. It stops an active crossing, approach or
exit drive if localization becomes unavailable.

Open the copied `config/maze.rviz` with fixed frame `map`. Its **Maze cells (green
visited)** display uses `/exploration/cells`. A cell turns green after two advancing
valid pose samples inside its interior or after a grid goal arrival confirmed inside
that same cell. The default 0.10 m edge margin avoids visits from poses on a cell
boundary. Check the explorer log for `Cell (i, j) visited`, then check RViz's marker
topic if the log shows a visit and the display stays grey.

The robot explores the arena on its own (frontier exploration + Nav2, then it drives to
whatever the camera has not looked at yet), crosses stairs /
uneven terrain / curtains with a special walking routine, finds survivors, hazard signs,
rooms and QR codes, reports them to the scoring server, and heads to the final cell.

```
camera ──► obstacle_detector (YOLO-World, local, ~3 fps) ──► stairs/curtain/rubble_ahead ─┐
camera ──► qr_reader (OpenCV) ─────────────────────────────► qr ──────────────────────────┤
camera ──► vlm_watcher (course scene server, while moving) ─► room / hazard / survivor    │
                     │                                                                      ▼
                     └────────► reporter ──► scoring server        mission_controller ──► special_obstacles
                                    │        (task_report)            │   (pause explorer,      (stop, gait,
                                    └─► /mission/tasks_done ─────────►│    cross, resume,        walk straight,
                                                                       │    time limit, exit)     gait back)
                                    explorer (frontiers ──► Nav2) ◄────┘
```

## Files

| File | What it does |
| --- | --- |
| `lite3_explore_pose/frontiers.py` | Finds and scores frontiers on the map. Pure numpy, no ROS. |
| `lite3_explore_pose/coverage.py` | Camera coverage: pose history, what the forward camera has seen (raycast on the map), goals where it has not looked, save / load. Pure numpy, no ROS. |
| `lite3_explore_pose/explorer_node.py` | Frontier exploration with Nav2. At start: sets the walking gait, then a Nav2 look-around spin (stops before walls). Timeouts, blacklist. No frontiers left → COVER (camera coverage). HOLD = stand still so the obstacle detector can confirm. |
| `lite3_explore_pose/special_obstacles.py` | Crosses stairs / rubble / curtain: stop → switch gait → walk straight → switch back. |
| `lite3_explore_pose/mission_controller.py` | The "boss": explore / look / cross / go to final cell, overall time limit. |
| `lite3_explore_pose/obstacle_detector.py` | YOLO-World (no training) + depth: sends `stairs_ahead`, `curtain_ahead`, `rubble_ahead`. |
| `lite3_explore_pose/vlm_watcher.py` | Asks the course scene server (room, hazard, survivor, objects) in the background while moving. |
| `lite3_explore_pose/qr_reader.py` | Reads QR codes with OpenCV, places them on the map. |
| `lite3_explore_pose/reporter.py` | Removes duplicates, sends task reports (JSON) to the scoring server, saves a log. |
| `lite3_explore_pose/terrain.py` | Depth floor check: flat / rubble (bumps) / potholes (holes). Pure numpy. |
| `lite3_explore_pose/locator.py` | Image box + depth + TF → map position (Lab 7, Exercise III-B). Shared helper. |
| `lite3_explore_pose/mission_config.py` | Reads `mission.yaml`; map ↔ arena-cell conversion. |
| `config/explorer.yaml` | Explorer + gait settings. |
| `config/special.yaml` | Crossing speeds, distances, gait codes, safety limits. |
| `config/mission.yaml` | **Competition-day settings**: group ID, servers, arena cells, label lists, time limit. |
| `launch/mission.launch.py` | Starts all of the above with one command. |
| `tools/measure.py` | Hand tests: live distance walked (`dist`, leg odometry) and IMU tilt (`pitch`). |
| `test/` | Unit tests (no ROS needed). |

### Changed OUTSIDE this package
* `~/ros2_ws/src/lite3_launcher/config/pointcloud_to_scan.yaml`: `min_height` `0.1` → `-0.22`
  so Nav2 sees obstacles from ~10 cm tall (short maze walls). Backup next to it:
  `pointcloud_to_scan.yaml.bak_before_lowwalls`. Needs a restart of the robot launch.

### CPU budget (important)
YOLO with default settings used ~2.6 cores and starved the robot's own programs (data bridge,
twist_mux, Fast-LIO): sensor rates collapsed and the robot stopped responding. The detector is
now capped (2 threads, idle threads sleep, OpenCV 1 thread, low priority, 320 px, 2 Hz): ~45 %
of one core. If sensor rates drop (`ros2 topic hz /imu/data` should be ~185), suspect CPU first
(`uptime`: load average should stay well below 10).

### Python packages installed (user account, `~/.local`)
`torch` (CPU), `torchvision`, `ultralytics`, `clip`, `ftfy`, `regex`, `tqdm` for YOLO-World.
Model file: `~/challenge_ws/models/yolov8s-worldv2.pt`.

⚠ **numpy must stay below 2** (ROS Humble breaks otherwise) and **do not keep a newer
`setuptools` in `~/.local`** (it breaks `colcon build`). Install new packages like this:
```bash
printf 'numpy<2\nopencv-python<4.11\n' > /tmp/c.txt
python3 -m pip install --user <package> -c /tmp/c.txt
python3 -m pip uninstall -y setuptools     # only if pip pulled one in; system 59.6 is used
```

## Build and test (no robot needed)

```bash
cd ~/challenge_ws && colcon build --packages-select lite3_explore_pose && source install/setup.bash
cd ~/challenge_ws/lite3_explore_pose && python3 -m pytest -p no:anyio test/
```
Rebuild after changing `.py` files or `config/mission.yaml` (the launch file and the nodes
read the installed copy). `explorer.yaml` / `special.yaml` are also read from the installed copy
when using the launch file, so rebuild after changing them too.

## Run

Terminal 1 — robot + SLAM + Nav2 with the turn hold:
`ros2 launch lite3_explore_pose robot.launch.py slam:=True navigation:=True`

Terminal 2 — our whole stack:
```bash
cd ~/challenge_ws && source install/setup.bash
ros2 launch lite3_explore_pose mission.launch.py                       # SAFE: explorer dry run, no crossings
ros2 launch lite3_explore_pose mission.launch.py live:=true            # robot explores on its own
ros2 launch lite3_explore_pose mission.launch.py live:=true publish_obstacles:=true   # + automatic crossings
```
Options: `use_vlm:=false`, `use_qr:=false`, `use_yolo:=false` switch single parts off.

Terminal 3 — `lite3_auto` before a live run, `lite3_manual` after.
**Stop:** Ctrl+C in Terminal 2, then `lite3_manual`. The joystick always overrides (twist_mux).

### Follow exploration decisions live

With the mission launch running, open another terminal on the computer that ran the launch:

```bash
cd ~/challenge_ws
tail -F log/lite3_explore_pose_decisions.log
```

The launch prints the full path and appends a header for each run. The log includes
the frontier ranking and selected goals, COVER coverage goals and %, HOLDs, Nav2 progress
and cancellation reasons, mission mode changes, and obstacle crossing status. It also works in dry run. Start the launch
with `decision_log:=/path/to/run.log` to keep a separate file for a run. Relative paths
are resolved from the directory where you launch the mission.

### Make sure the camera has seen everything (COVER)

SLAM maps a room from its doorway, so frontiers run out before the forward camera (which
reads the QR codes on the side walls) has looked inside. With `coverage_enabled: true` the
explorer remembers every robot pose, raycasts the camera's view (`camera_fov_deg` 70°,
`camera_range_m` 2.5 m, stopped by walls) on the current map, and when no frontiers are left
switches to **COVER**: it drives to free cells the camera has not seen yet, checking for new
frontiers first. `/exploration/done` is only sent when `coverage_target` (95 %) of the free
cells has been seen, or when no coverage goal is reachable for `done_after_empty_checks` ticks.
The pose history is saved to `~/challenge_ws/coverage/coverage_<time>.npz` every 10 s and on
shutdown; `resume_coverage: true` reloads it (only valid if SLAM keeps the same map frame).

`bounds` limits which cells count. With bounds off (`[0.0]`) every mapped free cell counts,
including the lab beyond the course openings, so the robot may keep going: stop it by hand.

Open `config/maze.rviz` and start the mission in dry run. *Camera coverage* shows free cells
the camera has not seen in purple (seen cells are transparent), the walked trail is cyan,
frontiers are orange / blue and coverage goals green. `/exploration/coverage_pct` gives the
percentage. Drive past a side room without looking in: it stays purple although it is mapped.
Turn the robot to face into it and the purple clears.

### Stand still so the gait change can fire (HOLD)

`obstacle_detector` only sends `stairs_ahead` / `rubble_ahead` / `curtain_ahead` /
`potholes_ahead` after `confirm_frames` frames in a row (3 at 2 Hz = 1.5 s), so a robot that
turns away straight after a goal never changes gait. The explorer therefore stands still
(state HOLD, `/mission/state` stays EXPLORE):

* `goal_dwell_s` (2.5 s) after reaching a frontier or coverage goal, facing the travel direction.
  Keep it above `confirm_frames / rate_hz` + ~1 s.
* `obstacle_hold_s` (4 s) when Nav2 fails, times out or makes no progress while a stairs / rubble /
  curtain / potholes detection is ahead (`/perception/obstacles`, within `obstacle_hold_m`, in the
  last `obstacle_recent_s`). The goal is not blacklisted; if no crossing starts, the normal
  blacklist + recovery follows. Once per goal.

Both are 0 (off) in the code defaults and never run in dry run. The crossing itself still needs
`publish_obstacles:=true`.

### Safety switches (all OFF by default)
| Switch | Where | Off means |
| --- | --- | --- |
| `live:=true` | launch | explorer only shows frontiers and coverage, never drives |
| `publish_obstacles:=true` | launch | detector only LOGS "would send: stairs_ahead" |
| `send_reports: true` | `config/mission.yaml` | reporter only logs to `~/challenge_ws/reports/` |

## Robot test session — step by step

Do the steps in order; each one only needs the ones before it. Write down what happened.

**0. Before going to the robot:** make sure everyone's latest code is in
`~/challenge_ws/lite3_explore_pose` on the group computer, rebuild, run the tests.

**1. Robot up (5 min)** — T1 robot launch. In another terminal:
```bash
ros2 run tf2_ros tf2_echo map base_link        # must print a position
ros2 topic hz /camera/color/image_raw/compressed
```
In RViz add `/scan`: short walls must show as dots, NOT a ring of dots on the empty floor.

**2. Gait in auto mode (5 min)** — `lite3_auto`, then
`ros2 topic pub --once /simple_cmd transfer_interfaces/msg/MotionSimpleCMD "{cmd_code: 0x21010401, type: 0}"`,
then a 2D Goal Pose in RViz. ✅ walks in the rough-ground gait → automatic gait works.
❌ → tell the code owner: the code must switch to manual, change gait, switch back.
Also hand-test a stairs gait (`0x21010407` high-step) on the metal stairs.
Switch back afterwards: `{cmd_code: 0x21010401, type: 0}` (our run gait).

**2b. Safe look-around spin (5 min)** — the explorer starts with a Nav2 spin (collision-checked).
`ros2 action send_goal /spin nav2_msgs/action/Spin "{target_yaw: 6.28}"` once in open space
(✅ full turn) and once ~20 cm from a wall (✅ stops and reports failure, does NOT touch the wall).

**3. Obstacle detector, robot still (10 min)** — `ros2 run lite3_explore_pose obstacle_detector`,
RViz Image display on `/perception/obstacle_image/compressed`. Walk the robot slowly (joystick)
toward the **metal stairs**, **curtain**, **uneven terrain**, from ~3 m.
Note for each: detected? from how far? `would send: ..._ahead` printed?

**4. Record bags (15 min)** — for tuning without the robot (one bag per obstacle):
```bash
ros2 bag record -o stairs /camera/color/image_raw/compressed /camera/color/camera_info \
  /camera/depth/points /tf /tf_static /imu/data
```
Walk toward it from ~3 m, slowly, a few angles. Same for `curtain`, `rubble`, `potholes`.

**5. Crossings by hand (15 min)** — `ros2 run lite3_explore_pose special_obstacles --ros-args
--params-file ~/challenge_ws/lite3_explore_pose/config/special.yaml`. Point the robot straight at
the obstacle, `lite3_auto`, then
`ros2 topic pub --once /special/start std_msgs/msg/String "{data: rubble}"` (or `curtain`, `stairs`).
✅ `/special/done` says `success`. Measure and set in `special.yaml`: `rubble_distance_m`,
`curtain_distance_m`, and on the stairs the IMU pitch (`ros2 topic echo /imu/data`) for
`on_stairs_pitch_deg` / `level_pitch_deg`.

**6. Scene VLM + QR, robot still (10 min)** — `ros2 run lite3_explore_pose reporter`,
`ros2 run lite3_explore_pose vlm_watcher`, `ros2 run lite3_explore_pose qr_reader`.
Trigger a look: `ros2 topic pub --once /perception/look std_msgs/msg/Bool "{data: true}"` while
the camera sees a hazard sign / survivor prop. Show a QR code at 0.5 / 1 / 2 m.
✅ `REPORT [dry_run] ...` lines with a sensible `pos`.

**7. Exploration in the maze (15 min)** — `ros2 launch lite3_explore_pose mission.launch.py live:=true
use_yolo:=false`. ✅ gait switch (2 s), look-around spin that stops before walls, then
walks to openings with no wall hits; reports appear while it walks.

**8. Full run (rest of the session)** — `... live:=true publish_obstacles:=true`. Time it.

## Gait per terrain (switches automatically)

| Ahead | Gait | Detected by | Crossing |
| --- | --- | --- | --- |
| Flat floor | Flat Slow `0x21010300` (start gait) | — | normal Nav2 driving |
| Uneven terrain | RUG General `0x21010401` ✅ hand-tested | **depth floor check** (bumps above floor) | `rubble` |
| Potholes | RUG H-Step `0x21010407` ✅ hand-tested | **depth floor check** (holes below floor) | `potholes` |
| Stairs | **AI mode** (rule) | YOLO-World ("metal staircase") | `stairs` |
| Curtain | no change | YOLO-World from afar; close up: **white colour + something blocking < 0.8 m** (tested live at the curtain) | `curtain` |

After every crossing the robot returns to the flat gait. Codes live in `config/special.yaml`.
The floor check (`lite3_explore_pose/terrain.py`) calibrates the floor height by itself whenever the
floor ahead looks flat; tested on simulated depth data (unit tests + node simulation), not yet
on the real terrain — record the rubble / pothole bags to tune it.

## Stairs = AI mode (organisers' rule, 17/11)

* AI mode (`0x21010528` enter / `0x2101052B` exit) is allowed **only for the stairs**: switch on at
  most 1 cell before, off at most 1 cell after. AI mode on any other terrain = no points for it.
* **No jumps / stunts**, and do not use the AI sub-modes on the controller. Switch AI mode by
  command only. AI teleop is fast — drive slowly; the crossing uses 0.10 m/s via cmd_vel.
* Crossings measure distance with the first working odometry of `/leg_odom2`, `/odom`; with
  none, they walk by time (speed × time) instead of failing.
  Odometry can jump (lidar IMU problems): a crossing only ends when odometry AND speed × time
  (≥ 70 %) agree, or speed × time alone reaches 130 %.
* `special_obstacles` does: stop → AI on → climb straight (tilt limit 40° for stairs) → level
  again → AI off → stand 1.5 s → normal gait → report. Tested in simulation (fake IMU/odom).
* Points for stairs are now mostly about **detection**: a wrong `stairs_ahead` puts the robot in
  AI mode on the wrong terrain. Keep `confirm_frames` high; test the detector on the real stairs.
* AI mode must first be unlocked on our robot by the organisers (access code).

## Gait test by hand (choose the gait for each obstacle)

T1 robot launch. T2: paste once (shortcuts last while the terminal is open):
```bash
cd ~/challenge_ws && source install/setup.bash
g() { ros2 topic pub --once /simple_cmd transfer_interfaces/msg/MotionSimpleCMD "{cmd_code: $1, type: 0}" > /dev/null && echo "gait $1 sent"; }
alias gait_flat='g 0x21010300'     # flat, slow (factory default)
alias gait_rough='g 0x21010401'    # rough ground, general  (our run gait)
alias gait_grip='g 0x21010402'     # rough ground, grip
alias gait_high='g 0x21010407'     # rough ground, high steps
alias ai_on='g 0x21010528'         # AI mode ON  - STAIRS ONLY (organisers' rule)
alias ai_off='g 0x2101052B'        # AI mode OFF - then gait_rough
alias pitch='python3 ~/challenge_ws/lite3_explore_pose/tools/measure.py pitch'   # IMU tilt (stairs)
alias dist='python3 ~/challenge_ws/lite3_explore_pose/tools/measure.py dist'     # distance walked (Enter = reset)
lite3_manual
```
For each gait: stop ~1 m before the obstacle → gait → wait 2 s → walk slowly and straight
across with the joystick → 2-3 tries each.
No tape measure needed: in a 3rd terminal (paste the block there too) run `dist`, press Enter
at the start of the obstacle, read the distance once the robot is fully past. `pitch` shows the
tilt live plus the biggest tilt so far.

| Obstacle | Try in order | Known so far |
| --- | --- | --- |
| Uneven terrain | rough → high → grip | rough ✅ |
| Potholes | high → grip | rough ❌ |
| Stairs up / down | `ai_on` ~1 cell before → slow joystick → `ai_off` → `gait_rough` | AI mode = organisers' rule |
| Curtain | flat → rough | — |

Record: gait, successes/tries, what went wrong, distance from obstacle start until fully past.
Results go into `config/special.yaml`: `stairs_gait_code`, `rubble_gait_code`,
`on_stairs_pitch_deg`, `level_pitch_deg`, `rubble_distance_m`, `curtain_distance_m`
(codes in decimal: 0x21010401 = 553714689, 0x21010402 = 553714690, 0x21010407 = 553714695).
Finish with `gait_rough`. Take the drone off for these tests.

## Talking to the mission controller (topics)

| Topic | Type | Use |
| --- | --- | --- |
| `/perception/event` | `std_msgs/String` | `stairs_ahead`, `rubble_ahead`, `curtain_ahead` (cross), `person`, `qr`, `new_area` (stop & look) |
| `/perception/look` / `look_done` | `std_msgs/Bool` | controller asks the VLM for a careful look / answer done |
| `/perception/room`, `/perception/hazard` | `std_msgs/String` | latest room type / hazard label |
| `/perception/found` | `std_msgs/String` (JSON) | item for the reporter |
| `/special/start`, `/special/done` | `std_msgs/String` | start a crossing / `success` or `failed` |
| `/exploration/enable` | `std_msgs/Bool` | pause / resume the explorer |
| `/mission/tasks_done` | `std_msgs/Bool` | all expected tasks reported → go to final cell |
| `/mission/state` | `std_msgs/String` | EXPLORE, INVESTIGATE, TRAVERSE, GO_TO_EXIT, LAUNCH, DONE |

## Still missing / known limits

* **Uneven terrain / potholes**: depth floor check written and simulation-tested, NOT yet tested
  on the real terrain (YOLO-World missed the wooden slats). Thresholds in `terrain.py`.
* **Curtain**: the lidar sees it as a wall, so the explorer will not plan through it and may never
  drive up to it. Detection only fires when the robot is within 1.2 m and facing it.
* **QR codes are on side walls**; the camera looks forward. COVER drives to what the camera has
  not seen and the 360° goal spin turns the camera, but the camera is never aimed at a wall on purpose.
* **Arena boundary**: the explorer may try to leave through the final-cell opening (bounds off:
  frontiers and camera coverage include the lab).
* **Curtain / stairs alignment**: the crossing walks straight from the current heading —
  the robot must already face the doorway / stairs.
* **Stairs**: AI mode wired in (simulation-tested only); pitch thresholds 8° / 3° are defaults —
  check on the real stairs. Waiting for organisers to unlock AI mode on our robot.
* **mission.yaml TODOs**: arena cells, start heading, organisers' label lists, task IDs, time limit,
  and `group05_XXXX` PIN before `send_reports: true`. Map origin = robot start pose, so the robot
  must start in the same cell, facing the same way, every run.
* **Scoring by ROS 2 topics** may replace the HTTP server (organisers' message) → only
  `reporter.py`'s sending part changes.
* **Drone launch**: `/launch_drone` is only a Bool; alignment to the target is not done.
* Distance for frontier choice is straight-line, not Nav2 path length.
