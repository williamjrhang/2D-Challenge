# lite3_wall_pins

Observe-only ROS 2 Humble node that pins **wall-mounted QR codes** (and, experimentally,
**printed pictures of people**) on the SLAM map while you drive the Lite3 by hand. Pins are
shown in RViz2 and saved to JSON so they reload for the same map after a restart.

**Safety:** the node only subscribes to the camera, depth, camera_info and TF, and only
publishes under `/wall_pins/*`. It has no `cmd_vel` publisher, no Nav2 action client, no
service clients and no control-mode or gait topics. `test/test_observe_only.py` enforces
this. The node does not depend on `lite3_explore` and can run with or without the
exploration mission.

## Build

```bash
cd ~/challenge_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select lite3_wall_pins
source install/setup.bash
```

The launch file reads the **installed** `config/wall_pins.yaml`. Rebuild after editing it,
or pass `params_file:=<path>`.

## Run

SLAM (publishing `map -> ... -> base_link`) and the camera must already be running.

```bash
# QR pins only (default), RViz with the preset config:
ros2 launch lite3_wall_pins wall_pins.launch.py rviz:=true

# One pins file per SLAM map. Use a new map_id whenever you start a new map:
ros2 launch lite3_wall_pins wall_pins.launch.py map_id:=arena_run2 rviz:=true

# EXPERIMENTAL printed-picture pins (YOLO-World, CPU, ~15 s model load):
ros2 launch lite3_wall_pins wall_pins.launch.py enable_pictures:=true rviz:=true
```

Launch arguments: `map_id` (default `default`), `pins_file` (`''` means
`~/.ros/lite3_wall_pins/<map_id>.json`), `enable_pictures`, `rviz`, `use_sim_time`,
`params_file`.

### RViz

If you use your own RViz window: **Add → By topic → `/wall_pins/markers` → MarkerArray**,
with Fixed Frame `map`. The topic is latched (Transient Local), so pins also show up in an
RViz started later.

| Marker namespace  | Meaning                                                       |
|-------------------|---------------------------------------------------------------|
| `wall_pins`       | sphere at the target's position on the wall (blue = QR, orange = picture) |
| `wall_pin_poles`  | line from the floor to the pin (easy to see in top-down view) |
| `wall_pin_labels` | `qr_1  QR: <text>  (xN)`, where N = number of observations    |
| `view_poses`      | green arrow: where the robot stood and which way it faced     |
| `view_rays`       | line from the viewing pose to the pin                         |

`/wall_pins/debug_image` (sensor_msgs/Image) shows the camera frame with the detections
drawn on it: blue = QR, orange = picture, red = seen but not pinned. It is built only while
something is subscribed to it.
`/wall_pins/events` (std_msgs/String) publishes one JSON message per **new** pin.

## How a pin is made

1. **QR detection:** OpenCV `QRCodeDetectorAruco` + `QRCodeDetector` (no AI). For each
   decoded code the node keeps the text and the four image corners. Small codes that are
   found but not decoded are retried on an upscaled crop.
2. **Synchronisation:** each RGB frame is paired with the `/camera/depth/points` cloud
   whose stamp is within `max_sync_dt_s` (50 ms). The cloud is moved into the colour
   camera frame **at the image time** through `map`, which compensates for robot motion
   between the two stamps.
3. **Depth in the region:** depth points are projected with the `camera_info` K. The
   detection polygon is undistorted with D and shrunk to its inner 80%. The node keeps
   points inside the polygon and takes the median of the dominant depth layer. It then
   requires enough points, enough coverage, and ≥ 60% of them on one surface.
4. **Map position:** the surface point is transformed into `map` with the camera→map TF at
   the image stamp. The robot pose (`base_link`) at that stamp gives the viewing pose.
5. **Skip, never guess:** the pin is skipped and the reason logged (throttled) if any of
   these hold: no camera_info yet; no depth frame within the sync window; TF unavailable at
   the image time; image older than `max_image_age_s` or stamped in the future (clock
   offset); too few or inconsistent depth points; target outside 0.2–4.0 m. There is no
   fallback that places a pin in front of the robot.

### Deduplication and persistence

Before creating a pin, the node checks the saved pins:

* **QR:** same payload **and** within `merge_radius_m` (0.6 m) means the same pin. The
  observation refines its position (running mean) and increments the `xN` counter.
  **Different payloads are never merged**, however close they are. The same payload seen
  farther away becomes a second pin, and the node logs a warning (second copy, or map
  drift).
* **Picture:** any picture pin within `picture_merge_radius_m` (0.8 m) is the same pin.

Pins are written to `~/.ros/lite3_wall_pins/<map_id>.json` as soon as they are created,
then every 5 s if changed, and on shutdown. Writes are atomic (temporary file, then
rename). On start the file is reloaded. A file whose `map_id` differs is refused rather
than mixed in. Each pin stores the wall position and the viewing pose **separately**:

```json
{
  "id": "qr_1", "kind": "qr", "payload": "ROOM-12", "label": "QR: ROOM-12",
  "wall":      {"x": 3.41, "y": -1.02, "z": 0.62},
  "view_pose": {"x": 2.55, "y": -0.98, "yaw": -0.03, "yaw_to_target": -0.05, "range_m": 0.88},
  "observations": 14, "first_seen": 1794990000.1, "last_seen": 1794990042.7,
  "experimental": false,
  "extra": {"size_px": 61.2, "size_m": 0.121, "range_m": 0.88, "depth_points": 412, "sync_ms": 12.3}
}
```

`view_pose` is the pose from the **closest** observation. To return to a target, drive to
`(x, y)` and turn to `yaw` (the heading you had). `yaw_to_target` is the heading that
points the robot straight at the pin. `view_pose` is `null` if `map → base_link` was not
available.

## Printed pictures of people (EXPERIMENTAL, off by default)

Model: `~/challenge_ws/models/yolov8s-worldv2.pt` (YOLO-World v2) with the local CLIP text
encoder `models/weights/clip/ViT-B-32.pt`. Runs on the CPU with ultralytics 8.4 and torch
2.14 (CPU). Nothing is downloaded. `torch` and `ultralytics` are imported only when
`enable_pictures:=true`.

**What the model can actually distinguish:** YOLO-World is an open-vocabulary detector. It
scores image regions against text prompts and has **no trained "printed photo vs. real
person" class**. The prompt `person` fires on real people **and** on photos of people. To
reduce false pins, a region only becomes a picture pin if **both** of these hold:

1. a *picture* prompt (`"poster of a person on a wall"`, `"printed photo of a person"`)
   scores ≥ `picture_conf` (0.30). The plain `person` score is only logged, never pinned;
2. the depth inside the box is **flat and vertical**: the centre of the box lies on the
   wall plane fitted around it (`picture_require_flat`). A real person standing in front of
   a wall fails this.

Known blind spots: a person pressed flat against a wall, TV screens, mirrors, posters of
non-people, and photos too small for the model at `picture_imgsz` 480.

Validation status: on an offline synthetic image (a printed photo on a plain wall) the
picture prompts scored 0.38–0.94, and on stock photos of real people they scored ≤ 0.06.
On one live frame from the robot (18 Nov) the model loaded locally in ~15 s, inference took
~0.8 s per frame, a real person in view scored `person` 0.80, and **no** picture prompt
fired, which is the intended behaviour. It has **not** been tested on a real printed
picture with the robot, so pins from this feature are marked `experimental` (orange,
semi-transparent). QR pinning does not depend on it: if the model fails to load, the node
logs an error and continues with QR only.

## Manual test on the robot

1. Start SLAM and the camera. Then run
   `ros2 launch lite3_wall_pins wall_pins.launch.py map_id:=walltest rviz:=true`.
   The log should print `wall_pins ready (observe only)` and the pins file path.
2. Drive to about 0.5–1.5 m from a wall with a printed QR code and face it. In the
   *Pin detections* image the code is outlined in blue. The log prints
   `NEW PIN qr_1 QR "<text>" at (x, y, z) m ...`.
3. In RViz, check that the blue sphere sits **on the wall line** of the map (not in front
   of the robot), at roughly the height of the code. Also check that the green arrow sits
   where the robot stood.
4. Drive a few metres away, then approach the same code again, from another angle if
   possible. The label's `(xN)` counter goes up, and **there is still only one pin**.
   Run `cat ~/.ros/lite3_wall_pins/walltest.json` and check that it has one entry for
   that payload.
5. Put a second, different QR code next to the first. It must get its own pin even if it
   is closer than 0.6 m.
6. Stop the node (Ctrl+C) and start it again with the same `map_id`. The log should say
   `loaded N pins`, and the pins should reappear in RViz.
7. Negative check: cover the depth camera, or stop SLAM. The QR code is still outlined (in
   red), and the log says why it is **not** pinned (no depth / TF unavailable). No pin
   appears.

If nothing gets pinned and the log says `image ... in the future` or `... s old`, the robot
and PC clocks are out of sync. Check chrony/NTP.

## Tests

```bash
cd ~/challenge_ws/lite3_wall_pins
source /opt/ros/humble/setup.bash
PYTHONPATH=$PWD:$PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q test
```

(`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` works around a broken `anyio` pytest plugin in
`~/.local` on this PC.) The tests cover the geometry and pin store (dedup, distinct
payloads, JSON round-trip, map_id check), QR decoding, the picture/person combination
logic, the observe-only guarantee, and the real node fed with a synthetic QR, wall cloud
and TF. In that last test the pin lands on the wall at the right depth and is deduplicated
over repeated frames, and missing depth, missing TF or future stamps produce no pin. The
node test runs on its own DDS domain and never talks to the robot.

## Verification status

| Behaviour | Status |
|---|---|
| Build, install, launch; latched MarkerArray in `map` | verified on this PC against the live robot topics |
| QR decode + 3D localisation + dedup + skip-on-missing-depth/TF/stale | verified in unit/node tests (synthetic data) |
| JSON save/reload, map_id guard, viewing pose stored separately | verified in tests |
| Never commands the robot | verified statically (test) and by design |
| Pin lands on the real wall, one pin after re-approach | **needs the manual test on the robot** |
| Real RGB/depth sync and TF latency during motion | **needs the robot** |
| Printed-picture detection | **EXPERIMENTAL**; loads and runs locally, not validated on a real printed picture |
