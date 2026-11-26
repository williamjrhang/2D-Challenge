# Lite3 camera recorder

Record the same compressed color feed the obstacle detector uses while you drive the
Lite3 by hand. The robot camera driver should be running. The recorder subscribes
directly to the camera feed, so it can run by itself during manual driving.

## Build and record

In a terminal with ROS 2 sourced, build the recorder and start it on the machine
receiving the camera feed. Use the same `ROS_DOMAIN_ID` as the camera publisher.

```bash
cd ~/challenge_ws
colcon build --packages-select lite3_camera_recorder
source install/setup.bash
ros2 run lite3_camera_recorder record_camera
```

The default topic is `/camera/color/image_raw/compressed`. On the desktop that uses
the Lite3 link relay, select the same throttled topic as the desktop obstacle detector.

```bash
ros2 run lite3_camera_recorder record_camera --ros-args \
  -p image_topic:=/camera/color/image_throttled/compressed -p playback_fps:=5.0
```

Wait for `Recording to ...`, then drive the robot. Press **Ctrl+C** to finalize the MP4
and print its path and frame count. Every run with camera images gets a unique file
under `~/challenge_ws/frames/recordings/`. Set `-p output_dir:=/path/to/recordings` to
store it elsewhere. `playback_fps` sets the video's playback rate (default `30.0`).
The recorder writes frames as they arrive, so playback speed can vary from real time
when the camera or relay changes its publishing rate. Start a new recording if you
change the camera resolution.

## Export frames for labeling

Use the MP4 path printed by the recorder. For example, export the first frame and then
every 15th frame as PNGs.

```bash
ros2 run lite3_camera_recorder extract_frames /path/to/recording.mp4 --every 15
```

The PNGs go into a new `<video_name>_frames/` folder next to the MP4. Omit `--every`
to export all frames, or set `--output-dir /path/to/new_folder` for another export
from the same recording.
