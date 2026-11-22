#!/bin/bash
# Record one run (what RViz shows, no camera) and save the map picture at the end.
#   bash ~/challenge_ws/lite3_explore/tools/save_run.sh      start BEFORE the mission
#   Ctrl+C                                                    after the run
# Saves to ~/challenge_ws/runs/<date_time>/  (bag folder + map.pgm / map.yaml)
# Replay later with the robot OFF:  bash ~/challenge_ws/lite3_explore/tools/save_run.sh play <folder>
source /opt/ros/humble/setup.bash

if [ "$1" = "play" ]; then
    dir="${2:-$(ls -td ~/challenge_ws/runs/*/ | head -1)}"
    export ROS_DOMAIN_ID=42          # separate from the live robot (domain 5)
    echo "Replaying $dir (domain 42). RViz opens on the Remote Desktop. Ctrl+C to stop."
    DISPLAY=:10 rviz2 -d ~/challenge_ws/lite3_explore/config/maze.rviz >/dev/null 2>&1 &
    rviz=$!
    ros2 bag play "$dir/bag"
    kill $rviz 2>/dev/null
    exit 0
fi

dir=~/challenge_ws/runs/$(date +%Y%m%d_%H%M%S)
mkdir -p "$dir"
echo "Recording to $dir  -- press Ctrl+C after the run."
ros2 bag record -o "$dir/bag" /map /tf /tf_static /scan /odom /plan \
    /exploration/frontiers /exploration/coverage /exploration/trail /exploration/status \
    /mission/state /special/state /perception/event /perception/obstacles >/dev/null 2>&1 &
rec=$!
trap 'echo; echo "Saving the map..."; kill -INT $rec; wait $rec 2>/dev/null;
      timeout 15 ros2 run nav2_map_server map_saver_cli -f "$dir/map" >/dev/null 2>&1 \
        && echo "Map picture: $dir/map.pgm" || echo "Map not saved (is the robot launch still running?)";
      echo "Saved run: $dir"; exit 0' INT
wait $rec
