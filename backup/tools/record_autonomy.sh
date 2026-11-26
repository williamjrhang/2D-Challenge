#!/bin/bash
# Record an autonomous test with everything needed to debug localization / readiness stalls.
#   bash ~/challenge_ws/lite3_explore_pose/tools/record_autonomy.sh     start AFTER the robot launch
#   Ctrl+C                                                          after the run
# Saves to ~/challenge_ws/runs/autonomy_<date_time>/bag   (~4 MB/s, mostly /livox/lidar:
# 15 min ~ 3.5 GB - check `df -h ~` first).
# /livox/lidar here is a 2nd copy over the robot link (FAST-LIO has the 1st). Measured OK on
# 25/11 together with FAST-LIO, but do not add more lidar/camera subscribers on top.
source /opt/ros/humble/setup.bash
source ~/lite3_ws/install/setup.bash          # livox_ros_driver2/CustomMsg type for /livox/lidar
source ~/ros2_ws/install/setup.bash
source ~/challenge_ws/install/setup.bash
# one NIC for DDS, like the launch files (else the bridge sends the recorder everything twice)
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$HOME/challenge_ws/install/lite3_explore_pose/share/lite3_explore_pose/config/fastdds_one_nic.xml}

dir=${1:-~/challenge_ws/runs/autonomy_$(date +%Y%m%d_%H%M%S)}   # optional: run folder
mkdir -p "$dir"
echo "Recording to $dir  -- press Ctrl+C after the run."
exec ros2 bag record -o "$dir/bag" \
    /livox/lidar /livox/imu /imu/data /fastlio/odometry /odom /tf /tf_static /scan \
    /map /map_metadata \
    /exploration/localization_ready /exploration/status /exploration/enable \
    /exploration/done /exploration/frontiers /exploration/cells /exploration/coverage \
    /exploration/trail \
    /mission/state /mission/tasks_done /mission/found_markers /special/state \
    /perception/scene /perception/room /perception/found /perception/qr /perception/hazard \
    /perception/event /perception/look /perception/look_done \
    /perception/obstacles /perception/wall_ahead \
    /navigate_to_pose/_action/status /navigate_to_pose/_action/feedback /plan \
    /local_costmap/costmap \
    /nav_cmd_vel_raw /navigation/turn_hold_active /nav_cmd_vel /cmd_vel /joy_cmd_vel \
    /simple_cmd \
    /rosout
