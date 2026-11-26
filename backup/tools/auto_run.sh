#!/bin/bash
# Full autonomous run with GO/NO-GO gates, RViz and a ROS bag of everything.
#
#   bash ~/challenge_ws/lite3_explore_pose/tools/auto_run.sh                 # grid explore
#   bash ~/challenge_ws/lite3_explore_pose/tools/auto_run.sh publish_obstacles:=true
#   (extra arguments go to the LIVE mission launch)
#
# Steps:  1 robot launch (no motion) -> 2 odometry check -> 3 RViz + recording
#         -> 4 dry run with full perception + readiness check (no motion)
#         -> 5 you type YES (operator at the robot with the stop control, arena clear)
#         -> 6 LIVE mission.  Ctrl+C at any time stops everything (mission first) and saves.
# The robot's own stop control is the emergency stop; Ctrl+C here is not.
# Everything lands in ~/challenge_ws/runs/auto_<date_time>/  (bag, logs, health.log, map).

source /opt/ros/humble/setup.bash
source ~/lite3_ws/install/setup.bash
source ~/ros2_ws/install/setup.bash
source ~/challenge_ws/install/setup.bash
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-5}
# Same one-NIC Fast DDS profile as robot.launch.py / mission.launch.py / record_autonomy.sh:
# processes WITHOUT it cannot see the robot launch's nodes (26/11: preflight said 'no
# /fastlio/odometry' while FAST-LIO ran at 10 Hz). Profile file unchanged.
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$HOME/challenge_ws/install/lite3_explore_pose/share/lite3_explore_pose/config/fastdds_one_nic.xml}

TOOLS=~/challenge_ws/lite3_explore_pose/tools
dir=~/challenge_ws/runs/auto_$(date +%Y%m%d_%H%M%S)
mkdir -p "$dir"
cd ~/challenge_ws                       # mission decision log path is relative to this
ROBOT= REC= MISSION= RVIZ= HEALTH=

say() { echo -e "\n=== [$(date +%H:%M:%S)] $*"; }

# start VAR NAME CMD...: run CMD in the background in its own session (Ctrl+C in this
# terminal does not hit it; stop() ends it in order), log to $dir/NAME.log, PID -> $VAR.
# Background jobs of a script start with SIGINT ignored, and ros2 launch / rclpy then
# ignore kill -INT: reset SIGINT to default before exec.
start() {
    local var=$1 name=$2; shift 2
    setsid python3 -c 'import os, signal, sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])' "$@" > "$dir/$name.log" 2>&1 < /dev/null &
    printf -v "$var" '%s' "$!"
}

stop() {                                 # stop PID (whole process group), INT then TERM
    local pid=$1
    [ -z "$pid" ] && return
    kill -0 "$pid" 2>/dev/null || return
    kill -INT -- -"$pid" 2>/dev/null
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || return; sleep 0.5; done
    kill -TERM -- -"$pid" 2>/dev/null; sleep 3
    kill -KILL -- -"$pid" 2>/dev/null
}

finish() {
    trap - INT TERM
    say "Stopping: mission first, then health monitor, recording, robot launch"
    stop "$MISSION"; stop "$HEALTH"; sleep 2; stop "$REC"
    timeout 15 ros2 run nav2_map_server map_saver_cli -f "$dir/map" > /dev/null 2>&1 \
        && echo "Map saved: $dir/map.pgm" || echo "Map not saved"
    stop "$ROBOT"
    [ -n "$RVIZ" ] && stop "$RVIZ"
    say "Run folder: $dir"
    echo "  bag:        $dir/bag"
    echo "  health:     $dir/health.log   (GO/NO-GO every ~20 s during the live run)"
    echo "  mission:    $dir/mission_live.log    robot launch: $dir/robot.log"
    exit "${1:-0}"
}
trap 'finish 1' INT TERM

nogo() {
    say "NO-GO: $*"
    echo "Nothing will move. Details above and in $dir/*.log"
    finish 1
}

# ---- 0. nothing already running (two robot launches = two FAST-LIOs fighting over TF)
if pgrep -f 'lite3_launcher lite3.launch.py|lite3_explore_pose robot.launch.py|lite3_explore_pose mission.launch.py|fastlio_mapping' > /dev/null; then
    echo "A robot launch / mission / FAST-LIO is already running on this PC:"
    pgrep -af 'lite3_launcher lite3.launch.py|lite3_explore_pose robot.launch.py|lite3_explore_pose mission.launch.py|fastlio_mapping'
    echo "Stop it first (or ask who started it). Nothing started."
    exit 1
fi
df -h ~ | tail -1 | awk '{print "Disk free: "$4"  (the bag needs ~4 MB/s, 15 min ~ 3.5 GB)"}'

# ---- 1. robot + SLAM + Nav2 (no motion)
say "1/6 Robot launch (FAST-LIO, SLAM, Nav2) - the robot does not move"
# robot.launch.py = the same lite3_launcher bringup + turn_hold (3 s still after a Nav2
# turn; it was missing in auto runs before 26/11). relay:=false: one machine, no desktop.
start ROBOT robot ros2 launch lite3_explore_pose robot.launch.py slam:=True navigation:=True relay:=false
sleep 25
grep -q 'max_lidar_backlog 3' "$dir/robot.log" \
    || echo "WARNING: FAST-LIO did not log 'max_lidar_backlog 3' - old build or config?"

# ---- 2. odometry fresh?
say "2/6 Odometry check"
ok=
for try in 1 2 3; do
    if python3 $TOOLS/odom_preflight.py --seconds 15 | tee -a "$dir/preflight.log"; then ok=1; break; fi
    echo "(attempt $try/3 failed, retrying in 10 s)"; sleep 10
done
[ -n "$ok" ] || nogo "odometry not fresh with only the robot launch running.
  /livox/imu < 150 Hz -> robot link (other lidar/camera subscribers? other groups?)
  IMU ok but /fastlio/odometry late -> grep 'Scan backlog' $dir/robot.log, check CPU (top)"

# ---- 3. RViz + recording
say "3/6 RViz + recording"
if pgrep -x rviz2 > /dev/null; then
    echo "RViz already open - using it (config: lite3_explore_pose/config/maze.rviz)"
else
    start RVIZ rviz rviz2 -d ~/challenge_ws/lite3_explore_pose/config/maze.rviz
fi
start REC record bash $TOOLS/record_autonomy.sh "$dir"
sleep 5
kill -0 "$REC" 2>/dev/null || nogo "recording did not start, see $dir/record.log"

# ---- 4. dry run with full perception (YOLO/QR/VLM load, no motion) + readiness
say "4/6 Dry run with full perception (does not move) - 60 s readiness check"
start MISSION mission_dry ros2 launch lite3_explore_pose mission.launch.py grid_explore:=true
sleep 30                                 # YOLO / VLM start-up
python3 $TOOLS/odom_preflight.py --ready --seconds 60 | tee -a "$dir/preflight.log"
dry_ok=${PIPESTATUS[0]}
stop "$MISSION"; MISSION=
[ "$dry_ok" = 0 ] || nogo "dry run (full perception) failed the check - see table above"

# ---- 5. operator confirmation
say "5/6 GO. Ready for the LIVE run."
echo "The robot WILL MOVE on its own. Confirm:"
echo "  - an operator is at the robot with the stop control"
echo "  - the arena is clear and no other group is using the robot"
read -r -p "Type YES to start the live run (anything else stops): " answer
[ "$answer" = "YES" ] || { say "Not confirmed - no live run."; finish 0; }

# ---- 6. live
say "6/6 LIVE mission: ros2 launch lite3_explore_pose mission.launch.py live:=true grid_explore:=true $*"
start MISSION mission_live ros2 launch lite3_explore_pose mission.launch.py live:=true grid_explore:=true "$@"
start HEALTH health_loop bash -c "while true; do echo \"--- \$(date +%H:%M:%S)\";
    python3 $TOOLS/odom_preflight.py --ready --seconds 10 | grep -E 'FAIL|GO'; sleep 10; done"
ln -sf health_loop.log "$dir/health.log"
echo "Watch RViz. Problems show up in $dir/health.log (tail -f) and the mission log."
echo "Press Ctrl+C here to end the run and save everything."
while kill -0 "$MISSION" 2>/dev/null; do sleep 2; done
say "Mission launch exited"
finish 0
