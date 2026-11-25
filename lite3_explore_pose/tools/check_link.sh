#!/bin/bash
# Two-machine link check (issue 9). Run on BOTH machines, each pointing at the other:
#   on the robot:    bash ~/challenge_ws/lite3_explore_pose/tools/check_link.sh 192.168.10.105
#   on the desktop:  bash ~/challenge_ws/lite3_explore_pose/tools/check_link.sh <robot ip>
# Checks: ROS env (same ROS_DOMAIN_ID on both!), ping, clock sync, topic rates of
# /livox/imu, /map, /tf and the throttled image, link_relay status. Read-only: it changes
# nothing and subscribes only to small topics (never the depth cloud or /livox/lidar).
PEER=${1:?usage: check_link.sh <peer ip or hostname>}
TOOLS=$(dirname "$(readlink -f "$0")")
source /opt/ros/humble/setup.bash
for ws in ~/lite3_ws ~/ros2_ws ~/challenge_ws; do
    [ -f $ws/install/setup.bash ] && source $ws/install/setup.bash
done
FAIL=0
bad() { echo "  !! $*"; FAIL=1; }

echo "=== 1. ROS environment on $(hostname)"
echo "  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset = 0>}  RMW=${RMW_IMPLEMENTATION:-<default rmw_fastrtps_cpp>}"
echo "  ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}  FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-<none>}"
[ "${ROS_LOCALHOST_ONLY:-0}" = "1" ] && bad "ROS_LOCALHOST_ONLY=1: the other machine cannot see this one"
echo "  -> compare ROS_DOMAIN_ID and RMW with the other machine: they must be equal"

echo "=== 2. ping $PEER"
if out=$(ping -c 5 -q -W 1 "$PEER" 2>&1); then
    echo "$out" | tail -2 | sed 's/^/  /'
    loss=$(echo "$out" | grep -oP '\d+(?=% packet loss)')
    [ "${loss:-100}" -gt 0 ] && bad "packet loss ${loss}%"
else
    bad "no reply from $PEER"
fi

echo "=== 3. clock"
if command -v chronyc > /dev/null; then
    chronyc tracking 2>/dev/null | grep -E "System time|Leap status" | sed 's/^/  /'
fi
timedatectl show -p NTPSynchronized --value 2>/dev/null | sed 's/^/  NTPSynchronized=/'
echo "  stamp age of robot topics (desktop: clock offset + latency; must be < 0.2 s):"
python3 "$TOOLS/stamp_age.py" --seconds 5 || bad "clock offset / latency > 0.2 s (sync clocks with chrony or NTP)"

echo "=== 4. topic rates (6 s)"
python3 "$TOOLS/diag_rates.py" --seconds 6 --period 6 --out /tmp/check_link_rates.csv \
    --topics /livox/imu /map /tf /camera/color/image_throttled/compressed | grep -v "^diag_rates\|^saved" | sed 's/^/  /'
grep -q "LOW" <(python3 - <<'PY'
import csv
rows = list(csv.reader(open('/tmp/check_link_rates.csv')))
h, v = rows[0], rows[-1]
lim = {'/livox/imu Hz': 150, '/tf Hz': 10, '/camera/color/image_throttled/compressed Hz': 3}
for k, lo in lim.items():
    if k in h and float(v[h.index(k)]) < lo:
        print('LOW', k)
PY
) && bad "a rate is below its minimum (see LOW above): /livox/imu >= 150, /tf >= 10, image >= 3 Hz"
echo "  (/map: ~0.5 Hz while the map changes; 0 is normal when the robot stands still)"

echo "=== 5. link_relay status (robot side)"
st=$(timeout 4 ros2 topic echo --once --qos-durability transient_local /link_relay/status \
    std_msgs/msg/String 2>/dev/null | grep data)
[ -n "$st" ] && echo "  $st" || bad "no /link_relay/status (link_relay not running on the robot / not visible)"
echo "$st" | grep -q degraded && bad "link_relay is DEGRADED: the robot's IMU / lidar rate is low"

echo
[ $FAIL = 0 ] && echo "=== LINK OK" || echo "=== LINK PROBLEMS (see !! above)"
exit $FAIL
