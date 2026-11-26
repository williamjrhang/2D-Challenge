#!/usr/bin/env python3
"""GO / NO-GO check that FAST-LIO odometry is fresh before an autonomous run.

    python3 ~/challenge_ws/lite3_explore_pose/tools/odom_preflight.py            # robot launch only
    python3 ~/challenge_ws/lite3_explore_pose/tools/odom_preflight.py --ready    # + explorer running

Watches for --seconds (default 15) and exits 0 only if every check passes.
Ages are (PC wall clock - header stamp); the Livox clock matches the PC clock (25/11),
so a large age means the data is late. /imu/data carries robot time-since-boot, so only
its rate is checked. Does NOT subscribe to /livox/lidar or the camera: every extra
subscriber of a heavy topic is one more copy over the robot link (that is what starved
/livox/imu on 25/11). Lidar health shows up as FAST-LIO's output rate and age.
"""
import argparse
import math
import sys
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

# topic: (min Hz, max age s or None)   -- thresholds from healthy runs on 25/11
CHECKS = {
    '/livox/imu': (150.0, 0.5),           # healthy: 200 Hz, ~0.01 s (single 0.3 s blips seen 25/11 are harmless)
    '/imu/data': (150.0, None),           # healthy: ~200 Hz (stamps are robot uptime)
    '/fastlio/odometry': (8.0, 0.5),      # healthy: 10 Hz, ~0.08 s; failed run: 2.6-24 s
    '/odom': (8.0, 0.5),
    'TF odom->base_link': (8.0, 0.5),
}
HEAVY = ['/camera/depth/points']          # must have no permanent subscriber


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=15.0)
    ap.add_argument('--ready', action='store_true',
                    help='also require /exploration/localization_ready to be true')
    ap.add_argument('--moving', action='store_true',
                    help='robot is driving: skip the standing-still drift check')
    a = ap.parse_args()
    positions = []                      # FAST-LIO position while the robot stands still

    rclpy.init()
    node = rclpy.create_node('odom_preflight')
    data = {k: [] for k in CHECKS}
    ready = []
    heavy_seen = {t: 0 for t in HEAVY}
    samples = 0

    def rec(key, stamp):
        now = time.time()
        data[key].append((now, now - (stamp.sec + stamp.nanosec * 1e-9)))

    be = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(Imu, '/livox/imu', lambda m: rec('/livox/imu', m.header.stamp), be)
    node.create_subscription(Imu, '/imu/data', lambda m: rec('/imu/data', m.header.stamp), be)
    for t in ('/fastlio/odometry', '/odom'):
        node.create_subscription(Odometry, t, lambda m, t=t: rec(t, m.header.stamp), be)

    def on_lio(m):
        p = m.pose.pose.position
        positions.append((p.x, p.y, p.z))
    node.create_subscription(Odometry, '/fastlio/odometry', on_lio, be)

    def on_tf(m):
        for x in m.transforms:
            if x.header.frame_id == 'odom' and x.child_frame_id == 'base_link':
                rec('TF odom->base_link', x.header.stamp)
    node.create_subscription(TFMessage, '/tf', on_tf, be)
    node.create_subscription(Bool, '/exploration/localization_ready',
                             lambda m: ready.append((time.time(), m.data)), 10)

    end = time.time() + a.seconds
    next_graph = 0.0
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
        if time.time() > next_graph:          # sample the graph ~1/s
            next_graph = time.time() + 1.0
            samples += 1
            for t in HEAVY:
                if node.get_subscriptions_info_by_topic(t):
                    heavy_seen[t] += 1

    ok = True
    print(f'{"input":22s} {"Hz":>6s} {"age med":>8s} {"age max":>8s}  result')
    for key, (min_hz, max_age) in CHECKS.items():
        v = data[key]
        if len(v) < 2:
            print(f'{key:22s} {"-":>6s} {"-":>8s} {"-":>8s}  FAIL: no messages')
            ok = False
            continue
        hz = (len(v) - 1) / (v[-1][0] - v[0][0])
        ages = sorted(x[1] for x in v)
        med, worst = ages[len(ages) // 2], ages[-1]
        bad = []
        if hz < min_hz:
            bad.append(f'rate < {min_hz:g} Hz')
        if max_age is not None and worst > max_age:
            bad.append(f'age > {max_age:g} s')
        if max_age is None:
            med_s = worst_s = '   (n/a)'
        else:
            med_s, worst_s = f'{med:8.3f}', f'{worst:8.3f}'
        print(f'{key:22s} {hz:6.1f} {med_s} {worst_s}  {"ok" if not bad else "FAIL: " + ", ".join(bad)}')
        ok &= not bad

    for t, n in heavy_seen.items():
        perm = samples and n >= max(3, samples * 0.6)
        print(f'{t:22s} subscribed in {n}/{samples} samples  '
              f'{"FAIL: permanent subscriber (starves the robot link)" if perm else "ok"}')
        ok &= not perm

    # Fresh is not the same as sane: on 25/11 13:03 FAST-LIO started while the robot was not
    # still, ran away to km within a minute and still published fresh 10 Hz odometry.
    # Standing still, the pose must not move and must stay near where the robot launch started.
    if not a.moving and len(positions) >= 2:
        moved = max(math.dist(p, positions[0]) for p in positions)
        from_start = max(math.dist(p, (0.0, 0.0, 0.0)) for p in positions)
        bad = []
        if moved > 0.15:
            bad.append(f'moved {moved:.2f} m in {a.seconds:g} s while standing still')
        if from_start > 1.0:
            bad.append(f'{from_start:.1f} m from where the robot launch started')
        print(f'{"FAST-LIO still?":22s} moved {moved:.2f} m, {from_start:.2f} m from start  '
              f'{"ok" if not bad else "FAIL: " + "; ".join(bad) + " -> restart T1 with the robot still"}')
        ok &= not bad

    if a.ready:
        recent = [r for t, r in ready if t > end - 3.0]
        if not recent:
            print('/exploration/localization_ready  FAIL: nothing in the last 3 s (explorer running?)')
            ok = False
        else:
            share = sum(recent) / len(recent)
            print(f'/exploration/localization_ready  true in {share:.0%} of the last 3 s  '
                  f'{"ok" if share == 1.0 else "FAIL"}')
            ok &= share == 1.0

    print('\nGO: odometry is fresh' if ok else '\nNO-GO: do not start autonomous movement')
    node.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
