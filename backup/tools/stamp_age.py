#!/usr/bin/env python3
"""Clock check between this machine and the robot (issue 9): median (local now - header
stamp) of small stamped topics. On the robot it is the processing delay (~0.0-0.1 s); on the
desktop it is clock offset + link latency. More than ~0.2 s = clocks not synced (chrony /
NTP), and TF lookups / pose ages (pose_max_age_s 2.5) go wrong.

    python3 tools/stamp_age.py [--seconds 5]
"""
import argparse
import statistics
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu
from tf2_msgs.msg import TFMessage

TOPICS = {'/livox/imu': Imu, '/tf': TFMessage,
          '/camera/color/image_throttled/compressed': CompressedImage}


def stamp_of(msg):
    if isinstance(msg, TFMessage):
        return msg.transforms[0].header.stamp if msg.transforms else None
    return msg.header.stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=5.0)
    a = ap.parse_args()
    rclpy.init()
    node = Node('stamp_age')
    ages = {t: [] for t in TOPICS}

    def cb(topic, msg):
        st = stamp_of(msg)
        if st is not None and (st.sec or st.nanosec):
            ages[topic].append(time.time() - (st.sec + st.nanosec * 1e-9))
    for t, typ in TOPICS.items():
        node.create_subscription(typ, t, lambda m, t=t: cb(t, m), qos_profile_sensor_data)
    end = time.monotonic() + a.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    worst = 0.0
    for t, v in ages.items():
        if v:
            med = statistics.median(v)
            worst = max(worst, abs(med))
            print(f'  {t:45s} median age {med:+.3f} s  ({len(v)} msgs)')
        else:
            print(f'  {t:45s} no messages')
    node.destroy_node()
    rclpy.try_shutdown()
    if not ages['/livox/imu']:
        print('  no /livox/imu: cannot judge the clock (robot not visible?)')
        raise SystemExit(1)
    raise SystemExit(0 if worst < 0.2 else 1)


if __name__ == '__main__':
    main()
