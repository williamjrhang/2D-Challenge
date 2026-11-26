#!/usr/bin/env python3
"""Live measurements for the hand gait tests (no tape measure needed).

  python3 measure.py dist    distance walked since start (leg odometry /leg_odom2,
                             the same odometry the automatic crossing uses)
  python3 measure.py pitch   body tilt from the IMU (/imu/data), for the stairs settings

Ctrl+C to stop. Press Enter (dist mode) to restart the distance from 0.
"""

import math
import select
import sys

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dist'
    rclpy.init()
    node = rclpy.create_node(f'measure_{mode}')
    start = [None]
    peak = [0.0]

    def on_odom(m):
        p = m.pose.pose.position
        if start[0] is None:
            start[0] = (p.x, p.y)
        d = math.hypot(p.x - start[0][0], p.y - start[0][1])
        print(f'\rdistance {d:5.2f} m   (Enter = reset to 0)   ', end='', flush=True)

    def on_imu(m):
        q = m.orientation
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (q.w * q.y - q.z * q.x)))))
        peak[0] = max(peak[0], abs(pitch))
        print(f'\rpitch {pitch:+5.1f} deg   (biggest so far {peak[0]:4.1f})   ', end='', flush=True)

    if mode == 'pitch':
        node.create_subscription(Imu, '/imu/data', on_imu, qos_profile_sensor_data)
    else:
        node.create_subscription(Odometry, '/leg_odom2', on_odom, 10)
    print(f'Waiting for {"/imu/data" if mode == "pitch" else "/leg_odom2"} ...')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if mode != 'pitch' and sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                start[0] = None
    except KeyboardInterrupt:
        pass
    print()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
