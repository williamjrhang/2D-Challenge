#!/usr/bin/env python3
"""Save camera photos while you walk the robot - no bag needed.

    python3 save_frames.py                  # saves into ~/challenge_ws/frames/photos/
    python3 save_frames.py curtain          # optional folder name: frames/curtain/

One photo every --every seconds (default 0.7). Ctrl+C to stop.
Needs ROS 2 sourced and the robot launch (camera) running.
"""
import argparse
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class Saver(Node):

    def __init__(self, folder, every, topic):
        super().__init__('save_frames')
        self.folder, self.every = folder, every
        self.last = 0.0
        self.count = 0
        os.makedirs(folder, exist_ok=True)
        self.prefix = f'{os.path.basename(folder)}_{time.strftime("%H%M%S")}'
        self.create_subscription(CompressedImage, topic, self.on_image, qos_profile_sensor_data)
        self.create_timer(5.0, self.check)
        self.get_logger().info(f'Saving one photo every {every} s to {folder}  (Ctrl+C to stop)')

    def on_image(self, msg):
        now = time.monotonic()
        if now - self.last < self.every:
            return
        self.last = now
        ext = 'png' if 'png' in msg.format.lower() else 'jpg'
        path = os.path.join(self.folder, f'{self.prefix}_{self.count:04d}.{ext}')
        with open(path, 'wb') as f:
            f.write(bytes(msg.data))
        self.count += 1
        if self.count % 10 == 0:
            self.get_logger().info(f'{self.count} photos saved')

    def check(self):
        if self.count == 0:
            self.get_logger().warn('No photos yet: is the camera running? '
                                   '(ros2 topic hz /camera/color/image_raw/compressed)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', nargs='?', default='photos',
                    help='folder name (optional, default: photos)')
    ap.add_argument('--every', type=float, default=0.7, help='seconds between photos')
    ap.add_argument('--topic', default='/camera/color/image_raw/compressed')
    ap.add_argument('--root', default=os.path.expanduser('~/challenge_ws/frames'))
    a = ap.parse_args()
    rclpy.init()
    node = Saver(os.path.join(a.root, a.name), a.every, a.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f'\n{node.count} photos saved in {node.folder}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
