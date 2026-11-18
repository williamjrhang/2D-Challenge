#!/usr/bin/env python3
"""Reads QR codes with OpenCV (no AI needed) and places them on the map.

QR codes (10-15 cm) are on the left/right walls of rooms of interest. Each new code is:
  - published on /perception/qr (decoded text)
  - sent to reporter.py on /perception/found (task_type "qr", message = text)
  - announced as event "qr" so mission_controller stops and the VLM names the room

Topics
  in   <image_topic>       sensor_msgs/CompressedImage
  in   /perception/room    std_msgs/String   latest room type from vlm_watcher
  out  /perception/qr      std_msgs/String
  out  /perception/found   std_msgs/String   JSON item for reporter.py
  out  /perception/event   std_msgs/String   "qr"
"""

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from lite3_explore.locator import Locator


class QrReader(Node):

    def __init__(self):
        super().__init__('qr_reader')
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('rate_hz', 3.0)
        g = lambda n: self.get_parameter(n).value
        self.period = 1.0 / g('rate_hz')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.locator = Locator(self, self.tf_buffer)
        self.detector = cv2.QRCodeDetector()

        self.create_subscription(CompressedImage, g('image_topic'), self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/perception/room', self.on_room, 10)
        self.qr_pub = self.create_publisher(String, '/perception/qr', 10)
        self.found_pub = self.create_publisher(String, '/perception/found', 10)
        self.event_pub = self.create_publisher(String, '/perception/event', 10)

        self.seen = set()
        self.room = 'unknown'
        self.last = 0.0
        self.get_logger().info('QR reader ready')

    def on_room(self, msg):
        self.room = msg.data

    def on_image(self, msg):
        now = time.monotonic()
        if now - self.last < self.period:
            return
        self.last = now
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        try:
            text, points, _ = self.detector.detectAndDecode(img)
        except cv2.error:
            return
        if not text or text in self.seen:
            return
        self.seen.add(text)

        h, w = img.shape[:2]
        pts = points.reshape(-1, 2)
        box = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        xy, how = self.locator.locate_box(box, w, h), 'depth'
        if xy is None:
            xy, how = self.locator.ahead(1.0), 'ahead'

        self.get_logger().warn(f'QR #{len(self.seen)}: "{text}" at {xy} ({how})')
        self.qr_pub.publish(String(data=text))
        self.event_pub.publish(String(data='qr'))
        if xy is not None:
            item = {'task_type': 'qr', 'label': '', 'scene_type': self.room,
                    'map_xy': [round(xy[0], 2), round(xy[1], 2)], 'located_by': how,
                    'message': text}
            self.found_pub.publish(String(data=json.dumps(item)))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = QrReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
