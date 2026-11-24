#!/usr/bin/env python3
"""Reads QR codes with OpenCV (no AI needed) and places them on the map.

QR codes (10-15 cm) are on the left/right walls of rooms of interest. A 12 cm code is only
~56 px wide at 1 m on the 640x480 camera, too small for a full-frame read, so each frame:
  1. full frame, OpenCV QRCodeDetector and QRCodeDetectorAruco
  2. a code found but not decoded -> crop around it, 3x upscale + sharpen, decode again
  3. nothing found -> left or right third of the image (side walls, alternating), 3x + sharpen
(synthetic test: full frame reads >= 90 px codes, steps 2-3 read ~55 px). One CPU thread,
~20 ms per frame (3 Hz = ~6% of one core).
Each new code is:
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

cv2.setNumThreads(1)          # never compete with the robot's own programs for CPU


def enhance(gray, x0, y0, x1, y1, scale=3.0):
    roi = cv2.resize(gray[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(roi, (0, 0), 2)
    return cv2.addWeighted(roi, 1.8, blur, -0.8, 0)


class QrDecoder:
    """Returns (text, corner points in full-image pixels) or ('', points) or ('', None)."""

    def __init__(self):
        self.dets = [cv2.QRCodeDetector()]
        if hasattr(cv2, 'QRCodeDetectorAruco'):
            self.dets.append(cv2.QRCodeDetectorAruco())
        self.side = 0

    def _decode(self, img, dets):
        for d in dets:
            try:
                text, pts, _ = d.detectAndDecode(img)
            except cv2.error:
                continue
            if text and pts is not None:
                return text, pts.reshape(-1, 2)
        return '', None

    def _find(self, img):
        try:
            ok, pts = self.dets[-1].detect(img)    # Aruco if available: finds smaller codes
        except cv2.error:
            return None
        return pts.reshape(-1, 2) if ok and pts is not None else None

    def read(self, gray, scale=3.0):
        """~20 ms per 640x480 frame on one core (measured on arena photos)."""
        text, pts = self._decode(gray, self.dets[:1])          # full frame, plain (~6 ms)
        if text:
            return text, pts
        h, w = gray.shape[:2]
        found = self._find(gray)                               # (~6 ms)
        if found is not None:                                  # rare: crop, both decoders
            x0, y0 = found.min(axis=0)
            x1, y1 = found.max(axis=0)
            m = max(x1 - x0, y1 - y0) * 0.5 + 8
            box = (int(max(0, x0 - m)), int(max(0, y0 - m)),
                   int(min(w, x1 + m)), int(min(h, y1 + m)))
            dets = self.dets
        else:                                                  # side wall, Aruco only (~10 ms)
            self.side ^= 1
            y0, y1 = int(h * 0.1), int(h * 0.9)
            box = (0, y0, w // 3, y1) if self.side else (w - w // 3, y0, w, y1)
            dets = self.dets[-1:]
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            return '', found
        text, pts = self._decode(enhance(gray, *box, scale=scale), dets)
        if text:
            return text, pts / scale + np.array([box[0], box[1]])
        return '', found


class QrReader(Node):

    def __init__(self):
        super().__init__('qr_reader')
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('rate_hz', 3.0)
        self.declare_parameter('place_attempts', 6)       # frames to try to get a position
        g = lambda n: self.get_parameter(n).value
        self.period = 1.0 / g('rate_hz')
        self.place_attempts = g('place_attempts')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.locator = Locator(self, self.tf_buffer, map_topic='/map', cloud_on_demand=True)
        self.decoder = QrDecoder()

        self.create_subscription(CompressedImage, g('image_topic'), self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/perception/room', self.on_room, 10)
        self.qr_pub = self.create_publisher(String, '/perception/qr', 10)
        self.found_pub = self.create_publisher(String, '/perception/found', 10)
        self.event_pub = self.create_publisher(String, '/perception/event', 10)

        self.seen = set()          # texts already sent (with a position, or out of attempts)
        self.announced = set()     # texts already announced (event 'qr' once per code)
        self.attempts = {}
        self.room = 'unknown'
        self.last = 0.0
        self.get_logger().info(f'QR reader ready ({len(self.decoder.dets)} detectors)')

    def on_room(self, msg):
        self.room = msg.data

    def on_image(self, msg):
        now = time.monotonic()
        if now - self.last < self.period:
            return
        self.last = now
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        snap = self.locator.snapshot() or self.locator.camera_snapshot()
        text, pts = self.decoder.read(img)
        if not text or text in self.seen:
            return

        if text not in self.announced:
            self.announced.add(text)
            self.get_logger().warn(f'QR read: "{text}"')
            self.qr_pub.publish(String(data=text))
            self.event_pub.publish(String(data='qr'))   # same as before: stop and look

        h, w = img.shape[:2]
        box = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
        if self.locator.on_demand and snap is not None and snap.get('cloud') is None:
            # depth only now (one cloud), placed when it arrives; never blocks this callback
            self.locator.request_cloud(
                lambda c: self.place_and_send(text, box, w, h, self.locator.attach_cloud(snap, c)),
                timeout_s=1.5)
            return
        self.place_and_send(text, box, w, h, snap)

    def place_and_send(self, text, box, w, h, snap):
        if text in self.seen:
            return
        where = self.locator.place(box, w, h, snap, on='wall', pull_m=0.25)
        self.attempts[text] = self.attempts.get(text, 0) + 1
        if where is None and self.attempts[text] < self.place_attempts:
            return                       # try again on the next frames (text still unsent)
        self.seen.add(text)
        item = {'task_type': 'qr', 'label': '', 'scene_type': self.room,
                'map_xy': where['cell_xy'] if where else None,
                'raw_xy': where['xy'] if where else None,
                'range_m': where['range_m'] if where else None,
                'located_by': where['how'] if where else 'none', 'message': text}
        self.get_logger().warn(f'QR #{len(self.seen)}: {item}')
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
