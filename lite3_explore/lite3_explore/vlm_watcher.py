#!/usr/bin/env python3
"""Scene understanding with the course AI server (ROS Session 6).

Asks WHILE THE ROBOT KEEPS MOVING, only at useful moments:
  - every ask_every_m metres travelled or ask_turn_deg turned (a new view), and
  - when mission_controller stops to look (/perception/look, e.g. QR code / person seen).
One request at a time (server rule). The depth + TF + robot pose are frozen when the
photo is taken, so positions stay correct even though the answer arrives seconds later.

Each answer gives: room/scene type, hazard symbol, survivor, objects of interest,
QR / exit seen. Things found are located on the map (box + depth + TF) and passed
to reporter.py on /perception/found.

Topics
  in   /perception/look        std_msgs/Bool     look now
  in   <image_topic>           sensor_msgs/CompressedImage
  out  /perception/look_done   std_msgs/Bool
  out  /perception/scene       std_msgs/String   full answer (JSON)
  out  /perception/room        std_msgs/String   scene type, e.g. "clinic"
  out  /perception/hazard      std_msgs/String   hazard label, e.g. "biohazard"
  out  /perception/found       std_msgs/String   JSON item for reporter.py

Server API (from the lab's client.py):
  POST <scene_server>  multipart form: group_id, prompt, image (JPEG, max 1024x768)
  -> {"success": true, "result": "<JSON string>", "processing_time_s": ...}
"""

import json
import math
import threading
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

from lite3_explore.locator import Locator
from lite3_explore.mission_config import load_mission

MAX_W, MAX_H = 1024, 768

PROMPT = """Analyze the attached image from a ground robot's front camera.

COMPETITION CONTEXT:
This image comes from a simulated post-disaster robotics competition arena.
The environment may use miniature, toy, cardboard, or printed props to represent
real objects, rooms, equipment, supplies, people and warning signs. Interpret these
props according to the real-world things they clearly represent. Survivors may be
dolls, figurines, mannequins or printed pictures of people.

Return exactly one JSON object with exactly these keys:
"scene_type", "scene_confidence", "hazard_label", "hazard_box", "survivor_visible",
"survivor_box", "objects", "qr_visible", "evidence".

Definitions:
- scene_type: one of {scene_types}. Use "unknown" if the evidence is insufficient.
- scene_confidence: number 0.0-1.0, confidence that scene_type is correct.
- hazard_label: one of {hazard_labels} for a yellow triangular warning sign that is
  clearly visible, otherwise "none".
- survivor_visible: true only if a person / survivor prop is clearly visible.
- objects: array (at most 3) of {{"label": one of {object_labels}, "box": [x1,y1,x2,y2]}}
  for clearly visible objects of interest. Empty array if none.
- hazard_box, survivor_box, and each object box: [x1, y1, x2, y2] with values from 0.0 to
  1.0 as fractions of image width/height (x right, y down), or null if not visible.
- qr_visible: true if a QR code is visible anywhere.
- evidence: array of at most three short strings describing what you see.

Output valid JSON only. No Markdown, code fences, extra keys or text outside the JSON.
Do not infer hidden objects. Do not invent coordinates for things that are not visible."""


def encode_multipart(fields, file_field, filename, data, content_type):
    boundary = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                   .encode())
    out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
               f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode())
    out.append(data)
    out.append(f'\r\n--{boundary}--\r\n'.encode())
    return b''.join(out), f'multipart/form-data; boundary={boundary}'


def parse_result(text):
    text = text.strip()
    if text.startswith('```'):                     # tolerate ```json fences
        text = text.strip('`')
        text = text[text.find('{'):]
    return json.loads(text[text.find('{'):text.rfind('}') + 1])


class VlmWatcher(Node):

    def __init__(self):
        super().__init__('vlm_watcher')
        self.declare_parameter('mission_file', '')
        self.declare_parameter('image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('timeout_s', 30.0)
        self.declare_parameter('min_confidence', 0.6)   # ignore scene types below this
        self.declare_parameter('ask_every_m', 1.5)      # background ask after moving this far
        self.declare_parameter('ask_turn_deg', 60.0)    # ... or turning this much
        g = lambda n: self.get_parameter(n).value
        self.cfg = load_mission(g('mission_file'))
        self.timeout = g('timeout_s')
        self.min_conf = g('min_confidence')
        self.ask_every_m = g('ask_every_m')
        self.ask_turn = math.radians(g('ask_turn_deg'))
        self.prompt = PROMPT.format(
            scene_types=json.dumps(self.cfg['scene_types']),
            hazard_labels=json.dumps(self.cfg['hazard_labels']),
            object_labels=json.dumps(self.cfg['object_labels']))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.locator = Locator(self, self.tf_buffer)

        self.create_subscription(CompressedImage, g('image_topic'), self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(Bool, '/perception/look', self.on_look, 10)
        self.create_subscription(String, '/mission/state', self.on_state, 10)
        self.look_done_pub = self.create_publisher(Bool, '/perception/look_done', 10)
        self.scene_pub = self.create_publisher(String, '/perception/scene', 10)
        self.room_pub = self.create_publisher(String, '/perception/room', 10)
        self.hazard_pub = self.create_publisher(String, '/perception/hazard', 10)
        self.found_pub = self.create_publisher(String, '/perception/found', 10)

        self.image = None
        self.busy = False
        self.look_wanted = False
        self.mission_state = None
        self.last_ask_pose = None
        self.create_timer(0.5, self.tick)
        self.get_logger().info(f'Scene VLM ready: {self.cfg["scene_server"]} as {self.cfg["group_id"]}')

    def on_image(self, msg):
        self.image = msg

    def on_state(self, msg):
        self.mission_state = msg.data

    def on_look(self, msg):
        if not msg.data:
            return
        if self.busy:
            # A background ask is running; its answer will close this look too.
            self.look_wanted = True
            return
        if not self.start(stopped=True):
            self.look_done_pub.publish(Bool(data=True))

    def tick(self):
        """Background asks while moving: only after a new view (moved or turned)."""
        if self.busy or self.mission_state not in (None, 'EXPLORE', 'GO_TO_EXIT'):
            return
        pose = self.locator.robot_pose()
        if pose is None:
            return
        if self.last_ask_pose is not None:
            lx, ly, lyaw = self.last_ask_pose
            moved = math.hypot(pose[0] - lx, pose[1] - ly)
            turned = abs(math.atan2(math.sin(pose[2] - lyaw), math.cos(pose[2] - lyaw)))
            if moved < self.ask_every_m and turned < self.ask_turn:
                return
        self.start(stopped=False)

    def start(self, stopped):
        if self.image is None:
            self.get_logger().warn('No camera image yet.')
            return False
        snap = self.locator.snapshot()           # depth + TF + pose at photo time
        pose = snap['pose'] if snap else self.locator.robot_pose()
        if pose is not None:
            self.last_ask_pose = pose
        self.busy = True
        self.look_wanted = stopped
        threading.Thread(target=self.look, args=(self.image, snap, pose), daemon=True).start()
        return True

    # ================= one look =================

    def look(self, msg, snap, pose):
        try:
            jpeg, w, h = self.prepare(msg)
            answer = self.ask(jpeg)
            if answer is not None:
                self.use(answer, w, h, snap, pose)
        except Exception as e:  # never leave the controller waiting
            self.get_logger().error(f'Look failed: {e!r}')
        finally:
            self.busy = False
            if self.look_wanted:
                self.look_wanted = False
                self.look_done_pub.publish(Bool(data=True))

    def prepare(self, msg):
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        scale = min(1.0, MAX_W / w, MAX_H / h)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes(), w, h

    def ask(self, jpeg):
        body, ctype = encode_multipart(
            {'group_id': self.cfg['group_id'], 'prompt': self.prompt},
            'image', 'frame.jpg', jpeg, 'image/jpeg')
        req = urllib.request.Request(self.cfg['scene_server'], data=body,
                                     headers={'Content-Type': ctype}, method='POST')
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                reply = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors='replace')[:300]
            wait = e.headers.get('retry-after')
            self.get_logger().warn(f'Server HTTP {e.code}: {detail}'
                                   + (f' (retry after {wait} s)' if wait else ''))
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            self.get_logger().warn(f'Cannot reach scene server (RI_network?): {e}')
            return None
        if not reply.get('success'):
            self.get_logger().warn(f'Server could not answer: {reply.get("error")}')
            return None
        try:
            answer = parse_result(reply['result'])
        except (ValueError, KeyError):
            self.get_logger().warn(f'Answer was not JSON: {str(reply.get("result"))[:200]}')
            return None
        answer['latency_s'] = round(time.monotonic() - start, 1)
        self.get_logger().info(f'[{answer["latency_s"]} s] {json.dumps(answer)}')
        return answer

    # ================= use the answer =================

    def use(self, a, w, h, snap, pose):
        self.scene_pub.publish(String(data=json.dumps(a)))
        scene = a.get('scene_type', 'unknown')
        if scene not in self.cfg['scene_types'] or a.get('scene_confidence', 0) < self.min_conf:
            scene = 'unknown'
        self.room_pub.publish(String(data=scene))

        hazard = a.get('hazard_label', 'none')
        if hazard in self.cfg['hazard_labels']:
            self.hazard_pub.publish(String(data=hazard))
            self.found('hazard', hazard, scene, a.get('hazard_box'), w, h, snap, pose)
        if a.get('survivor_visible'):
            self.found('survivor', 'survivor', scene, a.get('survivor_box'), w, h, snap, pose)
        for obj in a.get('objects') or []:
            if obj.get('label') in self.cfg['object_labels']:
                self.found('object', obj['label'], scene, obj.get('box'), w, h, snap, pose)

    def found(self, task_type, label, scene, box, w, h, snap, pose):
        xy, how = None, 'ahead'
        if isinstance(box, list) and len(box) == 4 and snap is not None:
            px = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
            xy = self.locator.locate_box(px, w, h, snap)
            how = 'depth'
        if xy is None:
            xy, how = self.locator.ahead(1.0, pose), 'ahead'
        if xy is None:
            self.get_logger().warn(f'{task_type} {label}: no robot pose, cannot place it.')
            return
        item = {'task_type': task_type, 'label': label, 'scene_type': scene,
                'map_xy': [round(xy[0], 2), round(xy[1], 2)], 'located_by': how, 'message': ''}
        self.get_logger().warn(f'FOUND {item}')
        self.found_pub.publish(String(data=json.dumps(item)))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = VlmWatcher()
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
