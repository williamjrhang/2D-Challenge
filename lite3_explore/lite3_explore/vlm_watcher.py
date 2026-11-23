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
from lite3_explore.mission_config import load_mission, range_weight

MAX_W, MAX_H = 1024, 768

PROMPT = """Analyze this image from a small ground robot's front camera in a search-and-rescue
competition arena. The arena is a maze of 1 m x 1 m rooms with wooden walls. Rooms are furnished
with miniature, toy, cardboard or printed props; interpret props as the real things they represent.

ROOM TYPES (scene_type) and what they typically contain:
<ROOM_HINTS>
Judge the room that fills most of the image (the nearest room you are looking into). Use "unknown"
if there are not enough clues (e.g. only bare walls or a corridor).

HAZARD SIGNS - report ONLY these 5 warning symbols (usually a yellow triangle with a black symbol):
- high_voltage: lightning bolt / electric arrow
- radioactive: trefoil (three blades around a central dot)
- chemical: three small solid balls joined by short lines to a central point, like a molecule
  (chemical weapon / chemical agent symbol); also a corrosive or toxic chemical symbol
- biological: biohazard symbol (three interlocking circles)
- flammable: flame (fire) above a line
NEVER report these look-alike signs, they are distractors: <DISTRACTORS>.
Careful: a flame is flammable but a starburst (spiky sun-like burst around a circle) or exploding
bomb is a distractor; a three-blade trefoil is radioactive but an antenna with radio waves is a
distractor; three balls joined like a molecule is chemical but a starburst is not. If unsure, do
not report the sign.

SURVIVORS: in this arena survivors are PICTURES or MODELS of people: a printed photo or poster
pasted on a wall (usually a room's left or right wall), a drawing, a picture on a screen, a doll,
toy figure or figurine. Every such picture or model of a person counts as a survivor, even if it
shows a real-looking adult. Do NOT count the actual living people physically present around the
robot (spectators' legs, feet, shoes, hands holding things, people standing in the background)
- only report the picture or model.

OBJECTS of interest: medical_kit (first-aid kit, red-cross box), hand_tools (hammer, screwdriver,
wrench, pliers), food (fruit, bread, cans, packaged food), water (water bottles), utensils
(cutlery, spoons, forks, ladles).

Return exactly one JSON object with exactly these keys:
{"scene_type": one of <SCENE_TYPES> or "unknown",
 "scene_confidence": number 0.0-1.0,
 "scene_box": [x1, y1, x2, y2] around the props that show the room type, or null if "unknown",
 "hazards": [{"label": one of <HAZARD_LABELS>, "box": [x1, y1, x2, y2], "confidence": 0.0-1.0}],
 "survivors": [{"box": [x1, y1, x2, y2], "confidence": 0.0-1.0}],
 "objects": [{"label": one of <OBJECT_LABELS>, "box": [x1, y1, x2, y2], "confidence": 0.0-1.0}],
 "qr_visible": true or false}
Boxes are fractions 0.0-1.0 of the image width/height (x right, y down). Use empty lists if none.
Only report things that are clearly visible; never invent objects or coordinates.
Output valid JSON only. No Markdown, code fences, extra keys or text outside the JSON."""


def build_prompt(cfg):
    hints = cfg.get('room_hints') or {}
    rooms = '\n'.join(f'- {s}: {hints.get(s, "")}'.rstrip(': ') for s in cfg['scene_types'])
    return (PROMPT.replace('<ROOM_HINTS>', rooms)
            .replace('<DISTRACTORS>', '; '.join(cfg.get('hazard_distractors') or []))
            .replace('<SCENE_TYPES>', json.dumps(cfg['scene_types']))
            .replace('<HAZARD_LABELS>', json.dumps(cfg['hazard_labels']))
            .replace('<OBJECT_LABELS>', json.dumps(cfg['object_labels'])))


def valid_box(box):
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    try:
        b = [min(1.0, max(0.0, float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    return b


def clean_answer(a, cfg):
    """Keep only what the booklet allows: [(task_type, label, box, confidence)], scene, conf.
    Also accepts the old single-item keys (hazard_label / survivor_visible)."""
    items = []
    hazards = list(a.get('hazards') or [])
    if a.get('hazard_label') not in (None, 'none'):
        hazards.append({'label': a.get('hazard_label'), 'box': a.get('hazard_box')})
    for h in hazards:
        if isinstance(h, dict) and h.get('label') in cfg['hazard_labels']:
            items.append(('hazard', h['label'], valid_box(h.get('box')), h.get('confidence')))
    survivors = list(a.get('survivors') or [])
    if a.get('survivor_visible'):
        survivors.append({'box': a.get('survivor_box')})
    for v in survivors:
        if isinstance(v, dict):
            items.append(('survivor', 'survivor', valid_box(v.get('box')), v.get('confidence')))
    for o in a.get('objects') or []:
        if isinstance(o, dict) and o.get('label') in cfg['object_labels']:
            items.append(('object', o['label'], valid_box(o.get('box')), o.get('confidence')))
    out = []
    for t, label, box, conf in items:
        try:
            conf = float(conf) if conf is not None else 0.7
        except (TypeError, ValueError):
            conf = 0.7
        out.append((t, label, box, conf))
    scene = a.get('scene_type')
    try:
        sconf = float(a.get('scene_confidence') or 0.0)
    except (TypeError, ValueError):
        sconf = 0.0
    if scene not in cfg['scene_types']:
        scene, sconf = 'unknown', 0.0
    return scene, sconf, out


def scene_box(a):
    """Box around the room's props (where the room is), else None."""
    return valid_box(a.get('scene_box'))


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
        self.declare_parameter('min_interval_s', 2.0)   # never ask more often than this
        self.declare_parameter('ask_every_s', 0.0)      # >0: also ask on a timer, even standing
                                                        # still (bench testing). 0 = off
        self.declare_parameter('item_min_confidence', 0.4)
        self.declare_parameter('photo_dir', '~/challenge_ws/reports/vlm_photos')  # '' = off
        g = lambda n: self.get_parameter(n).value
        self.cfg = load_mission(g('mission_file'))
        self.timeout = g('timeout_s')
        self.min_conf = g('min_confidence')
        self.ask_every_m = g('ask_every_m')
        self.ask_turn = math.radians(g('ask_turn_deg'))
        self.min_interval = g('min_interval_s')
        self.ask_every_s = g('ask_every_s')
        self.item_min_conf = g('item_min_confidence')
        self.prompt = build_prompt(self.cfg)
        self.next_ok = 0.0          # monotonic time before which we do not ask (429 / errors)
        self.failures = 0
        self.last_start = 0.0
        self.photo = 0
        self.photo_dir = ''
        if g('photo_dir'):
            import os
            self.photo_dir = os.path.join(os.path.expanduser(g('photo_dir')),
                                          time.strftime('%Y%m%d_%H%M%S'))
            os.makedirs(self.photo_dir, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.locator = Locator(self, self.tf_buffer, map_topic='/map')

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
        threading.Thread(target=self.health_check, daemon=True).start()
        self.get_logger().info(f'Scene VLM ready: {self.cfg["scene_server"]} as {self.cfg["group_id"]}')

    def health_check(self):
        """GET <server>/health until it answers (only logs, never blocks anything)."""
        url = self.cfg['scene_server'].rsplit('/', 1)[0] + '/health'
        for attempt in range(30):
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    body = r.read().decode(errors='replace')[:200]
                self.get_logger().info(f'Scene server healthy: {body}')
                return
            except Exception as e:  # noqa: BLE001  (log and retry)
                if attempt == 0 or attempt % 6 == 5:
                    self.get_logger().error(f'Scene server NOT reachable ({url}): {e}')
            time.sleep(10.0)

    def can_ask(self):
        now = time.monotonic()
        return now >= self.next_ok and now - self.last_start >= self.min_interval

    def back_off(self, seconds=None):
        """After a 429 (seconds = Retry-After) or a failure (exponential, max 10 s)."""
        self.failures += 1
        wait = seconds if seconds is not None else min(10.0, 2.0 ** self.failures)
        self.next_ok = time.monotonic() + wait

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
        # Server backing off (429 / down): answer at once so the robot does not wait.
        if time.monotonic() < self.next_ok or not self.start(stopped=True):
            self.look_done_pub.publish(Bool(data=True))

    def tick(self):
        """Background asks while moving: only after a new view (moved or turned)."""
        if (self.busy or self.mission_state not in (None, 'EXPLORE', 'GO_TO_EXIT')
                or not self.can_ask()):
            return
        pose = self.locator.robot_pose()
        timed = (self.ask_every_s > 0
                 and time.monotonic() - self.last_start >= self.ask_every_s)
        if pose is None:
            if timed:                        # bench test without SLAM: still ask
                self.start(stopped=False)
            return
        if self.last_ask_pose is not None:
            lx, ly, lyaw = self.last_ask_pose
            moved = math.hypot(pose[0] - lx, pose[1] - ly)
            turned = abs(math.atan2(math.sin(pose[2] - lyaw), math.cos(pose[2] - lyaw)))
            if moved < self.ask_every_m and turned < self.ask_turn and not timed:
                return
        self.start(stopped=False)

    def start(self, stopped):
        if self.image is None:
            self.get_logger().warn('No camera image yet.')
            return False
        snap = self.locator.snapshot() or self.locator.camera_snapshot()  # depth + TF + pose now
        pose = snap['pose'] if snap else self.locator.robot_pose()
        if pose is not None:
            self.last_ask_pose = pose
        self.busy = True
        self.last_start = time.monotonic()
        self.look_wanted = stopped
        threading.Thread(target=self.look, args=(self.image, snap, pose), daemon=True).start()
        return True

    # ================= one look =================

    def look(self, msg, snap, pose):
        try:
            jpeg, w, h = self.prepare(msg)
            answer = self.ask(jpeg)
            self.save_photo(jpeg, answer)
            if answer is not None:
                self.use(answer, w, h, snap, pose)
        except Exception as e:  # never leave the controller waiting
            self.get_logger().error(f'Look failed: {e!r}')
        finally:
            self.busy = False
            if self.look_wanted:
                self.look_wanted = False
                self.look_done_pub.publish(Bool(data=True))

    def save_photo(self, jpeg, answer):
        """Every photo sent + its answer, to check what the server really saw."""
        if not self.photo_dir:
            return
        try:
            name = time.strftime('%H%M%S') + f'_{self.photo + 1:03d}'
            with open(f'{self.photo_dir}/{name}.jpg', 'wb') as f:
                f.write(jpeg)
            with open(f'{self.photo_dir}/{name}.json', 'w') as f:
                json.dump(answer, f)
        except OSError as e:
            self.get_logger().warn(f'Cannot save photo: {e}', throttle_duration_sec=60.0)

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
            try:
                self.back_off(float(wait) if wait else (5.0 if e.code == 429 else None))
            except ValueError:
                self.back_off(5.0)
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            self.get_logger().warn(f'Cannot reach scene server (RI_network?): {e}')
            self.back_off()
            return None
        if not reply.get('success'):
            self.get_logger().warn(f'Server could not answer: {reply.get("error")}')
            self.back_off()
            return None
        self.failures = 0
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
        scene, sconf, items = clean_answer(a, self.cfg)
        if sconf < self.min_conf:
            scene = 'unknown'
        self.room_pub.publish(String(data=scene))
        self.photo += 1
        photo = self.photo

        # Scene vote, placed where the room's props are (scene_box), else the image centre.
        # (Centre alone fails when e.g. rubble in front fills the middle of the picture.)
        if scene != 'unknown':
            b = scene_box(a) or [0.3, 0.3, 0.7, 0.7]
            where = self.locator.place([b[0] * w, b[1] * h, b[2] * w, b[3] * h], w, h, snap,
                                       on='wall', pull_m=0.25)
            if where:
                self.publish('scene', '', scene, where, sconf, photo)
            else:
                self.get_logger().warn(f'scene {scene}: cannot place the view (no depth/map).')

        for task_type, label, box, conf in items:
            if conf < self.item_min_conf:
                continue
            if task_type == 'hazard':
                self.hazard_pub.publish(String(data=label))
            where = None
            if box is not None:
                px = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]
                on = 'floor' if task_type == 'object' else 'wall'
                where = self.locator.place(px, w, h, snap, on=on,
                                           pull_m=0.1 if on == 'floor' else 0.25)
            if where is None:
                # never guess a position: an unplaced item cannot score its pos points
                self.get_logger().warn(f'{task_type} {label}: seen but cannot place it '
                                       f'(box={box}); not reported.')
                continue
            self.publish(task_type, label, scene, where, conf, photo)

    def publish(self, task_type, label, scene, where, conf, photo):
        """One sighting for reporter.py. map_xy = the point used for the cell (pulled
        towards the camera); weight = confidence x distance weight."""
        weight = round(conf * range_weight(where['range_m']), 3)
        item = {'task_type': task_type, 'label': label, 'scene_type': scene,
                'map_xy': where['cell_xy'], 'raw_xy': where['xy'], 'cam_xy': where['cam_xy'],
                'range_m': where['range_m'], 'located_by': where['how'],
                'confidence': round(conf, 2), 'weight': weight, 'photo': photo, 'message': ''}
        if task_type != 'scene':
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
