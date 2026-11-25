#!/usr/bin/env python3
"""Collects every sighting, decides the answer for each mission and submits it.

  /perception/found (sightings from vlm_watcher / qr_reader)
     -> Evidence (per arena cell: scene votes, items, QR texts; mission_config.py)
     -> one answer per mission (config/mission.yaml `missions:`, M01..M05)
     -> POST <report_server> {group_id, task_id, task_type, scene_type, label, pos, message}

Each item gets the room of ITS OWN cell (votes from several photos), pos = that cell's centre.
An answer is sent once it is confirmed (enough sightings and a clear room vote). The server
keeps only the LATEST submission per task, so a task is resent only when the answer changes
AND the new evidence is clearly stronger (never a weak guess over a confirmed one).
report_best_guess_at_s after the mission starts (or at GO_TO_EXIT), still-open missions get
their best unconfirmed guess.

send_reports: false -> only logs what would be sent. Every sighting and every report
(sent or not, with the server reply) goes to ~/challenge_ws/reports/run_<time>.jsonl.

This node only listens to perception: it never commands the robot.

Topics
  in   /perception/found        std_msgs/String  JSON sighting
  in   /mission/state           std_msgs/String  start time + GO_TO_EXIT
  out  /mission/tasks_done      std_msgs/Bool    all expected_counts reached
  out  /mission/found_markers   visualization_msgs/MarkerArray  (RViz)
"""

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explore_pose.mission_config import (Arena, Evidence, load_mission, load_missions,
                                          make_report, parse_mission, range_weight)

WALL_KINDS = ('qr', 'hazard', 'survivor')   # things pinned on walls (booklet)
COLOURS = {'survivor': (1.0, 0.2, 0.2), 'hazard': (1.0, 0.8, 0.0),
           'object': (0.2, 0.6, 1.0), 'qr': (0.6, 0.2, 1.0)}
PLACEHOLDERS = ('PIN', 'XXXX')


def merge_wall_item(items, kind, text, xy, merge_m):
    """Same kind + text within merge_m = same item (running mean); else a new one."""
    for w in items:
        if (w['kind'] == kind and w['text'] == text
                and ((w['x'] - xy[0]) ** 2 + (w['y'] - xy[1]) ** 2) ** 0.5 < merge_m):
            n = w['n']
            w['x'] = (w['x'] * n + xy[0]) / (n + 1)
            w['y'] = (w['y'] * n + xy[1]) / (n + 1)
            w['n'] = n + 1
            return w
    w = {'id': len(items), 'kind': kind, 'text': text, 'x': float(xy[0]), 'y': float(xy[1]),
         'n': 1}
    items.append(w)
    return w


def should_send(new, old, margin):
    """Send `new` for a task whose last submission was `old` (None = never sent)?"""
    if old is None:
        return True
    same = all(new[k] == old[k] for k in ('task_type', 'label', 'scene_type', 'pos', 'message'))
    if same:
        return False
    if old['confirmed'] and not new['confirmed']:
        return False                       # never a guess over a confirmed answer
    if new['confirmed'] and not old['confirmed']:
        return True
    return new['score'] >= old['score'] + margin


class Reporter(Node):

    def __init__(self):
        super().__init__('reporter')
        self.declare_parameter('mission_file', '')
        self.declare_parameter('log_dir', os.path.expanduser('~/challenge_ws/reports'))
        # Issue 2: wall items (QR / hazard / survivor) as RViz markers at their WALL position
        # (raw_xy), merged within wall_item_merge_m and moved to the running mean.
        self.declare_parameter('wall_item_merge_m', 0.5)
        self.wall_merge = float(self.get_parameter('wall_item_merge_m').value)
        self.wall_items = []        # {key, kind, text, x, y, n}
        self.cfg = load_mission(self.get_parameter('mission_file').value)
        self.arena = Arena(self.cfg)
        self.evidence = Evidence(self.cfg, self.arena)
        self.send = bool(self.cfg.get('send_reports', False))
        self.guess_at = float(self.cfg.get('report_best_guess_at_s', 0) or 0)

        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, time.strftime('run_%Y%m%d_%H%M%S.jsonl'))

        texts = load_missions(self.cfg, self.cfg.get('missions_file', ''))
        self.missions = {}
        for tid, text in texts.items():
            m = parse_mission(text, self.cfg)
            self.get_logger().info(f'Mission {tid}: "{text}" -> {m}')
            if m['task_type'] is None:
                self.get_logger().error(f'Mission {tid} NOT UNDERSTOOD: "{text}" (ignored)')
                continue
            self.missions[tid] = m
        self.log({'event': 'missions', 'texts': texts, 'parsed': self.missions})

        gid = str(self.cfg.get('group_id', ''))
        if self.send and any(p in gid for p in PLACEHOLDERS):
            self.get_logger().error(f'group_id "{gid}" is still a placeholder: NOT sending.')
            self.send = False

        self.create_subscription(String, '/perception/found', self.on_found, 50)
        self.create_subscription(String, '/mission/state', self.on_state, 10)
        self.done_pub = self.create_publisher(Bool, '/mission/tasks_done', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/mission/found_markers', 10)

        self.sent = {}              # task_id -> last answer submitted (with score)
        self.start_time = None
        self.final = False
        self.lock = threading.Lock()
        self.outbox = queue.Queue()
        self.pending = {}           # task_id -> newest report waiting for the sender
        threading.Thread(target=self.sender, daemon=True).start()
        self.create_timer(1.0, self.tick)
        mode = 'SENDING to ' + self.cfg['report_server'] if self.send else 'DRY RUN (log only)'
        self.get_logger().info(f'Reporter ready: {mode} as {gid}. Log: {self.log_path}')

    # ================= inputs =================

    def on_state(self, msg):
        if self.start_time is None and msg.data not in ('', 'IDLE', 'WAIT'):
            self.start_time = time.monotonic()
        if msg.data == 'GO_TO_EXIT' and not self.final:
            self.get_logger().info('GO_TO_EXIT: sending best guesses for open missions.')
            self.final = True
            self.evaluate()

    def tick(self):
        if (not self.final and self.guess_at > 0 and self.start_time is not None
                and time.monotonic() - self.start_time > self.guess_at):
            self.get_logger().info(f'{self.guess_at:.0f} s: sending best guesses for open missions.')
            self.final = True
            self.evaluate()

    def on_found(self, msg):
        try:
            item = json.loads(msg.data)
        except ValueError:
            return
        t = item.get('task_type')
        xy = item.get('map_xy')
        cell = self.evidence.cell(xy) if xy else None
        weight = item.get('weight')
        if weight is None:
            weight = float(item.get('confidence', 0.7)) * range_weight(item.get('range_m'))
        with self.lock:
            if t == 'qr':
                if not item.get('message'):
                    return
                self.evidence.add_qr(item['message'], cell)
            elif cell is None:
                return
            elif t == 'scene':
                self.evidence.add_scene(cell, item.get('scene_type'), weight)
            elif t in ('hazard', 'object', 'survivor'):
                label = item.get('label') if t != 'survivor' else 'survivor'
                photo = item.get('photo', time.monotonic())
                self.evidence.add_item(t, label, cell, weight, photo)
                # the photo's scene also counts (less) for the item's own cell
                self.evidence.add_scene(cell, item.get('scene_type'),
                                        weight * self.evidence.p['item_scene_weight'])
            else:
                return
        self.log({'event': 'sighting', 'cell': list(cell) if cell else None, 'item': item})
        self.wall_item(item)
        self.evaluate()

    # ================= decide and send =================

    def evaluate(self):
        with self.lock:
            answers = self.evidence.decide(self.missions, final=self.final)
            margin = self.evidence.p['resubmit_margin']
            to_send = []
            for tid, a in answers.items():
                if not should_send(a, self.sent.get(tid), margin):
                    continue
                report, problem = make_report(self.cfg, a, a['pos'])
                if problem:
                    self.get_logger().info(f'{tid} not sent ({problem}): {a}',
                                           throttle_duration_sec=5.0)
                    continue
                report = {'group_id': report['group_id'], 'task_id': tid,
                          **{k: v for k, v in report.items() if k != 'group_id'}}
                old = self.sent.get(tid)
                self.sent[tid] = a
                to_send.append((tid, report, a, old))
        for tid, report, a, old in to_send:
            why = 'first' if old is None else f'replaces {old["scene_type"] or old["message"]}' \
                f'@{old["pos"]} (score {old["score"]})'
            self.get_logger().warn(f'{tid} -> {"CONFIRMED" if a["confirmed"] else "best guess"} '
                                   f'score {a["score"]} ({why}): {json.dumps(report)}')
            self.log({'event': 'decision', 'task_id': tid, 'answer': a, 'report': report})
            self.add_marker(tid, a)
            self.pending[tid] = report
            self.outbox.put(tid)
        if to_send:
            self.check_done()

    def sender(self):
        """One report at a time, newest answer per task, retries with backoff."""
        while True:
            tid = self.outbox.get()
            report = self.pending.pop(tid, None)
            if report is None:
                continue                         # a newer one for this task was sent already
            status, reply = 'dry_run', None
            if self.send:
                status, reply = self.post(report)
            self.get_logger().warn(f'REPORT [{status}] {json.dumps(report)}')
            self.log({'event': 'report', 'status': status, 'reply': reply, 'report': report})

    def post(self, report, tries=4, timeout=5.0):
        data = json.dumps(report).encode()
        wait = 1.0
        status, reply = 'unreachable', None
        for attempt in range(tries):
            if attempt and report['task_id'] in self.pending:
                return 'superseded', None        # a newer answer is waiting: send that one
            req = urllib.request.Request(self.cfg['report_server'], data=data,
                                         headers={'Content-Type': 'application/json'},
                                         method='POST')
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    reply = r.read().decode(errors='replace')[:500]
                try:
                    status = json.loads(reply).get('status', 'ok')
                except (ValueError, AttributeError):
                    status = 'ok'
                if status not in ('accepted', 'ok'):
                    self.get_logger().error(f'REJECTED {report["task_id"]}: {reply}')
                return status, reply
            except urllib.error.HTTPError as e:
                reply = e.read().decode(errors='replace')[:500]
                status = f'http_{e.code}'
                self.get_logger().error(f'Report {report["task_id"]} HTTP {e.code}: {reply}')
                if e.code == 429:
                    try:
                        wait = float(e.headers.get('retry-after') or wait)
                    except ValueError:
                        pass
                elif 400 <= e.code < 500:
                    return status, reply          # our report is wrong: retrying won't help
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                status, reply = 'unreachable', str(e)
                self.get_logger().error(f'Scoring server unreachable (try {attempt + 1}): {e}')
            time.sleep(min(wait, 10.0))
            wait *= 2
        return status, reply

    def log(self, record):
        record = {'time': time.time(), **record}
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')

    def check_done(self):
        expected = self.cfg.get('expected_counts') or {}
        wanted = {k: v for k, v in expected.items() if v}
        counts = {}
        for a in self.sent.values():
            counts[a['task_type']] = counts.get(a['task_type'], 0) + 1
        if wanted and all(counts.get(k, 0) >= v for k, v in wanted.items()):
            self.get_logger().info('All expected tasks reported.')
            self.done_pub.publish(Bool(data=True))

    def wall_item(self, item):
        """Marker at the wall where a QR / hazard / survivor picture was placed."""
        t, raw = item.get('task_type'), item.get('raw_xy')
        if t not in WALL_KINDS or not raw:
            return
        text = item.get('message') if t == 'qr' else (item.get('label') or t)
        w = merge_wall_item(self.wall_items, t, text, raw, self.wall_merge)
        arr = MarkerArray()
        for k, kind in enumerate((Marker.CUBE, Marker.TEXT_VIEW_FACING)):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'wall_items', w['id'] * 2 + k, kind, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(w['x']), float(w['y'])
            m.pose.position.z = 0.35 if kind == Marker.CUBE else 0.65
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.12
            m.scale.z = 0.12 if kind == Marker.CUBE else 0.15
            m.color.r, m.color.g, m.color.b = COLOURS.get(t, (1.0, 1.0, 1.0))
            m.color.a = 0.95
            m.text = f'{t}: {str(text)[:24]} (x{w["n"]})'
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def add_marker(self, tid, a):
        """One cylinder + label per task, at the reported cell (moves if the answer changes)."""
        if a['pos'] is None:
            return
        x, y = self.arena.arena_to_map(*a['pos'])
        idx = sorted(self.missions).index(tid) if tid in self.missions else 99
        arr = MarkerArray()
        for k, kind in enumerate((Marker.CYLINDER, Marker.TEXT_VIEW_FACING)):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'found', idx * 2 + k, kind, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(x), float(y)
            m.pose.position.z = 0.3 if kind == Marker.CYLINDER else 0.8
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.25
            m.scale.z = 0.6 if kind == Marker.CYLINDER else 0.2
            m.color.r, m.color.g, m.color.b = COLOURS.get(a['task_type'], (1.0, 1.0, 1.0))
            m.color.a = 0.9 if a['confirmed'] else 0.4
            m.text = f'{tid} {a["label"] or a["scene_type"]}'
            arr.markers.append(m)
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = Reporter()
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
