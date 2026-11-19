#!/usr/bin/env python3
"""Collects everything found, removes duplicates and submits it to the scoring server.

Each item from /perception/found becomes ONE task report (competition brief, section F):
  POST <report_server>  JSON {group_id, task_id, task_type, scene_type, label, pos, message}
  pos = centre of the arena cell where the item is (arena coordinates, see mission.yaml)

Duplicates: same task_type + label in the same cell (or same QR text) is reported once;
survivors in a room type already reported count as the same survivor.
task_id comes from task_ids in mission.yaml (M01..M05). Each task_id is sent only ONCE.
Items with no matching task_id, or that cannot score (room unknown, label not allowed),
are logged but not sent.
send_reports: false in mission.yaml -> only logs what would be sent (safe for testing).
Every report (sent or not) is appended to ~/challenge_ws/reports/run_<time>.jsonl.

Topics
  in   /perception/found        std_msgs/String  JSON item
  out  /mission/tasks_done      std_msgs/Bool    all expected_counts reached
  out  /mission/found_markers   visualization_msgs/MarkerArray  (RViz)
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from lite3_explore.mission_config import Arena, load_mission, make_report, pick_task_id

PREFIX = {'survivor': 'S', 'hazard': 'H', 'object': 'O', 'qr': 'Q'}
COLOURS = {'survivor': (1.0, 0.2, 0.2), 'hazard': (1.0, 0.8, 0.0),
           'object': (0.2, 0.6, 1.0), 'qr': (0.6, 0.2, 1.0)}


class Reporter(Node):

    def __init__(self):
        super().__init__('reporter')
        self.declare_parameter('mission_file', '')
        self.declare_parameter('log_dir', os.path.expanduser('~/challenge_ws/reports'))
        self.cfg = load_mission(self.get_parameter('mission_file').value)
        self.arena = Arena(self.cfg)
        self.send = bool(self.cfg.get('send_reports', False))

        log_dir = self.get_parameter('log_dir').value
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, time.strftime('run_%Y%m%d_%H%M%S.jsonl'))

        self.create_subscription(String, '/perception/found', self.on_found, 10)
        self.done_pub = self.create_publisher(Bool, '/mission/tasks_done', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/mission/found_markers', 10)

        self.reported = {}      # key -> report
        self.used_ids = set()   # task_ids already sent (each one only once)
        self.counts = {k: 0 for k in PREFIX}
        self.markers = MarkerArray()
        self.lock = threading.Lock()
        mode = 'SENDING to ' + self.cfg['report_server'] if self.send else 'DRY RUN (log only)'
        self.get_logger().info(f'Reporter ready: {mode}. Log: {self.log_path}')

    def on_found(self, msg):
        try:
            item = json.loads(msg.data)
        except ValueError:
            return
        t = item.get('task_type')
        if t not in PREFIX or not item.get('map_xy'):
            return
        pos = list(self.arena.map_to_cell_centre(*item['map_xy']))
        report, problem = make_report(self.cfg, item, pos)
        if t == 'qr':
            key = ('qr', report['message'])
        elif t == 'survivor' and report['scene_type']:
            key = ('survivor', report['scene_type'])     # one survivor per room type
        else:
            key = (t, report['label'], tuple(pos))

        with self.lock:
            if key in self.reported:
                return
            if problem:
                self.get_logger().info(f'not reported ({problem}): {item}',
                                       throttle_duration_sec=5.0)
                return
            tid = pick_task_id(self.cfg.get('task_ids'), report, self.used_ids)
            self.reported[key] = report
            if tid is None:
                self.get_logger().warn(f'NO task_id for {json.dumps(report)} (logged only)')
                self.log(report, 'no_task_id')
                return
            self.used_ids.add(tid)
            self.counts[t] += 1
            report = {'group_id': report['group_id'], 'task_id': tid,
                      **{k: v for k, v in report.items() if k != 'group_id'}}
            self.reported[key] = report
        self.add_marker(t, item['map_xy'], report['label'] or report['scene_type']
                        or report['message'][:20])
        threading.Thread(target=self.submit, args=(report,), daemon=True).start()
        self.check_done()

    def submit(self, report):
        status = 'dry_run'
        if self.send:
            req = urllib.request.Request(
                self.cfg['report_server'], data=json.dumps(report).encode(),
                headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    reply = json.loads(r.read().decode())
                status = reply.get('status', '?')
                if status != 'accepted':
                    self.get_logger().error(f'REJECTED {report["task_id"]}: {reply.get("reason")}')
            except urllib.error.HTTPError as e:
                status = f'http_{e.code}'
                self.get_logger().error(f'Report HTTP {e.code}: {e.read().decode(errors="replace")[:200]}')
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                status = 'unreachable'
                self.get_logger().error(f'Scoring server unreachable: {e}')
        self.get_logger().warn(f'REPORT [{status}] {json.dumps(report)}')
        self.log(report, status)

    def log(self, report, status):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'time': time.time(), 'status': status, 'report': report}) + '\n')

    def check_done(self):
        expected = self.cfg.get('expected_counts') or {}
        wanted = {k: v for k, v in expected.items() if v}
        if wanted and all(self.counts.get(k, 0) >= v for k, v in wanted.items()):
            self.get_logger().info('All expected tasks reported.')
            self.done_pub.publish(Bool(data=True))

    def add_marker(self, task_type, xy, text):
        i = len(self.markers.markers)
        for kind in (Marker.CYLINDER, Marker.TEXT_VIEW_FACING):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id, m.type, m.action = 'found', i, kind, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(xy[0]), float(xy[1])
            m.pose.position.z = 0.3 if kind == Marker.CYLINDER else 0.8
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 0.25
            m.scale.z = 0.6 if kind == Marker.CYLINDER else 0.2
            m.color.r, m.color.g, m.color.b = COLOURS[task_type]
            m.color.a = 0.9
            m.text = f'{task_type}: {text}'
            self.markers.markers.append(m)
            i += 1
        self.marker_pub.publish(self.markers)


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
