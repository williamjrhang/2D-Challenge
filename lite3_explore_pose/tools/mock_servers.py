#!/usr/bin/env python3
"""Fake scene server (:8000) and scoring server (:8001) for testing without the real ones.

  python3 tools/mock_servers.py [--answer answer.json] [--rate-limit 3] [--delay 2]
then point mission.yaml (a COPY) at http://127.0.0.1:8000/analyze and :8001/task_report.

  /analyze      returns --answer (default: a kitchen with a survivor and a flammable sign)
                after --delay s; every --rate-limit'th request gets HTTP 429 Retry-After: 2
  /health       {"status": "ok"}
  /task_report  logs the report, checks the v6.1 vocabulary, {"status": "accepted"}
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = {'scene_type': {'', 'clothes_shop', 'fitness_centre', 'kitchen', 'market', 'workshop',
                        'clinic', 'cleaning_room', 'furniture_shop'},
         'label': {'', 'high_voltage', 'radioactive', 'chemical', 'biological', 'flammable',
                   'medical_kit', 'hand_tools', 'food', 'water', 'utensils'}}
DEFAULT = {'scene_type': 'kitchen', 'scene_confidence': 0.85,
           'hazards': [{'label': 'flammable', 'box': [0.40, 0.30, 0.55, 0.50], 'confidence': 0.8}],
           'survivors': [{'box': [0.60, 0.30, 0.75, 0.70], 'confidence': 0.8}],
           'objects': [{'label': 'utensils', 'box': [0.2, 0.6, 0.3, 0.8], 'confidence': 0.7}],
           'qr_visible': False}

ap = argparse.ArgumentParser()
ap.add_argument('--answer', default=None)
ap.add_argument('--rate-limit', type=int, default=0)
ap.add_argument('--delay', type=float, default=1.0)
args = ap.parse_args()
answer = json.load(open(args.answer)) if args.answer else DEFAULT
count = [0]


class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body, headers=None):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {'status': 'ok'}) if self.path.endswith('/health') else self.reply(404, {})

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        if self.path.endswith('/analyze'):
            count[0] += 1
            if args.rate_limit and count[0] % args.rate_limit == 0:
                return self.reply(429, {'detail': 'busy'}, {'Retry-After': '2'})
            gid = body.split(b'name="group_id"\r\n\r\n', 1)[-1].split(b'\r\n', 1)[0]
            print(f'[scene] request {count[0]} from {gid.decode(errors="replace")}, '
                  f'{len(body)} bytes', flush=True)
            time.sleep(args.delay)
            return self.reply(200, {'success': True, 'result': json.dumps(answer),
                                    'processing_time_s': args.delay})
        if self.path.endswith('/task_report'):
            r = json.loads(body)
            bad = [k for k, ok in VOCAB.items() if r.get(k) not in ok]
            print(f'[score] {"REJECT " + str(bad) if bad else "accept"} {r}', flush=True)
            return self.reply(200, {'status': 'rejected', 'reason': bad} if bad
                              else {'status': 'accepted'})
        self.reply(404, {})

    def log_message(self, *a):
        pass


for port in (8000, 8001):
    s = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
print('mock servers on 127.0.0.1:8000 (scene) and :8001 (scoring). Ctrl+C to stop.', flush=True)
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
