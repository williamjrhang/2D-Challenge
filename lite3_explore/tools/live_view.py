#!/usr/bin/env python3
"""Lightweight live view of what the robot detects, in a web browser.

Why: RViz over Remote Desktop used ~7 of 10 CPU cores and starved the robot (no IMU /
odometry -> it stops walking). This only forwards the JPEG the obstacle detector already
draws (/perception/obstacle_image/compressed, boxes + labels) and a few status topics, so
it costs almost no CPU on the robot PC; your laptop's browser does the drawing.

Run (robot PC, any terminal):
    python3 ~/challenge_ws/lite3_explore/tools/live_view.py
Open http://localhost:8090 - in VS Code (Remote SSH) the port is forwarded automatically
(see the PORTS tab). Listens on 127.0.0.1 only.
"""
import collections
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32, String

PORT = 8090
PAGE = b"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Robot live view</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#eee}
main{display:flex;flex-wrap:wrap;gap:12px;padding:12px}
img{max-width:100%;width:640px;border:1px solid #333;background:#000}
section{flex:1;min-width:260px}
h2{font-size:15px;margin:4px 0 8px;color:#9cf}
.state{font-size:22px;font-weight:600;margin-bottom:10px}
li{margin:2px 0;font-family:ui-monospace,monospace;font-size:13px}
.ok{color:#7d7}.warn{color:#fc6}
</style></head><body><main>
<img src="/stream" alt="detector image (waiting for /perception/obstacle_image)">
<section>
<div class="state" id="state">...</div>
<div id="info"></div>
<h2>Last events</h2><ul id="events"></ul>
<h2>Last log lines (detector)</h2><ul id="log"></ul>
</section></main>
<script>
async function tick(){
  try{
    const s=await (await fetch('/status')).json();
    document.getElementById('state').textContent='Mission: '+s.mission+'   Crossing: '+s.special;
    document.getElementById('info').innerHTML=
      'Wall ahead: '+(s.wall<0?'none':s.wall.toFixed(2)+' m')+
      ' &nbsp; image age: '+(s.img_age<0?'no image':s.img_age.toFixed(1)+' s');
    document.getElementById('events').innerHTML=s.events.map(e=>'<li class="warn">'+e+'</li>').join('');
    document.getElementById('log').innerHTML=s.log.map(e=>'<li>'+e+'</li>').join('');
  }catch(e){}
}
setInterval(tick,1000);tick();
</script></body></html>"""


class LiveView(Node):

    def __init__(self):
        super().__init__('live_view')
        self.jpeg, self.jpeg_t = None, 0.0
        self.cond = threading.Condition()
        self.mission, self.special, self.wall = '-', '-', -1.0
        self.events = collections.deque(maxlen=12)
        self.log = collections.deque(maxlen=12)
        self.create_subscription(CompressedImage, '/perception/obstacle_image/compressed',
                                 self.on_img, 2)
        self.create_subscription(String, '/mission/state',
                                 lambda m: setattr(self, 'mission', m.data), 10)
        self.create_subscription(String, '/special/state',
                                 lambda m: setattr(self, 'special', m.data), 10)
        self.create_subscription(Float32, '/perception/wall_ahead',
                                 lambda m: setattr(self, 'wall', m.data), 10)
        self.create_subscription(String, '/perception/event', self.on_event, 10)
        from rcl_interfaces.msg import Log
        self.create_subscription(Log, '/rosout', self.on_log, 50)

    def on_img(self, msg):
        with self.cond:
            self.jpeg, self.jpeg_t = bytes(msg.data), time.time()
            self.cond.notify_all()

    def on_event(self, msg):
        self.events.appendleft(f'{time.strftime("%H:%M:%S")}  {msg.data}')

    def on_log(self, msg):
        if msg.name in ('obstacle_detector', 'special_obstacles', 'mission_controller') and \
                not msg.msg.startswith(('lighting:', 'floor ahead:')):
            self.log.appendleft(f'{time.strftime("%H:%M:%S")} [{msg.name}] {msg.msg[:160]}')

    def status(self):
        return {'mission': self.mission, 'special': self.special, 'wall': self.wall,
                'img_age': time.time() - self.jpeg_t if self.jpeg else -1,
                'events': list(self.events), 'log': list(self.log)}


def make_handler(node):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == '/':
                self._send(200, 'text/html; charset=utf-8', PAGE)
            elif self.path == '/status':
                self._send(200, 'application/json', json.dumps(node.status()).encode())
            elif self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                last = 0.0
                try:
                    while rclpy.ok():
                        with node.cond:
                            node.cond.wait_for(lambda: node.jpeg_t > last, timeout=2.0)
                            jpeg, last = node.jpeg, node.jpeg_t
                        if jpeg is None:
                            continue
                        self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpeg)).encode() + b'\r\n\r\n' + jpeg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, 'text/plain', b'not found')

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    rclpy.init()
    node = LiveView()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), make_handler(node))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    node.get_logger().info(f'Live view on http://localhost:{PORT}  (Ctrl+C to stop)')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
