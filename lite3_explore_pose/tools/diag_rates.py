#!/usr/bin/env python3
"""Run-time diagnostics (issue 3): topic rates, CPU load and link bandwidth, logged to CSV.

    python3 ~/challenge_ws/lite3_explore_pose/tools/diag_rates.py                # until Ctrl+C
    python3 ~/challenge_ws/lite3_explore_pose/tools/diag_rates.py --seconds 120 --period 2
    python3 ~/challenge_ws/lite3_explore_pose/tools/diag_rates.py --out /tmp/diag.csv

Run it on EACH machine (robot and desktop) during a run: the rates show what arrives on
that machine, the bandwidth is that machine's NICs.

Every --period seconds one line per machine: Hz of each topic, its worst gap, CPU % (all
cores), load average, the busiest processes and rx/tx MB/s per network interface. Rates
below the healthy minimum (same thresholds as odom_preflight.py, 25/11) are flagged LOW.

Light on purpose: it never subscribes to /livox/lidar or the depth cloud (every extra
subscriber of a heavy topic is one more copy over the robot link, which starved
/livox/imu on 25/11). Lidar health shows as FAST-LIO's output rate.
"""
import argparse
import csv
import os
import signal
import sys
import time

import psutil
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Imu, LaserScan
from tf2_msgs.msg import TFMessage

# topic: (type, qos, healthy min Hz or None)
TOPICS = {
    '/livox/imu': (Imu, 'sensor', 150.0),            # healthy 200 Hz
    '/imu/data': (Imu, 'sensor', 150.0),             # robot IMU, ~200 Hz
    '/fastlio/odometry': (Odometry, 'reliable', 8.0),  # = lidar health (10 Hz)
    '/odom': (Odometry, 'reliable', 8.0),
    '/scan': (LaserScan, 'sensor', 8.0),
    '/map': (OccupancyGrid, 'map', None),            # ~0.5 Hz (map_update_interval 2 s)
    '/tf': (TFMessage, 'reliable', None),
    '/camera/color/image_throttled/compressed': (CompressedImage, 'sensor', 3.0),
}


class Diag(Node):
    def __init__(self, topics):
        super().__init__('diag_rates')
        self.stamps = {t: [] for t in topics}
        qos = {'sensor': qos_profile_sensor_data,
               'reliable': QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE),
               'map': QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)}
        for t in topics:
            typ, q, _ = TOPICS[t]
            self.create_subscription(typ, t, lambda m, t=t: self.stamps[t].append(
                time.monotonic()), qos[q])

    def take(self):
        out, self.stamps = self.stamps, {t: [] for t in self.stamps}
        return out


def net_counters():
    return {k: (v.bytes_recv, v.bytes_sent) for k, v in psutil.net_io_counters(pernic=True).items()
            if k != 'lo' and not k.startswith(('docker', 'br-', 'veth'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=0.0, help='0 = until Ctrl+C')
    ap.add_argument('--period', type=float, default=2.0)
    ap.add_argument('--out', default=os.path.expanduser(
        time.strftime('~/challenge_ws/runs/diag_%Y%m%d_%H%M%S.csv')))
    ap.add_argument('--topics', nargs='*', default=list(TOPICS))
    a = ap.parse_args()

    rclpy.init()
    node = Diag(a.topics)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    f = open(a.out, 'w', newline='')
    w = csv.writer(f)
    nics = sorted(net_counters())
    w.writerow(['time'] + [f'{t} Hz' for t in a.topics] + [f'{t} max_gap_s' for t in a.topics]
               + ['cpu_pct', 'load1'] + [f'{n} rx_MBps' for n in nics]
               + [f'{n} tx_MBps' for n in nics] + ['top_processes'])
    stop = {'now': False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    psutil.cpu_percent(None)
    for p in psutil.process_iter():
        try:
            p.cpu_percent(None)
        except psutil.Error:
            pass
    last_net, last_t, start = net_counters(), time.monotonic(), time.monotonic()
    print(f'diag_rates -> {a.out}  (Ctrl+C to stop)')
    while not stop['now'] and rclpy.ok():
        end = time.monotonic() + a.period
        while time.monotonic() < end and not stop['now']:
            rclpy.spin_once(node, timeout_sec=0.05)
        now = time.monotonic()
        dt = now - last_t
        stamps = node.take()
        hz, gaps, flags = [], [], []
        for t in a.topics:
            s = stamps[t]
            r = len(s) / dt
            gap = max((b - x for x, b in zip(s, s[1:])), default=float('nan') if not s else 0.0)
            hz.append(round(r, 1))
            gaps.append(round(gap, 3))
            lo = TOPICS[t][2]
            if lo is not None and r < lo:
                flags.append(f'{t} LOW {r:.1f} Hz (< {lo:.0f})')
        net = net_counters()
        rx = [round((net.get(n, (0, 0))[0] - last_net.get(n, (0, 0))[0]) / dt / 1e6, 2) for n in nics]
        tx = [round((net.get(n, (0, 0))[1] - last_net.get(n, (0, 0))[1]) / dt / 1e6, 2) for n in nics]
        procs = []
        for p in psutil.process_iter(['name', 'cmdline']):
            try:
                cmd = [os.path.basename(c) for c in (p.info['cmdline'] or [])[:3]
                       if not c.startswith('-')]
                name = ' '.join(c for c in cmd if c not in ('python3', 'python')) or p.info['name']
                procs.append((p.cpu_percent(None), name[:40]))
            except psutil.Error:
                pass
        top = '; '.join(f'{n} {c:.0f}%' for c, n in sorted(procs, reverse=True)[:5])
        cpu = psutil.cpu_percent(None)
        load1 = os.getloadavg()[0]
        w.writerow([round(time.time(), 2)] + hz + gaps + [cpu, round(load1, 2)] + rx + tx + [top])
        f.flush()
        print(f'[{time.strftime("%H:%M:%S")}] ' + '  '.join(
            f'{t.split("/")[-1] or t}={h}' for t, h in zip(a.topics, hz))
            + f'  cpu={cpu:.0f}% load={load1:.1f}  '
            + '  '.join(f'{n} rx={r} tx={x} MB/s' for n, r, x in zip(nics, rx, tx))
            + (f'\n    !! {"; ".join(flags)}' if flags else '') + f'\n    top: {top}')
        last_net, last_t = net, now
        if a.seconds and now - start >= a.seconds:
            break
    f.close()
    node.destroy_node()
    rclpy.try_shutdown()
    print(f'saved {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
