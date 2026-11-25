"""Rate guard for the robot -> desktop link (issue 9). Pure Python, no ROS.

25/11: one 4.4 MB depth cloud every 2 s over the robot link dropped /livox/imu from 200 to
77 Hz and FAST-LIO lost track. The relay on the robot watches the rates the robot's own
navigation needs and backs off BEFORE they fall below what FAST-LIO needs:
  /livox/imu        >= imu_min_hz   (healthy 200 Hz, odom_preflight.py limit 150)
  /fastlio/odometry >= lidar_min_hz (healthy 10 Hz = lidar health, limit 8)
Degraded: no depth at all, colour image at the low rate. Back to normal only after the
rates were healthy for recover_s (no flapping).
"""
from collections import deque


class RateMeter:
    def __init__(self, window_s=2.0):
        self.window_s = window_s
        self.stamps = deque()

    def tick(self, t):
        self.stamps.append(t)
        self.trim(t)

    def trim(self, now):
        while self.stamps and now - self.stamps[0] > self.window_s:
            self.stamps.popleft()

    def hz(self, now):
        self.trim(now)
        return len(self.stamps) / self.window_s


class LinkGuard:

    def __init__(self, imu_min_hz=150.0, lidar_min_hz=8.0, recover_s=5.0, window_s=2.0,
                 start_s=None):
        self.imu = RateMeter(window_s)
        self.lidar = RateMeter(window_s)
        self.imu_min = imu_min_hz
        self.lidar_min = lidar_min_hz
        self.recover_s = recover_s
        self.window_s = window_s
        self.degraded = True          # start degraded until the rates are proven healthy
        self.healthy_since = None
        self.reason = 'starting: waiting for healthy IMU / lidar rates'
        self.start = start_s

    def update(self, now):
        """Returns (degraded: bool, changed: bool, reason)."""
        if self.start is None:
            self.start = now
        imu, lidar = self.imu.hz(now), self.lidar.hz(now)
        bad = []
        if self.imu_min > 0 and imu < self.imu_min:
            bad.append(f'/livox/imu {imu:.0f} Hz < {self.imu_min:.0f}')
        if self.lidar_min > 0 and lidar < self.lidar_min:
            bad.append(f'lidar (FAST-LIO odometry) {lidar:.1f} Hz < {self.lidar_min:.0f}')
        was = self.degraded
        if bad:
            self.healthy_since = None
            self.degraded = True
            self.reason = '; '.join(bad)
        else:
            if self.healthy_since is None:
                self.healthy_since = now
            if self.degraded and now - self.healthy_since >= self.recover_s:
                self.degraded = False
                self.reason = f'healthy: /livox/imu {imu:.0f} Hz, lidar {lidar:.1f} Hz'
        return self.degraded, self.degraded != was, self.reason


class Throttle:
    """At most rate_hz messages (0 = none)."""

    def __init__(self):
        self.last = None

    def ok(self, now, rate_hz):
        if rate_hz <= 0:
            return False
        if self.last is None or now - self.last >= 1.0 / rate_hz - 1e-6:
            self.last = now
            return True
        return False
