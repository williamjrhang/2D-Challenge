"""Issue 9: link guard keeps the robot's IMU / lidar first; depth thinned before the link."""
import numpy as np

from lite3_explore_pose.launch_util import profile_problem
from lite3_explore_pose.link_guard import LinkGuard, Throttle
from lite3_explore_pose.link_relay import thin_cloud


def feed(g, t0, t1, imu_hz, lidar_hz):
    t = t0
    while t < t1:
        g.imu.tick(t)
        if lidar_hz and int(t * 1000) % int(1000 / lidar_hz) < int(1000 / imu_hz):
            g.lidar.tick(t)
        t += 1.0 / imu_hz
    return g.update(t1)


def test_guard_starts_degraded_recovers_and_drops_on_low_imu():
    g = LinkGuard(150, 8, recover_s=5.0, window_s=2.0)
    assert g.update(0.0)[0]                                  # degraded until proven healthy
    assert feed(g, 0.0, 3.0, 200, 10)[0]                     # healthy but < recover_s
    deg, changed, _ = feed(g, 3.0, 8.5, 200, 10)
    assert not deg and changed
    deg, changed, why = feed(g, 8.5, 11.0, 77, 10)          # 25/11: IMU fell to 77 Hz
    assert deg and changed and '/livox/imu' in why
    deg, _, why = feed(g, 11.0, 14.0, 200, 3)               # lidar (FAST-LIO) starving
    assert deg and 'lidar' in why


def test_throttle():
    th = Throttle()
    sent = sum(th.ok(t / 30.0, 5.0) for t in range(300))     # 10 s of a 30 Hz stream
    assert 49 <= sent <= 51
    assert not Throttle().ok(0.0, 0.0)


def test_thin_cloud_crops_and_limits_points():
    rng = np.random.default_rng(0)
    xyz = rng.uniform([-2, -2, 0.1], [2, 2, 6.0], (307200, 3))
    xyz[::50] = np.nan
    out = thin_cloud(xyz, 25000, 3.0)
    assert len(out) <= 25000 and out.dtype == np.float32
    assert np.isfinite(out).all() and out[:, 2].max() <= 3.0
    assert out.nbytes < 400_000                              # vs ~4.4 MB raw


def test_dds_profile_only_where_its_address_exists(tmp_path):
    p = tmp_path / 'p.xml'
    p.write_text('<profiles><address>10.255.255.254</address></profiles>')
    assert 'does not have' in profile_problem(str(p))
    p.write_text('<profiles><address>127.0.0.1</address></profiles>')
    assert profile_problem(str(p)) is None
