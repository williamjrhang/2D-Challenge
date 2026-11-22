"""Node-level test: the real WallPins node, fed directly (no DDS traffic, no robot).

Camera optical frame = depth frame = map orientation; a wall 1.0 m in front of the camera
with a QR code centred on the optical axis -> the pin must be at map (~0, ~0, 1.0).
"""
import os

import cv2
import numpy as np
import pytest

rclpy = pytest.importorskip('rclpy')
from geometry_msgs.msg import TransformStamped  # noqa: E402
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2, PointField  # noqa: E402

from lite3_wall_pins.pin_node import WallPins  # noqa: E402

DOMAIN = 99            # isolated test domain (robot uses 5)
K = [459.432, 0.0, 320.0, 0.0, 459.38, 240.0, 0.0, 0.0, 1.0]


def tf(parent, child, stamp):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.header.stamp = stamp
    t.transform.rotation.w = 1.0
    return t


def qr_frame(text='UNIT-QR', px=120):
    q = cv2.QRCodeEncoder.create().encode(text)
    q = cv2.resize(q, (px, px), interpolation=cv2.INTER_NEAREST)
    img = np.full((480, 640, 3), 170, np.uint8)
    img[240 - px // 2:240 + px // 2, 320 - px // 2:320 + px // 2] = q[..., None]
    return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def wall_cloud(stamp, depth=1.0):
    g = np.arange(-0.6, 0.6, 0.004)
    x, y = np.meshgrid(g, g)
    pts = np.stack([x.ravel(), y.ravel(), np.full(x.size, depth)], 1).astype(np.float32)
    c = PointCloud2()
    c.header.frame_id, c.header.stamp = 'cam_depth', stamp
    c.height, c.width = 1, len(pts)
    c.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                for i, n in enumerate('xyz')]
    c.point_step, c.row_step, c.is_dense = 12, 12 * len(pts), True
    c.data = pts.tobytes()
    return c


@pytest.fixture
def node(tmp_path):
    # Own DDS domain + unused topic names: never sees (or talks to) a live robot.
    rclpy.init(domain_id=DOMAIN, args=[
        '--ros-args', '-p', f'pins_file:={tmp_path / "p.json"}', '-p', 'map_id:=unit',
        '-p', 'sync_wait_s:=0.05', '-p', 'tf_timeout_s:=0.05',
        '-p', 'image_topic:=/wall_pins_unit/img', '-p', 'cloud_topic:=/wall_pins_unit/cloud',
        '-p', 'camera_info_topic:=/wall_pins_unit/info',
        '-r', '/tf:=/wall_pins_unit/tf', '-r', '/tf_static:=/wall_pins_unit/tf_static',
        '-r', '/wall_pins/markers:=/wall_pins_unit/markers',
        '-r', '/wall_pins/events:=/wall_pins_unit/events',
        '-r', '/wall_pins/debug_image:=/wall_pins_unit/debug_image'])
    n = WallPins()
    n.running = False                     # stop the worker; the test drives process()
    n.worker.join(timeout=2)
    yield n
    n.destroy_node()
    rclpy.shutdown()


def feed(n, stamp, with_cloud=True, with_tf=True):
    if with_tf:
        n.tf_buffer.set_transform_static(tf('map', 'cam_color', stamp), 'test')
        n.tf_buffer.set_transform_static(tf('cam_color', 'cam_depth', stamp), 'test')
        n.tf_buffer.set_transform(tf('map', 'base_link', stamp), 'test')
    info = CameraInfo()
    info.header.frame_id, info.width, info.height = 'cam_color', 640, 480
    info.k = K
    info.d = [0.0] * 5
    n.on_info(info)
    if with_cloud:
        n.on_cloud(wall_cloud(stamp))
    msg = CompressedImage()
    msg.header.stamp, msg.header.frame_id, msg.format = stamp, 'cam_color', 'jpeg'
    msg.data = qr_frame()
    return msg


def test_pin_created_on_wall_then_deduplicated(node):
    stamp = node.get_clock().now().to_msg()
    node.process(feed(node, stamp), False)
    assert len(node.store.pins) == 1
    p = node.store.pins[0]
    assert p.payload == 'UNIT-QR'
    assert abs(p.z - 1.0) < 0.02 and abs(p.x) < 0.03 and abs(p.y) < 0.03
    assert p.view is not None and p.view.range_m == pytest.approx(1.0, abs=0.03)
    assert os.path.isfile(node.store.path)          # new pins are saved immediately
    for _ in range(3):
        node.process(feed(node, node.get_clock().now().to_msg()), False)
    assert len(node.store.pins) == 1 and node.store.pins[0].observations == 4


def test_no_depth_no_pin(node):
    stamp = node.get_clock().now().to_msg()
    node.process(feed(node, stamp, with_cloud=False), False)
    assert node.store.pins == []


def test_future_stamp_no_pin(node):
    from rclpy.duration import Duration
    stamp = (node.get_clock().now() + Duration(seconds=5.0)).to_msg()
    node.process(feed(node, stamp), False)
    assert node.store.pins == []


def test_no_map_tf_no_pin(node):
    stamp = node.get_clock().now().to_msg()
    node.process(feed(node, stamp, with_tf=False), False)
    assert node.store.pins == []


def test_log_all_levels_same_key(node):
    for lvl in ('info', 'warn', 'error', 'debug', 'info', 'warn'):
        node.log(f'k{lvl}', 'x', lvl, period=0.0)
