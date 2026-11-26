#!/usr/bin/env python3
"""Robot-side relay: what the desktop (192.168.10.105) gets from the camera (issue 9).

Runs ON THE ROBOT, next to the camera driver, so the heavy topics never cross the link:
  colour  <image_in>  (compressed, ~30 Hz)  -> <image_out> at image_rate_hz (5 Hz)
  depth   <depth_in>  (640x480 cloud, ~4.4 MB) -> <depth_out>: cropped to depth_max_m and
          thinned to depth_max_points (~0.3 MB), at depth_rate_hz, ONLY while someone on
          the desktop subscribes to it. The raw cloud is grabbed one message at a time
          (subscribe, take one, unsubscribe), never streamed.

Rate guard (link_guard.py): while /livox/imu < imu_min_hz or FAST-LIO odometry
< lidar_min_hz (the robot's navigation is starving), no depth at all and the colour image
drops to guard_image_hz, until the rates are healthy again for recover_s.
Light: one Python process, numpy striding only, no image decoding.

Topics
  in   image_in, depth_in, /livox/imu, /fastlio/odometry
  out  image_out, depth_out
  out  /link_relay/status   std_msgs/String  'ok' or 'degraded: <reason>' (latched)
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage, Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from lite3_explore_pose.link_guard import LinkGuard, Throttle


def thin_cloud(xyz, max_points, max_range_m):
    """Drop NaNs and points beyond max_range_m (camera z), then stride down to max_points."""
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if max_range_m > 0:
        xyz = xyz[(xyz[:, 2] > 0.0) & (xyz[:, 2] <= max_range_m)]
    if max_points > 0 and len(xyz) > max_points:
        xyz = xyz[::int(np.ceil(len(xyz) / max_points))]
    return xyz.astype(np.float32)


class LinkRelay(Node):

    def __init__(self):
        super().__init__('link_relay')
        p = self.declare_parameter
        p('image_in', '/camera/color/image_raw/compressed')
        p('image_out', '/camera/color/image_throttled/compressed')
        p('image_rate_hz', 5.0)
        p('depth_in', '/camera/depth/points')
        p('depth_out', '/camera/depth/points_small')
        p('depth_rate_hz', 2.0)
        p('depth_max_points', 25000)
        p('depth_max_m', 3.0)
        p('imu_topic', '/livox/imu')
        p('lidar_topic', '/fastlio/odometry')
        p('imu_min_hz', 150.0)
        p('lidar_min_hz', 8.0)
        p('guard_window_s', 2.0)
        p('guard_recover_s', 5.0)
        p('guard_image_hz', 1.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.P = {n: g(n) for n in [
            'image_in', 'image_out', 'image_rate_hz', 'depth_in', 'depth_out',
            'depth_rate_hz', 'depth_max_points', 'depth_max_m', 'imu_topic', 'lidar_topic',
            'imu_min_hz', 'lidar_min_hz', 'guard_window_s', 'guard_recover_s',
            'guard_image_hz']}
        self.guard = LinkGuard(self.P['imu_min_hz'], self.P['lidar_min_hz'],
                               self.P['guard_recover_s'], self.P['guard_window_s'])
        self.img_throttle = Throttle()
        self.depth_throttle = Throttle()
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/link_relay/status', latched)
        self.img_pub = self.create_publisher(CompressedImage, self.P['image_out'],
                                             qos_profile_sensor_data)
        self.depth_pub = self.create_publisher(PointCloud2, self.P['depth_out'],
                                               qos_profile_sensor_data)
        self.create_subscription(CompressedImage, self.P['image_in'], self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(Imu, self.P['imu_topic'],
                                 lambda m: self.guard.imu.tick(self.now_s()),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, self.P['lidar_topic'],
                                 lambda m: self.guard.lidar.tick(self.now_s()), 20)
        self.depth_sub = None
        self.create_timer(0.1, self.tick)
        self.status_pub.publish(String(data=f'degraded: {self.guard.reason}'))
        self.get_logger().info(
            f'link_relay: {self.P["image_in"]} -> {self.P["image_out"]} at '
            f'{self.P["image_rate_hz"]} Hz; {self.P["depth_in"]} -> {self.P["depth_out"]} '
            f'(<= {self.P["depth_max_points"]} points, {self.P["depth_rate_hz"]} Hz, only '
            f'while subscribed); guard /livox/imu >= {self.P["imu_min_hz"]} Hz, lidar >= '
            f'{self.P["lidar_min_hz"]} Hz')

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def tick(self):
        now = self.now_s()
        degraded, changed, reason = self.guard.update(now)
        if changed:
            if degraded:
                self.get_logger().error(f'link_relay DEGRADED (no depth, image '
                                        f'{self.P["guard_image_hz"]} Hz): {reason}')
            else:
                self.get_logger().warn(f'link_relay back to normal: {reason}')
            self.status_pub.publish(String(data=('degraded: ' + reason) if degraded else 'ok'))
        # Depth: grab ONE raw cloud per period, only while the desktop wants it.
        wanted = self.depth_pub.get_subscription_count() > 0 and not degraded
        if (wanted and self.depth_sub is None
                and self.depth_throttle.ok(now, self.P['depth_rate_hz'])):
            self.depth_sub = self.create_subscription(PointCloud2, self.P['depth_in'],
                                                      self.on_depth, qos_profile_sensor_data)

    def on_image(self, msg):
        rate = self.P['guard_image_hz'] if self.guard.degraded else self.P['image_rate_hz']
        if self.img_throttle.ok(self.now_s(), rate):
            self.img_pub.publish(msg)

    def on_depth(self, msg):
        if self.depth_sub is not None:            # one message, then unsubscribe
            self.destroy_subscription(self.depth_sub)
            self.depth_sub = None
        if self.guard.degraded:
            return
        try:
            xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'),
                                                 skip_nans=True)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f'depth cloud unreadable: {e!r}', throttle_duration_sec=10.0)
            return
        pts = thin_cloud(np.asarray(xyz, dtype=np.float64), self.P['depth_max_points'],
                         self.P['depth_max_m'])
        self.depth_pub.publish(point_cloud2.create_cloud_xyz32(msg.header, pts))


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = LinkRelay()
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
