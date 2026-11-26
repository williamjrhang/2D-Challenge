"""Record the obstacle detector's color camera while the robot is driven by hand."""

import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkstemp

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class CameraRecorder(Node):

    def __init__(self):
        super().__init__('camera_recorder')
        self.image_topic = self.declare_parameter(
            'image_topic', '/camera/color/image_raw/compressed').value
        self.output_dir = Path(self.declare_parameter(
            'output_dir', '~/challenge_ws/frames/recordings').value).expanduser().resolve()
        self.playback_fps = self.declare_parameter('playback_fps', 30.0).value
        if not math.isfinite(self.playback_fps) or self.playback_fps <= 0:
            raise ValueError('playback_fps must be positive and finite')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writer = None
        self.video_path = None
        self.frame_size = None
        self.frames_written = 0
        self.create_subscription(CompressedImage, self.image_topic, self.on_image,
                                 qos_profile_sensor_data)
        self.wait_timer = self.create_timer(5.0, self.report_waiting)
        self.get_logger().info(
            f'Waiting for images on {self.image_topic}. Videos go to {self.output_dir}. '
            'Press Ctrl+C to stop recording.')

    def report_waiting(self):
        self.get_logger().warning(f'Waiting for a decodable image on {self.image_topic}')

    def on_image(self, msg):
        try:
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        except cv2.error:
            frame = None
        if frame is None:
            self.get_logger().warning('Skipping an unreadable camera image',
                                      throttle_duration_sec=5.0)
            return

        height, width = frame.shape[:2]
        if self.writer is None:
            self.open_video(width, height)
        elif (width, height) != self.frame_size:
            recording_width, recording_height = self.frame_size
            self.get_logger().warning(
                f'Skipping a {width}x{height} image. '
                f'Recording size is {recording_width}x{recording_height}.',
                throttle_duration_sec=5.0)
            return

        # Pad odd dimensions so the encoder preserves the last row and column.
        if width % 2 or height % 2:
            frame = cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2, cv2.BORDER_REPLICATE)
        self.writer.write(frame)
        self.frames_written += 1

    def open_video(self, width, height):
        prefix = datetime.now(timezone.utc).strftime('camera_%Y%m%d_%H%M%S_')
        descriptor, filename = mkstemp(prefix=prefix, suffix='.mp4', dir=self.output_dir)
        os.close(descriptor)
        path = Path(filename)
        writer = cv2.VideoWriter()
        try:
            if not writer.open(str(path), cv2.VideoWriter_fourcc(*'mp4v'), self.playback_fps,
                               (width + width % 2, height + height % 2)):
                raise OSError(f'Could not open {path} for MP4 recording')
        except (OSError, cv2.error, KeyboardInterrupt):
            writer.release()
            path.unlink(missing_ok=True)
            raise

        self.video_path = path
        self.frame_size = (width, height)
        self.writer = writer
        self.wait_timer.cancel()
        self.get_logger().info(f'Recording to {path} at {self.playback_fps:g} fps for playback')

    def finish(self):
        saved = 0
        if self.writer is not None:
            self.writer.release()
            video = cv2.VideoCapture(str(self.video_path))
            try:
                saved = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                video.release()
            if saved < self.frames_written:
                raise OSError(
                    f'{self.video_path} contains {saved} of {self.frames_written} frames')
            if not saved:
                self.video_path.unlink(missing_ok=True)

        if saved:
            print(f'Saved {saved} frames to {self.video_path}', flush=True)
        else:
            print(f'No camera frames saved from {self.image_topic}', flush=True)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = CameraRecorder()
        try:
            rclpy.spin(node)
        finally:
            try:
                node.finish()
            finally:
                node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (OSError, ValueError, cv2.error) as exc:
        print(f'Camera recorder failed. {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        rclpy.try_shutdown()
    return 0
