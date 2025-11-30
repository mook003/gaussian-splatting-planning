#!/usr/bin/env python3
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

import numpy as np
import cv2


class VideoRecorderNode(Node):
    def __init__(self):
        super().__init__('video_recorder')

        self.declare_parameter('output_path', 'cinematic_output.mp4')
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('video_duration_sec', 60.0)

        self.output_path: str = self.get_parameter('output_path').value
        self.fps: float = float(self.get_parameter('fps').value)
        self.duration_sec: float = float(
            self.get_parameter('video_duration_sec').value
        )

        self.max_frames = int(self.fps * self.duration_sec)
        self.frame_count = 0
        self.writer = None

        os.makedirs(os.path.dirname(self.output_path) or '.', exist_ok=True)

        self.subscription = self.create_subscription(
            Image, 'cinematic_view', self.image_callback, 10
        )

        self.get_logger().info(
            f"VideoRecorder initialized: path={self.output_path}, "
            f"fps={self.fps}, duration={self.duration_sec}s "
            f"(max_frames={self.max_frames})"
        )

    def init_writer(self, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (width, height)
        )
        if not self.writer.isOpened():
            self.get_logger().error("Failed to open VideoWriter")
            self.writer = None
        else:
            self.get_logger().info(
                f"VideoWriter opened: {width}x{height} @ {self.fps}fps"
            )

    def image_callback(self, msg: Image):
        if self.writer is None:
            width = msg.width
            height = msg.height
            self.init_writer(width, height)
            if self.writer is None:
                return

        # msg.encoding = "rgb8" из рендерера
        np_arr = np.frombuffer(msg.data, dtype=np.uint8)
        frame = np_arr.reshape((msg.height, msg.width, 3))

        # OpenCV ожидает BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        self.writer.write(frame_bgr)
        self.frame_count += 1

        if self.frame_count >= self.max_frames:
            self.get_logger().info(
                f"Reached max_frames={self.max_frames}, closing video."
            )
            self.writer.release()
            self.writer = None
            # Можно при желании завершить ноду:
            # rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.writer is not None:
        node.writer.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
