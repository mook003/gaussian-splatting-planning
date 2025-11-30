#!/usr/bin/env python3
import json

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class CameraPathRecorderNode(Node):
    def __init__(self):
        super().__init__("camera_path_recorder")

        # Параметры
        self.declare_parameter("output_path", "camera_path.json")
        self.declare_parameter("sample_hz", 30.0)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("horizontal_fov_deg", 70.0)

        self.output_path = str(self.get_parameter("output_path").value)
        self.sample_hz = float(self.get_parameter("sample_hz").value)
        self.sample_period = 1.0 / self.sample_hz

        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.hfov_deg = float(self.get_parameter("horizontal_fov_deg").value)

        self.frames = []
        self.start_time = None
        self.last_sample_time = None

        self.subscription = self.create_subscription(
            PoseStamped, "camera/pose", self.pose_callback, 10
        )

        self.get_logger().info(
            f"CameraPathRecorderNode started, writing to {self.output_path}"
        )

    def pose_callback(self, msg: PoseStamped):
        now = self.get_clock().now()

        if self.start_time is None:
            self.start_time = now
            self.last_sample_time = now

        dt = (now - self.last_sample_time).nanoseconds / 1e9
        if dt < self.sample_period:
            return

        self.last_sample_time = now
        t = (now - self.start_time).nanoseconds / 1e9

        p = msg.pose.position
        q = msg.pose.orientation

        self.frames.append(
            {
                "t": t,
                "position": [p.x, p.y, p.z],
                "orientation_xyzw": [q.x, q.y, q.z, q.w],
            }
        )

    def save_json(self):
        data = {
            "image_width": self.image_width,
            "image_height": self.image_height,
            "horizontal_fov_deg": self.hfov_deg,
            "frames": self.frames,
        }
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
        self.get_logger().info(
            f"Saved {len(self.frames)} frames to {self.output_path}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraPathRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_json()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
