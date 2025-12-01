#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import List, Dict

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from std_srvs.srv import Trigger

from tf2_ros import Buffer, TransformListener, TransformException


class CameraPathRecorder(Node):
    """
    Нода:
      - слушает TF (map -> camera),
      - с заданной частотой пишет позы в список,
      - по сервису сохраняет траекторию в JSON.
    """

    def __init__(self):
        super().__init__("camera_path_recorder")

        # -------- Параметры --------
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "camera")
        self.declare_parameter("sample_hz", 10.0)
        self.declare_parameter("output_path", "paths/camera_path.json")

        self.map_frame: str = self.get_parameter("map_frame").value
        self.camera_frame: str = self.get_parameter("camera_frame").value
        self.sample_hz: float = float(self.get_parameter("sample_hz").value)
        self.output_path: Path = Path(self.get_parameter("output_path").value)

        if self.sample_hz <= 0.0:
            self.sample_hz = 10.0

        self.get_logger().info(
            f"CameraPathRecorder: map_frame={self.map_frame}, "
            f"camera_frame={self.camera_frame}, sample_hz={self.sample_hz}, "
            f"output_path={self.output_path}"
        )

        # TF
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Путь: список поз
        self.poses: List[Dict] = []
        self.recording: bool = True

        # Таймер выборки
        period = 1.0 / self.sample_hz
        self.timer = self.create_timer(period, self.timer_cb)

        # Сервисы
        self.srv_save = self.create_service(
            Trigger, "save_camera_path", self.handle_save_path
        )
        self.srv_reset = self.create_service(
            Trigger, "reset_camera_path", self.handle_reset_path
        )

    # -------- Основной цикл --------

    def timer_cb(self):
        if not self.recording:
            return

        try:
            # Берём "последний доступный" трансформ
            now = Time()
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                now,
                timeout=Duration(seconds=0.1),
            )
        except TransformException as ex:
            self.get_logger().debug(f"TF lookup failed: {ex}")
            return

        tr = tf.transform.translation
        rot = tf.transform.rotation
        stamp = tf.header.stamp

        t_sec = stamp.sec + stamp.nanosec * 1e-9

        pose_dict = {
            "t": t_sec,
            "frame_id": tf.header.frame_id,
            "child_frame_id": tf.child_frame_id,
            "x": tr.x,
            "y": tr.y,
            "z": tr.z,
            "qx": rot.x,
            "qy": rot.y,
            "qz": rot.z,
            "qw": rot.w,
        }
        self.poses.append(pose_dict)

    # -------- Сервисы --------

    def handle_save_path(self, request, response):
        if not self.poses:
            response.success = False
            response.message = "No poses recorded yet"
            return response

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "map_frame": self.map_frame,
            "camera_frame": self.camera_frame,
            "sample_hz": self.sample_hz,
            "num_poses": len(self.poses),
            "poses": self.poses,
        }

        with self.output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        response.success = True
        response.message = f"Saved {len(self.poses)} poses to {self.output_path}"
        self.get_logger().info(response.message)
        return response

    def handle_reset_path(self, request, response):
        n = len(self.poses)
        self.poses.clear()
        response.success = True
        response.message = f"Cleared in-memory path (removed {n} poses)"
        self.get_logger().info(response.message)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = CameraPathRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
