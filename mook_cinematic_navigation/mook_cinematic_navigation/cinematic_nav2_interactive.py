#!/usr/bin/env python3
import math
import time
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.time import Time as RclpyTime

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from nav2_simple_commander.robot_navigator import BasicNavigator


# === Вспомогательные функции ===

def euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def path_to_xy(path: Path) -> np.ndarray:
    pts = []
    for pose in path.poses:
        pts.append([pose.pose.position.x, pose.pose.position.y])
    return np.array(pts, dtype=np.float64)


def resample_path_xy(xy: np.ndarray, step: float):
    if xy.shape[0] < 2:
        s = np.zeros((xy.shape[0],), dtype=np.float64)
        return xy, s

    seg = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    s = np.zeros((xy.shape[0],), dtype=np.float64)
    s[1:] = np.cumsum(seg)
    total_len = s[-1]

    if total_len <= 0.0:
        return xy, s

    num_samples = int(total_len / step) + 1
    s_new = np.linspace(0.0, total_len, num_samples)

    xy_res = np.zeros((num_samples, 2), dtype=np.float64)
    j = 0
    for i in range(num_samples):
        si = s_new[i]
        while j + 1 < len(s) and s[j + 1] < si:
            j += 1
        if j + 1 == len(s):
            xy_res[i] = xy[-1]
        else:
            denom = max(s[j + 1] - s[j], 1e-6)
            u = (si - s[j]) / denom
            xy_res[i] = (1.0 - u) * xy[j] + u * xy[j + 1]

    return xy_res, s_new


def unwrap_angles(yaws: np.ndarray) -> np.ndarray:
    return np.unwrap(yaws)


def build_cinematic_poses(
    path: Path,
    map_frame: str,
    camera_height: float,
    look_at: Tuple[float, float, float],
    fps: float,
    speed: float,
) -> List[PoseStamped]:
    xy = path_to_xy(path)
    if xy.shape[0] == 0:
        return []

    step = max(speed / fps, 1e-3)
    xy_res, _ = resample_path_xy(xy, step)

    # yaw по касательной к пути
    tangents = np.zeros_like(xy_res)
    tangents[:-1] = xy_res[1:] - xy_res[:-1]
    tangents[-1] = tangents[-2]
    yaws = np.arctan2(tangents[:, 1], tangents[:, 0])
    yaws = unwrap_angles(yaws)

    look_x, look_y, look_z = look_at

    poses: List[PoseStamped] = []
    dt = 1.0 / fps

    for i in range(xy_res.shape[0]):
        x = float(xy_res[i, 0])
        y = float(xy_res[i, 1])
        z = camera_height

        # вектор на цель
        dx = look_x - x
        dy = look_y - y
        dz = look_z - z
        r_xy = math.sqrt(dx * dx + dy * dy) + 1e-6

        yaw_look = math.atan2(dy, dx)
        pitch_look = math.atan2(-dz, r_xy)

        # смешиваем "смотрим по пути" и "смотрим на цель"
        alpha = 0.7  # 1.0 — чистый look-at, 0.0 — чисто по пути
        yaw_path = yaws[i]
        yaw = alpha * yaw_look + (1.0 - alpha) * yaw_path
        roll = 0.0

        qx, qy, qz, qw = euler_to_quat(roll, pitch_look, yaw)

        pose = PoseStamped()
        pose.header.frame_id = map_frame
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        poses.append(pose)

    return poses


# === Основной класс ===

class CinematicNavigator(BasicNavigator):
    def __init__(self):
        super().__init__()

        # Параметры камеры
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_height", 1.5)
        self.declare_parameter("look_at_x", 0.0)
        self.declare_parameter("look_at_y", 0.0)
        self.declare_parameter("look_at_z", 1.2)
        self.declare_parameter("camera_speed", 0.5)  # м/с
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("goal_topic", "/goal_pose")         # 2D Nav Goal
        self.declare_parameter("initialpose_topic", "/initialpose")

        self.map_frame = self.get_parameter("map_frame").value
        self.camera_height = float(self.get_parameter("camera_height").value)
        self.look_at_x = float(self.get_parameter("look_at_x").value)
        self.look_at_y = float(self.get_parameter("look_at_y").value)
        self.look_at_z = float(self.get_parameter("look_at_z").value)
        self.camera_speed = float(self.get_parameter("camera_speed").value)
        self.fps = float(self.get_parameter("fps").value)

        goal_topic = self.get_parameter("goal_topic").value
        initialpose_topic = self.get_parameter("initialpose_topic").value

        # Паблишеры
        self.camera_pose_pub = self.create_publisher(PoseStamped, "camera_pose", 10)
        self.camera_path_pub = self.create_publisher(Path, "camera_path", 10)

        # Подписчики
        self.goal_sub = self.create_subscription(
            PoseStamped, goal_topic, self.goal_callback, 10
        )
        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped, initialpose_topic, self.initialpose_callback, 10
        )

        # "Текущая" позиция камеры для старта следующего пути
        self.current_pose: PoseStamped = None

        self.get_logger().info("Waiting for Nav2 to become active...")
        self.waitUntilNav2Active()
        self.get_logger().info("Nav2 is active. Ready for goals from RViz.")

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        # Используем initialpose из RViz для установки начальной позиции камеры
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.current_pose = pose
        self.get_logger().info(
            f"Initial camera pose set from /initialpose at "
            f"({pose.pose.position.x:.2f}, {pose.pose.position.y:.2f})"
        )

    def goal_callback(self, goal: PoseStamped):
        self.get_logger().info(
            f"Received goal from RViz: "
            f"({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})"
        )

        # Если ещё нет текущей позы камеры — стартуем из goal (камера "телепортируется")
        if self.current_pose is None:
            self.current_pose = PoseStamped()
            self.current_pose.header.frame_id = self.map_frame
            self.current_pose.header.stamp = self.get_clock().now().to_msg()
            self.current_pose.pose.position.x = goal.pose.position.x
            self.current_pose.pose.position.y = goal.pose.position.y
            self.current_pose.pose.position.z = 0.0
            self.current_pose.pose.orientation = goal.pose.orientation
            self.get_logger().warn(
                "current_pose was None, using first goal as start pose"
            )

        # Обновляем frame_id/время
        start = PoseStamped()
        start.header.frame_id = self.map_frame
        start.header.stamp = self.get_clock().now().to_msg()
        start.pose = self.current_pose.pose

        g = PoseStamped()
        g.header.frame_id = self.map_frame
        g.header.stamp = self.get_clock().now().to_msg()
        g.pose = goal.pose

        # 1. Получаем путь от Nav2
        nav_path = self.getPath(start, g)
        if nav_path is None or len(nav_path.poses) == 0:
            self.get_logger().error("Nav2 getPath() returned empty path")
            return

        smooth_path = self.smoothPath(nav_path)
        if smooth_path is None or len(smooth_path.poses) == 0:
            self.get_logger().warn("smoothPath() empty, using original path")
            smooth_path = nav_path

        self.get_logger().info(
            f"Got nav2 path: {len(smooth_path.poses)} poses, building cinematic trajectory"
        )

        # 2. Кинематографический путь камеры
        cinematic_poses = build_cinematic_poses(
            path=smooth_path,
            map_frame=self.map_frame,
            camera_height=self.camera_height,
            look_at=(self.look_at_x, self.look_at_y, self.look_at_z),
            fps=self.fps,
            speed=self.camera_speed,
        )

        if not cinematic_poses:
            self.get_logger().error("Cinematic path is empty")
            return

        # 3. Публикуем Path для RViz
        path_msg = Path()
        path_msg.header.frame_id = self.map_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = cinematic_poses
        self.camera_path_pub.publish(path_msg)

        # 4. Проигрываем траекторию камеры (предпросмотр)
        dt = 1.0 / self.fps
        for pose in cinematic_poses:
            if not rclpy.ok():
                break
            pose.header.stamp = self.get_clock().now().to_msg()
            self.camera_pose_pub.publish(pose)
            time.sleep(dt)

        # Обновляем current_pose на конец траектории
        self.current_pose = cinematic_poses[-1]
        self.get_logger().info("Cinematic trajectory playback completed.")


def main(args=None):
    rclpy.init(args=args)
    node = CinematicNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroyNode()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
