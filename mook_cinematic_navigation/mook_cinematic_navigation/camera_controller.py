#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_from_quaternion(q):
    """Вытащить yaw (угол вокруг Z) из geometry_msgs/Quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class CameraSimulator(Node):
    """
    Простейший 2D-симулятор:
    - хранит (x, y, yaw) в системе координат 'map'
    - слушает /initialpose и /cmd_vel (по умолчанию)
    - публикует TF: map -> camera
    """

    def __init__(self):
        super().__init__("camera_simulator")

        # --- Параметры ---
        self.declare_parameter("frame_id", "map")          # родительский фрейм
        self.declare_parameter("child_frame_id", "camera") # фрейм робота
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("publish_frequency", 50.0)

        self.declare_parameter("initial_x", 0.0)
        self.declare_parameter("initial_y", 0.0)
        self.declare_parameter("initial_yaw", 0.0)  # рад

        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.publish_frequency = float(self.get_parameter("publish_frequency").value)

        # --- Состояние позы робота ---
        self.x = float(self.get_parameter("initial_x").value)
        self.y = float(self.get_parameter("initial_y").value)
        self.yaw = float(self.get_parameter("initial_yaw").value)

        # Последняя скорость (в СК робота)
        self.last_vx = 0.0
        self.last_vy = 0.0   # На случай holonomic, но Nav2 обычно даёт 0
        self.last_wz = 0.0

        self.last_time = self.get_clock().now()

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Подписки ---
        # 1) Команды скорости от Nav2
        self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_vel_callback,
            10,
        )

        # 2) Начальная поза из RViz (2D Pose Estimate)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "initialpose",
            self.initialpose_callback,
            10,
        )

        # --- Таймер интеграции и публикации TF ---
        period = 1.0 / self.publish_frequency if self.publish_frequency > 0.0 else 0.02
        self.timer = self.create_timer(period, self.update)

        self.get_logger().info(
            f"CameraSimulator started. frame_id={self.frame_id}, "
            f"child_frame_id={self.child_frame_id}, cmd_vel_topic={self.cmd_vel_topic}"
        )
        self.get_logger().info(
            f"Initial pose: x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f} rad"
        )

    # -------------------- Callbacks --------------------

    def cmd_vel_callback(self, msg: Twist):
        """Сохраняем последнюю команду скорости от Nav2."""
        self.last_vx = msg.linear.x
        self.last_vy = msg.linear.y
        self.last_wz = msg.angular.z

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Перехватываем 2D Pose Estimate из RViz и
        выставляем абсолютную позу камеры в frame_id (обычно 'map').
        """
        # Можно проверить frame_id, но для простоты считаем, что это "map"
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        self.x = pos.x
        self.y = pos.y
        self.yaw = yaw_from_quaternion(ori)

        # Сбрасываем время, чтобы интеграция скоростей продолжилась от этой позы
        self.last_time = self.get_clock().now()

        self.get_logger().info(
            f"Initial pose received from /initialpose: "
            f"x={self.x:.3f}, y={self.y:.3f}, yaw={self.yaw:.3f} rad"
        )

    # -------------------- Main loop --------------------

    def update(self):
        """Интегрируем движение и публикуем TF."""
        now = self.get_clock().now()
        dt = (now.nanoseconds - self.last_time.nanoseconds) * 1e-9
        if dt <= 0.0 or dt > 1.0:  # защитимся от странных прыжков во времени
            dt = 0.0
        self.last_time = now

        # Интеграция скоростей (vx, vy, wz) из СК робота в мировую (map)
        if dt > 0.0:
            vx = self.last_vx
            vy = self.last_vy
            wz = self.last_wz

            # Скорости в мировой системе координат
            v_world_x = vx * math.cos(self.yaw) - vy * math.sin(self.yaw)
            v_world_y = vx * math.sin(self.yaw) + vy * math.cos(self.yaw)

            self.x += v_world_x * dt
            self.y += v_world_y * dt
            self.yaw += wz * dt

            # Нормализуем yaw, чтобы он не убегал куда-то в ±∞
            self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        # Публикуем TF map -> camera
        self.publish_tf(now)

    def publish_tf(self, now):
        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.child_frame_id

        t.transform.translation.x = float(self.x)
        t.transform.translation.y = float(self.y)
        t.transform.translation.z = 0.0

        qz = math.sin(self.yaw / 2.0)
        qw = math.cos(self.yaw / 2.0)
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = CameraSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
