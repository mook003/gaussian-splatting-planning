#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

from cv_bridge import CvBridge

import open3d as o3d
import open3d.visualization.rendering as rendering


def quaternion_to_rotation_matrix(x, y, z, w) -> np.ndarray:
    """
    ROS-кватернион (x, y, z, w) -> матрица вращения 3x3.
    Нормализацию на всякий случай тоже делаем.
    """
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

    # Стандартная формула
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


class SceneRendererNode(Node):
    """
    Узел:
      - грузит PLY-сцену;
      - поднимает OffscreenRenderer;
      - по PoseStamped рендерит картинку и публикует её в Image-топик.
    """

    def __init__(self):
        super().__init__("scene_renderer")

        # ---------------- Параметры ----------------
        self.declare_parameter("scene_ply_path", "")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("horizontal_fov_deg", 60.0)
        # нужен только для заполнения header.frame_id у Image
        self.declare_parameter("camera_frame_id", "camera")

        scene_ply_path: str = self.get_parameter("scene_ply_path").value
        width: int = int(self.get_parameter("image_width").value)
        height: int = int(self.get_parameter("image_height").value)
        fov_deg: float = float(self.get_parameter("horizontal_fov_deg").value)
        self.camera_frame_id: str = self.get_parameter("camera_frame_id").value

        if not scene_ply_path:
            self.get_logger().error("Parameter 'scene_ply_path' is empty")
            raise RuntimeError("scene_ply_path must be set")

        self.get_logger().info(f"Loading scene from: {scene_ply_path}")
        # --------- Загрузка PLY как point cloud/mesh ----------
        # Для Gaussian Splatting PLY большинство библиотек (в т.ч. Open3D)
        # увидят как point cloud с позициями + цветами.
        scene = o3d.io.read_point_cloud(scene_ply_path)
        if len(scene.points) == 0:
            self.get_logger().error("Loaded point cloud has 0 points")
            raise RuntimeError("Empty scene")

        if not scene.has_colors():
            self.get_logger().warn(
                "Point cloud has no colors, setting uniform gray"
            )
            scene.paint_uniform_color([0.7, 0.7, 0.7])

        self.scene_geometry = scene
        self.width = width
        self.height = height

        # ------------ Камера: intrinsics ------------
        # f = W / (2 * tan(FOV/2))
        fov_rad = math.radians(fov_deg)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx * (height / width)  # если хотим одинаковый FOV по вертикали
        cx = width / 2.0
        cy = height / 2.0

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        self.intrinsic = intrinsic

        # ------------ OffscreenRenderer ------------
        # OffscreenRenderer(width, height) -> render_to_image()
        self.renderer = rendering.OffscreenRenderer(width, height)

        # Материал для point cloud
        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 1.0
        self.material = material

        # Добавляем геометрию в сцену
        self.renderer.scene.add_geometry(
            "scene", self.scene_geometry, self.material
        )

        # Можно добавить простой свет, но для defaultUnlit он не обязателен

        # ------------ ROS интерфейс ------------
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "cinematic_view", 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, "camera/pose", self.pose_callback, 10
        )

        self.get_logger().info(
            f"SceneRenderer initialized "
            f"(W={width}, H={height}, FOV={fov_deg} deg)"
        )

    # ---------- Вспомогательные функции ----------

    def pose_to_extrinsic(self, pose: PoseStamped) -> np.ndarray:
        """
        ROS PoseStamped -> extrinsic (4x4, world -> camera).
        Предполагаем, что pose задаёт позу камеры в мировой системе.
        """
        p = pose.pose.position
        q = pose.pose.orientation

        R_wc = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        t_wc = np.array([p.x, p.y, p.z], dtype=np.float64)

        # extrinsic: world->camera
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R_cw
        extrinsic[:3, 3] = t_cw
        return extrinsic

    def render_from_pose(self, pose: PoseStamped) -> Optional[np.ndarray]:
        extrinsic = self.pose_to_extrinsic(pose)

        # Настройка камеры по intrinsics + extrinsic
        # setup_camera(intrinsics, extrinsic_matrix)
        self.renderer.setup_camera(self.intrinsic, extrinsic)

        # Offscreen рендер
        o3d_image = self.renderer.render_to_image()
        if o3d_image is None:
            self.get_logger().warn("render_to_image() returned None")
            return None

        np_image = np.asarray(o3d_image)  # H x W x 3, uint8, RGB
        return np_image

    # ---------- Callback ----------

    def pose_callback(self, msg: PoseStamped):
        img_np = self.render_from_pose(msg)
        if img_np is None:
            return

        # Преобразуем в ROS Image через cv_bridge (encoding: rgb8)
        ros_image = self.bridge.cv2_to_imgmsg(img_np, encoding="rgb8")
        ros_image.header.stamp = msg.header.stamp
        ros_image.header.frame_id = self.camera_frame_id

        self.image_pub.publish(ros_image)


def main(args=None):
    rclpy.init(args=args)
    node = SceneRendererNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
