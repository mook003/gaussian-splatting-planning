#!/usr/bin/env python3
import math
from typing import Optional

import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering

import rclpy
from rclpy.node import Node
from rclpy.time import Time

from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

from tf2_ros import Buffer, TransformListener, TransformException


def quaternion_to_rotation_matrix(x, y, z, w) -> np.ndarray:
    """ROS-кватернион (x, y, z, w) -> матрица 3x3."""
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm

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
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


class SceneRendererNode(Node):
    """
    Узел:
      - грузит PLY-сцену;
      - приводит её в ту же систему координат, что и OccupancyGrid (PLY->map);
      - по TF (map->camera_frame) рендерит изображение и публикует его в Image-топик.
    """

    def __init__(self):
        super().__init__("scene_renderer")

        # ----- Параметры -----
        self.declare_parameter("scene_ply_path", "")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("horizontal_fov_deg", 60.0)

        # TF
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("camera_frame", "camera")
        # camera_frame похож на base_link (X вперёд, Y влево, Z вверх)?
        self.declare_parameter("camera_frame_is_baselink", True)

        # те же флаги, что и в ply_to_occupancy
        self.declare_parameter("flip_x", False)
        self.declare_parameter("flip_y", False)
        self.declare_parameter("flip_z", False)

        # какой frame_id писать в Image
        self.declare_parameter("image_frame_id", "camera")

        scene_ply_path: str = self.get_parameter("scene_ply_path").value
        width: int = int(self.get_parameter("image_width").value)
        height: int = int(self.get_parameter("image_height").value)
        fov_deg: float = float(self.get_parameter("horizontal_fov_deg").value)

        self.map_frame: str = self.get_parameter("map_frame").value
        self.camera_frame: str = self.get_parameter("camera_frame").value
        self.camera_frame_is_baselink: bool = bool(
            self.get_parameter("camera_frame_is_baselink").value
        )

        flip_x = bool(self.get_parameter("flip_x").value)
        flip_y = bool(self.get_parameter("flip_y").value)
        flip_z = bool(self.get_parameter("flip_z").value)

        self.image_frame_id: str = self.get_parameter("image_frame_id").value

        if not scene_ply_path:
            self.get_logger().error("Parameter 'scene_ply_path' is empty")
            raise RuntimeError("scene_ply_path must be set")

        # ----- Загрузка и трансформ PLY -> map -----
        self.get_logger().info(f"Loading scene from: {scene_ply_path}")
        pcd_raw = o3d.io.read_point_cloud(scene_ply_path)
        if len(pcd_raw.points) == 0:
            self.get_logger().error("Loaded point cloud has 0 points")
            raise RuntimeError("Empty scene")

        pts_raw = np.asarray(pcd_raw.points, dtype=np.float64)
        pts_map = self.ply_to_map_coords(pts_raw, flip_x, flip_y, flip_z)

        # создаём новое облако уже в координатах map
        pcd_map = o3d.geometry.PointCloud()
        pcd_map.points = o3d.utility.Vector3dVector(pts_map)

        if pcd_raw.has_colors():
            # цвета просто переносим как есть
            pcd_map.colors = pcd_raw.colors
        else:
            self.get_logger().warn(
                "Point cloud has no colors, setting uniform gray"
            )
            pcd_map.paint_uniform_color([0.7, 0.7, 0.7])

        self.scene_geometry = pcd_map
        self.width = width
        self.height = height

        # ----- Камера: intrinsics -----
        fov_rad = math.radians(fov_deg)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx * (height / width)
        cx = width / 2.0
        cy = height / 2.0

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)
        self.intrinsic = intrinsic

        # ----- OffscreenRenderer -----
        self.renderer = rendering.OffscreenRenderer(width, height)

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 1.0
        self.material = material

        self.renderer.scene.add_geometry("scene", self.scene_geometry, self.material)

        # ----- TF -----
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ----- ROS интерфейс -----
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(RosImage, "cinematic_view", 10)

        # таймер рендера (30 FPS условно)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info(
            f"SceneRenderer initialized "
            f"(W={width}, H={height}, FOV={fov_deg} deg, "
            f"map_frame={self.map_frame}, camera_frame={self.camera_frame})"
        )

        # базовая матрица B->C (base_link -> optical)
        # base: X вперед, Y влево, Z вверх
        # cam:  X вправо, Y вниз, Z вперед
        self.R_cb = np.array(
            [
                [0.0, -1.0, 0.0],  # X_cam
                [0.0,  0.0, -1.0], # Y_cam
                [1.0,  0.0,  0.0], # Z_cam
            ],
            dtype=np.float64,
        )

    # ---------- Общий PLY->map трансформ (как в ply_to_occupancy) ----------

    @staticmethod
    def ply_to_map_coords(pts_raw: np.ndarray,
                          flip_x: bool,
                          flip_y: bool,
                          flip_z: bool) -> np.ndarray:
        pts = pts_raw.copy()
        if flip_x:
            pts[:, 0] *= -1.0
        if flip_y:
            pts[:, 1] *= -1.0
        if flip_z:
            pts[:, 2] *= -1.0

        Xp = pts[:, 0]
        Yp = pts[:, 1]
        Zp = pts[:, 2]

        pts_map = np.zeros_like(pts)
        pts_map[:, 0] = Xp   # X_map
        pts_map[:, 1] = Zp   # Y_map (горизонталь)
        pts_map[:, 2] = Yp   # Z_map (высота)

        return pts_map

    # ---------- Камера: extrinsic из TF ----------

    def get_extrinsic_from_tf(self) -> Optional[np.ndarray]:
        """
        Строим extrinsic world->camera для Open3D из TF:
          - world = map_frame
          - "camera_frame" даёт позу Base (base_link-подобный) в map
          - если camera_frame_is_baselink=True, добавляем поворот Base->Optical
        """
        try:
            # transform: map_frame <- camera_frame
            # т.е. p_map = R_wb * p_camBase + t_wb
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.camera_frame,
                Time()
            )
        except TransformException as ex:
            # чтобы не спамить, логируем не слишком часто
            self.get_logger().warn(
                f"TF map->{self.camera_frame} not available yet: {ex}"
            )
            return None

        t = transform.transform.translation
        q = transform.transform.rotation

        t_wb = np.array([t.x, t.y, t.z], dtype=np.float64)
        R_wb = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)

        # R_wb: B -> W (p_w = R_wb * p_b)
        R_bw = R_wb.T  # W -> B

        if self.camera_frame_is_baselink:
            # B -> C (base_link -> optical)
            R_cb = self.R_cb
        else:
            # предполагаем, что camera_frame уже оптический
            R_cb = np.eye(3, dtype=np.float64)

        # world -> camera:
        # p_c = R_cb * R_bw * (p_w - t_wb)
        R_cw = R_cb @ R_bw
        t_cw = -R_cw @ t_wb

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R_cw
        extrinsic[:3, 3] = t_cw
        return extrinsic

    # ---------- Рендер ----------

    def timer_callback(self):
        extrinsic = self.get_extrinsic_from_tf()
        if extrinsic is None:
            return

        self.renderer.setup_camera(self.intrinsic, extrinsic)

        o3d_image = self.renderer.render_to_image()
        if o3d_image is None:
            self.get_logger().warn("render_to_image() returned None")
            return

        np_image = np.asarray(o3d_image)  # H x W x 3, uint8, RGB

        ros_image = self.bridge.cv2_to_imgmsg(np_image, encoding="rgb8")
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = self.image_frame_id

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
