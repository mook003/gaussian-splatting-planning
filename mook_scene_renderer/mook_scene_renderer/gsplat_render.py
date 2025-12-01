#!/usr/bin/env python3
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

# ---------- Вспомогательные структуры ----------


@dataclass
class GsplatScene:
    means: torch.Tensor      # [N, 3]
    quats: torch.Tensor      # [N, 4]
    scales: torch.Tensor     # [N, 3]
    opacities: torch.Tensor  # [N]
    colors: torch.Tensor     # [N, 3]

    def to(self, device: torch.device) -> "GsplatScene":
        return GsplatScene(
            means=self.means.to(device),
            quats=self.quats.to(device),
            scales=self.scales.to(device),
            opacities=self.opacities.to(device),
            colors=self.colors.to(device),
        )


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """
    Кватернион (x, y, z, w) -> матрица вращения 3x3.
    """
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np.eye(3, dtype=np.float32)

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
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    return R


# ---------- Основной ROS2 узел ----------


class GsplatCinematicNode(Node):
    """
    Узел:
      - читает заранее записанный JSON траектории камеры (CameraPathRecorder),
      - загружает 3D Gaussian Splatting .ply сцену,
      - по сервису /render_cinematic рендерит видео через gsplat и сохраняет на диск.
    """

    def __init__(self):
        super().__init__("gsplat_render")

        # -------- Параметры --------
        self.declare_parameter("scene_path", "src/data/ConferenceHall.ply")
        self.declare_parameter("camera_path_json", "paths/camera_path.json")
        self.declare_parameter("video_output", "cinematic_output.mp4")
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("horizontal_fov_deg", 70.0)
        self.declare_parameter("fps", 30.0)

        scene_path_str = self.get_parameter("scene_path").value
        camera_path_str = self.get_parameter("camera_path_json").value
        video_output_str = self.get_parameter("video_output").value
        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.fov_deg = float(self.get_parameter("horizontal_fov_deg").value)
        self.fps = float(self.get_parameter("fps").value)

        self.scene_path = Path(scene_path_str)
        self.camera_path_json = Path(camera_path_str)
        self.video_output = Path(video_output_str)

        # ----- устройство (gsplat по сути CUDA-библиотека) -----
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # Можно попробовать на CPU, но gsplat в норме CUDA-зависимый.
            # Лучше честно сказать, что без GPU будет плохо / не запустится.
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ----- интринсики камеры -----
        fov_rad = math.radians(self.fov_deg)
        fx = self.width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx * (self.height / self.width)
        cx = self.width / 2.0
        cy = self.height / 2.0

        self.K = torch.tensor(
            [[fx, 0.0, cx],
             [0.0, fy, cy],
             [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        # ----- сервис -----
        self.render_srv = self.create_service(
            Trigger,
            "render_cinematic",
            self.handle_render_cinematic,
        )

        self.get_logger().info(
            f"GsplatCinematicNode started. scene={self.scene_path}, "
            f"camera_path_json={self.camera_path_json}, video={self.video_output}, "
            f"size={self.width}x{self.height}, FOV={self.fov_deg} deg"
        )

    # ---------- Работа с JSON траекторией ----------

    def load_camera_path(self, path: Path) -> List[dict]:
        if not path.exists():
            raise FileNotFoundError(f"camera_path_json not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Формат от CameraPathRecorder:
        # {
        #   "map_frame": "map",
        #   "camera_frame": "camera",
        #   "sample_hz": 10.0,
        #   "num_poses": N,
        #   "poses": [ { x, y, z, qx, qy, qz, qw, ... }, ... ]
        # }

        if not isinstance(data, dict) or "poses" not in data:
            raise ValueError(
                "Unsupported camera path JSON structure: expected dict with key 'poses'"
            )

        poses = data["poses"]
        if not isinstance(poses, list) or len(poses) == 0:
            raise ValueError("camera_path_json: 'poses' must be non-empty list")

        self.get_logger().info(f"Loaded {len(poses)} camera poses from {path}")
        return poses

    def poses_to_extrinsics(self, poses: List[dict]) -> np.ndarray:
        """
        Превращаем позы (map -> camera) в матрицы extrinsic (world->camera),
        как для Open3D / классического pinhole.
        """
        extrinsics = []

        for p in poses:
            x = float(p["x"])
            y = float(p["y"])
            z = float(p["z"])
            qx = float(p["qx"])
            qy = float(p["qy"])
            qz = float(p["qz"])
            qw = float(p["qw"])

            # R_wc: rotation world<-camera (поза камеры в мире)
            R_wc = quaternion_to_rotation_matrix(qx, qy, qz, qw)
            t_wc = np.array([x, y, z], dtype=np.float32)

            # extrinsic: world -> camera
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc

            ext = np.eye(4, dtype=np.float32)
            ext[:3, :3] = R_cw
            ext[:3, 3] = t_cw
            extrinsics.append(ext)

        return np.stack(extrinsics, axis=0)  # [F, 4, 4]

    # ---------- Загрузка gsplat-сцены из PLY ----------

    def load_gsplat_scene(self, path: Path) -> GsplatScene:
        if not path.exists():
            raise FileNotFoundError(f"scene_path not found: {path}")

        try:
            from plyfile import PlyData
        except ImportError:
            raise RuntimeError(
                "Для загрузки .ply нужен пакет 'plyfile'. "
                "Установи: pip install plyfile"
            )

        self.get_logger().info(f"Загружаю gsplat-сцену из {path}")
        ply = PlyData.read(str(path))
        v = ply["vertex"].data  # structured array
        N = len(v)
        if N == 0:
            raise RuntimeError("PLY vertex list is empty")

        names = v.dtype.names

        def get_field(field_name: str, required: bool = True, default: float = 0.0):
            if field_name in names:
                return np.asarray(v[field_name], dtype=np.float32)
            if required:
                raise KeyError(f"PLY is missing required field '{field_name}'")
            return np.full((N,), default, dtype=np.float32)

        # --- позиции ---
        x = get_field("x")
        y = get_field("y")
        z = get_field("z")
        means = np.stack([x, y, z], axis=1)  # [N, 3]

        # --- log-scale -> scale ---
        s0 = get_field("scale_0")
        s1 = get_field("scale_1")
        s2 = get_field("scale_2")
        log_scales = np.stack([s0, s1, s2], axis=1)  # [N, 3]
        scales = np.exp(log_scales)

        # --- кватернионы ---
        r0 = get_field("rot_0")
        r1 = get_field("rot_1")
        r2 = get_field("rot_2")
        r3 = get_field("rot_3")
        quats = np.stack([r0, r1, r2, r3], axis=1)  # [N, 4]

        # нормализуем на всякий случай
        q_norm = np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8
        quats = quats / q_norm

        # --- opacity (logit -> alpha) ---
        opacity_raw = get_field("opacity")
        opacities = 1.0 / (1.0 + np.exp(-opacity_raw))  # sigmoid

        # --- цвет ---
        colors = None
        # Вариант 1: стандартный 3DGS: f_dc_0/1/2 (DC коэффициенты SH)
        if ("f_dc_0" in names) and ("f_dc_1" in names) and ("f_dc_2" in names):
            dc0 = get_field("f_dc_0")
            dc1 = get_field("f_dc_1")
            dc2 = get_field("f_dc_2")
            dc = np.stack([dc0, dc1, dc2], axis=1)  # [N, 3]
            # Грубый, но рабочий декод: просто sigmoid(dc)
            colors = 1.0 / (1.0 + np.exp(-dc))
        # Вариант 2: есть r/g/b как в обычном point cloud
        elif ("r" in names) and ("g" in names) and ("b" in names):
            r = get_field("r")
            g = get_field("g")
            b = get_field("b")
            rgb = np.stack([r, g, b], axis=1)
            colors = rgb / 255.0
        else:
            raise RuntimeError(
                "Не найдено ни (f_dc_0/1/2), ни (r/g/b) в PLY. "
                "Нужно подстроить загрузку под твой формат."
            )

        # --- в тензоры ---
        means_t = torch.from_numpy(means).float()
        quats_t = torch.from_numpy(quats).float()
        scales_t = torch.from_numpy(scales).float()
        opacities_t = torch.from_numpy(opacities).float()
        colors_t = torch.from_numpy(colors).float()

        scene = GsplatScene(
            means=means_t,
            quats=quats_t,
            scales=scales_t,
            opacities=opacities_t,
            colors=colors_t,
        ).to(self.device)

        self.get_logger().info(
            f"Сцена загружена: N={N} гауссиан, поля: "
            f"x/y/z, scale_0/1/2, rot_0/1/2/3, opacity + цвет"
        )
        return scene

    # ---------- Рендер одного кадра ----------

    def render_one_frame_gsplat(
        self,
        scene: GsplatScene,
        K: torch.Tensor,
        extrinsic: torch.Tensor,
    ) -> np.ndarray:
        """
        Рендер одного кадра через gsplat.rasterization.
        extrinsic – world->camera 4x4.
        """
        if self.device.type != "cuda":
            # gsplat по сути CUDA-шный, честно предупреждаем
            raise RuntimeError(
                "gsplat, как правило, требует CUDA GPU. "
                "torch.cuda.is_available() == False."
            )

        try:
            from gsplat import rasterization
        except ImportError:
            raise RuntimeError(
                "Не удалось импортировать 'gsplat'. "
                "Установи: pip install gsplat"
            )

        # viewmats / Ks имеют батч-размер по числу камер
        viewmats = extrinsic.unsqueeze(0)  # [1, 4, 4]
        Ks = K.unsqueeze(0)                # [1, 3, 3]

        with torch.no_grad():
            imgs, _meta = rasterization(
                means=scene.means,
                quats=scene.quats,
                scales=scene.scales,
                opacities=scene.opacities,
                colors=scene.colors,
                viewmats=viewmats,
                Ks=Ks,
                width=self.width,
                height=self.height,
            )

        # imgs: [1, H, W, 3], значения в [0,1]
        img = imgs[0].detach().cpu().clamp(0.0, 1.0).numpy()  # H x W x 3
        img_uint8 = (img * 255.0 + 0.5).astype(np.uint8)
        return img_uint8

    # ---------- Рендер всего видео ----------

    def render_video(
        self,
        scene: GsplatScene,
        extrinsics: np.ndarray,
    ) -> int:
        """
        Рендерит последовательность кадров по extrinsics и
        сохраняет в mp4 при помощи imageio.
        """
        try:
            import imageio.v2 as imageio
        except ImportError:
            raise RuntimeError(
                "Для записи видео нужен 'imageio[ffmpeg]'. "
                "Установи: pip install 'imageio[ffmpeg]'"
            )

        self.video_output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(self.video_output),
            fps=self.fps,
            codec="libx264",
            quality=8,
        )

        K = self.K.to(self.device)

        num_frames = 0
        try:
            for i, ext_np in enumerate(extrinsics):
                ext_t = torch.from_numpy(ext_np).float().to(self.device)
                frame = self.render_one_frame_gsplat(scene, K, ext_t)
                writer.append_data(frame)
                num_frames += 1

                if (i + 1) % 10 == 0 or i == 0:
                    self.get_logger().info(
                        f"Rendered frame {i + 1}/{len(extrinsics)}"
                    )
        finally:
            writer.close()

        return num_frames

    # ---------- Сервисный callback ----------

    def handle_render_cinematic(self, request, response):
        try:
            poses = self.load_camera_path(self.camera_path_json)
            extrinsics = self.poses_to_extrinsics(poses)
            scene = self.load_gsplat_scene(self.scene_path)

            num_frames = self.render_video(scene, extrinsics)

            msg = (
                f"Успешно отрендерено {num_frames} кадров в {self.video_output}"
            )
            self.get_logger().info(msg)
            response.success = True
            response.message = msg
            return response

        except Exception as e:
            # ВАЖНО: никакого logger.exception – у rcutils его нет.
            self.get_logger().error(f"Ошибка при рендеринге gsplat-видео: {e}")
            response.success = False
            response.message = f"Ошибка при рендеринге: {e}"
            return response


def main(args=None):
    rclpy.init(args=args)
    node = GsplatCinematicNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
