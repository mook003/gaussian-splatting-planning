#!/usr/bin/env python3
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from geometry_msgs.msg import Pose


class PlyToOccupancyNode(Node):
    def __init__(self):
        super().__init__("ply_to_occupancy")

        # --------- Базовые параметры карты ---------
        self.declare_parameter("ply_path", "")
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("min_points_per_cell", 3)
        self.declare_parameter("output_prefix", "maps/conference")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("publish_topic", "scene_map")

        # --------- Фильтр по высоте над полом ---------
        # Пол = минимальный Y (после всех отражений по осям!)
        self.declare_parameter("enable_height_filter", True)
        self.declare_parameter("height_min_above_floor", 0.05)  # м над полом
        self.declare_parameter("height_max_above_floor", 2.0)   # м над полом

        # --------- Зеркалирование (чтобы «развернуть» сцену) ---------
        # Работают ОДИНАКОВО здесь и в scene_renderer.
        self.declare_parameter("flip_x", False)
        self.declare_parameter("flip_y", False)
        self.declare_parameter("flip_z", False)

        # --------- Отладка: сохранить отфильтрованное облако ---------
        self.declare_parameter("save_filtered_ply", True)
        self.declare_parameter("filtered_ply_path", "src/data/conference_filtered.ply")

        # --------- Чтение параметров ---------
        ply_path = self.get_parameter("ply_path").value
        resolution = float(self.get_parameter("resolution").value)
        min_points = int(self.get_parameter("min_points_per_cell").value)
        output_prefix = Path(self.get_parameter("output_prefix").value)
        map_frame = self.get_parameter("map_frame").value
        publish_topic = self.get_parameter("publish_topic").value

        enable_height_filter = bool(self.get_parameter("enable_height_filter").value)
        h_min = float(self.get_parameter("height_min_above_floor").value)
        h_max = float(self.get_parameter("height_max_above_floor").value)

        flip_x = bool(self.get_parameter("flip_x").value)
        flip_y = bool(self.get_parameter("flip_y").value)
        flip_z = bool(self.get_parameter("flip_z").value)

        save_filtered = bool(self.get_parameter("save_filtered_ply").value)
        filtered_ply_path = Path(self.get_parameter("filtered_ply_path").value)

        if not ply_path:
            self.get_logger().error("Parameter 'ply_path' is empty")
            raise RuntimeError("ply_path must be set")

        if enable_height_filter:
            # Если перепутали местами — меняем аккуратно.
            if h_max > 0.0 and h_min > h_max:
                self.get_logger().warn(
                    f"height_min_above_floor ({h_min}) > height_max_above_floor ({h_max}), "
                    f"меняю их местами."
                )
                h_min, h_max = h_max, h_min

            if h_max <= 0.0:
                self.get_logger().info(
                    f"Height filter: используем только нижнюю границу >= {h_min:.2f} м над полом "
                    f"(верхняя граница отключена)."
                )
            else:
                self.get_logger().info(
                    f"Height filter ENABLED: [{h_min:.2f}, {h_max:.2f}] м над полом (ось Y)."
                )
        else:
            self.get_logger().info("Height filter DISABLED — используем все точки по высоте.")

        if flip_x or flip_y or flip_z:
            self.get_logger().info(
                f"Axis flips: flip_x={flip_x}, flip_y={flip_y}, flip_z={flip_z}"
            )

        # --------- Генерация OccupancyGrid + PGM/YAML ---------
        grid_msg = self.generate_map_and_grid(
            ply_path=Path(ply_path),
            resolution=resolution,
            min_points_per_cell=min_points,
            output_prefix=output_prefix,
            frame_id=map_frame,
            enable_height_filter=enable_height_filter,
            h_min=h_min,
            h_max=h_max,
            flip_x=flip_x,
            flip_y=flip_y,
            flip_z=flip_z,
            save_filtered_ply=save_filtered,
            filtered_ply_path=filtered_ply_path,
        )

        # QoS для "латченного" топика
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, publish_topic, qos)

        self.grid_msg = grid_msg
        self.timer = self.create_timer(1.0, self.publish_map)

        self.get_logger().info(
            f"Static OccupancyGrid ready, publishing on '{publish_topic}' "
            f"in frame '{map_frame}'"
        )

    def publish_map(self):
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.grid_msg)

    # ------------------------------------------------------------

    def generate_map_and_grid(
        self,
        ply_path: Path,
        resolution: float,
        min_points_per_cell: int,
        output_prefix: Path,
        frame_id: str,
        enable_height_filter: bool,
        h_min: float,
        h_max: float,
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        save_filtered_ply: bool,
        filtered_ply_path: Path,
    ) -> OccupancyGrid:
        self.get_logger().info(f"Loading point cloud from {ply_path} ...")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            self.get_logger().error("Point cloud is empty")
            raise RuntimeError("Point cloud is empty")

        # В сыром виде (как в PLY)
        pts = np.asarray(pcd.points, dtype=np.float64)

        # --------- Зеркалим оси, чтобы "развернуть" сцену ---------
        if flip_x or flip_y or flip_z:
            pts = pts.copy()
            if flip_x:
                pts[:, 0] *= -1.0
            if flip_y:
                pts[:, 1] *= -1.0
            if flip_z:
                pts[:, 2] *= -1.0
            pcd.points = o3d.utility.Vector3dVector(pts)

        # После отражений — это уже координаты в нашей map-системе
        xs_all = pts[:, 0]
        ys_all = pts[:, 1]  # вертикаль (Y вверх)
        zs_all = pts[:, 2]

        self.get_logger().info(
            f"Bounds after flips (map frame): "
            f"X [{xs_all.min():.3f}, {xs_all.max():.3f}], "
            f"Y [{ys_all.min():.3f}, {ys_all.max():.3f}], "
            f"Z [{zs_all.min():.3f}, {zs_all.max():.3f}]"
        )

        N = pts.shape[0]

        # --------- Пол = минимальный Y ---------
        floor_y = ys_all.min()
        self.get_logger().info(f"Assuming FLOOR at Y = {floor_y:.3f} (min Y).")

        # --------- Фильтр по высоте над полом ---------
        if enable_height_filter:
            h = ys_all - floor_y  # высота над полом, всегда >= 0

            if h_max > 0.0:
                mask = (h >= h_min) & (h <= h_max)
            else:
                mask = (h >= h_min)

            kept = int(mask.sum())
            if kept == 0:
                self.get_logger().warn(
                    "Height filter: 0 points in given range, "
                    "DISABLING filter and using all points."
                )
                pts_used = pts
            else:
                pts_used = pts[mask]
                self.get_logger().info(
                    f"Height filter kept {kept} / {N} points "
                    f"({100.0 * kept / N:.1f}%)."
                )

                if save_filtered_ply:
                    filtered_pcd = o3d.geometry.PointCloud()
                    filtered_pcd.points = o3d.utility.Vector3dVector(pts_used)
                    if pcd.has_colors():
                        colors = np.asarray(pcd.colors)[mask]
                        filtered_pcd.colors = o3d.utility.Vector3dVector(colors)
                    filtered_ply_path.parent.mkdir(parents=True, exist_ok=True)
                    o3d.io.write_point_cloud(str(filtered_ply_path), filtered_pcd)
                    self.get_logger().info(
                        f"Saved filtered point cloud (for MeshLab etc.) to {filtered_ply_path}"
                    )
        else:
            pts_used = pts

        if pts_used.shape[0] == 0:
            self.get_logger().error(
                "After filtering there are 0 points. "
                "Try changing height_min/height_max or disabling filter."
            )
            raise RuntimeError("No points left after filtering")

        # --------- 2D-проекция: X и Z -> карта ---------
        xs = pts_used[:, 0]
        ys_map = pts_used[:, 2]  # карту строим в плоскости (X, Z)

        min_x, max_x = xs.min(), xs.max()
        min_y_map, max_y_map = ys_map.min(), ys_map.max()

        res = resolution
        width = int(math.ceil((max_x - min_x) / res)) + 1
        height = int(math.ceil((max_y_map - min_y_map) / res)) + 1

        self.get_logger().info(
            f"Map size: {width} x {height} cells, resolution={res} m/cell"
        )
        self.get_logger().info(
            f"Map bounds (map frame): X [{min_x:.2f}, {max_x:.2f}], "
            f"Y(from Z) [{min_y_map:.2f}, {max_y_map:.2f}]"
        )

        # --------- Счётчик точек по ячейкам ---------
        counts = np.zeros((height, width), dtype=np.int32)

        ix = ((xs - min_x) / res).astype(int)
        iy = ((ys_map - min_y_map) / res).astype(int)
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)

        for x_i, y_i in zip(ix, iy):
            counts[y_i, x_i] += 1

        occupied = counts >= min_points_per_cell
        num_occ = int(occupied.sum())
        self.get_logger().info(
            f"Cells with >= {min_points_per_cell} pts: "
            f"{num_occ} ({100.0 * num_occ / (width * height):.1f}% of map)"
        )

        # --------- Простая диляция препятствий ---------
        dilated = occupied.copy()
        hgt, wdt = occupied.shape
        for y in range(hgt):
            if not occupied[y].any():
                continue
            for x in range(wdt):
                if not occupied[y, x]:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = y + dy
                        xx = x + dx
                        if 0 <= yy < hgt and 0 <= xx < wdt:
                            dilated[yy, xx] = True
        occupied = dilated

        # --------- OccupancyGrid ---------
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.frame_id = frame_id

        grid.info.resolution = res
        grid.info.width = width
        grid.info.height = height

        origin = Pose()
        origin.position.x = float(min_x)
        origin.position.y = float(min_y_map)
        origin.position.z = 0.0
        origin.orientation.w = 1.0
        grid.info.origin = origin

        data = np.full((height, width), 0, dtype=np.int8)  # 0 = free
        data[occupied] = 100  # 100 = occupied
        grid.data = [int(v) for v in data.flatten()]

        # --------- PGM + YAML ---------
        out_img_path = output_prefix.with_suffix(".pgm")
        out_yaml_path = output_prefix.with_suffix(".yaml")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        # PGM: 0 = чёрный (занято), 254 = белый (свободно)
        img = np.full((height, width), 254, dtype=np.uint8)
        img[occupied] = 0
        # map_server ожидает (0,0) внизу, а в PGM (0,0) — сверху,
        # поэтому переворачиваем по вертикали только для изображения.
        img_pgm = np.flipud(img)

        self.get_logger().info(f"Writing image map to {out_img_path} ...")
        Image.fromarray(img_pgm).save(out_img_path)

        yaml_text = f"""image: {out_img_path.name}
resolution: {res}
origin: [{min_x:.6f}, {min_y_map:.6f}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        self.get_logger().info(f"Writing YAML map to {out_yaml_path} ...")
        out_yaml_path.write_text(yaml_text)

        self.get_logger().info("Map generation done (OccupancyGrid + PGM/YAML).")
        return grid


def main(args=None):
    rclpy.init(args=args)
    node = PlyToOccupancyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
