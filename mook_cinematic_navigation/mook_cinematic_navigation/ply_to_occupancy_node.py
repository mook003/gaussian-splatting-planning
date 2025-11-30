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
        super().__init__('ply_to_occupancy')

        # Параметры
        self.declare_parameter('ply_path', '')
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('min_points_per_cell', 5)
        self.declare_parameter('output_prefix', 'maps/conference')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('publish_topic', 'scene_map')  # можно сделать 'map'

        ply_path = self.get_parameter('ply_path').get_parameter_value().string_value
        resolution = float(self.get_parameter('resolution').value)
        min_points = int(self.get_parameter('min_points_per_cell').value)
        output_prefix = self.get_parameter('output_prefix').get_parameter_value().string_value
        map_frame = self.get_parameter('map_frame').get_parameter_value().string_value
        publish_topic = self.get_parameter('publish_topic').get_parameter_value().string_value

        if not ply_path:
            self.get_logger().error("Parameter 'ply_path' is empty")
            raise RuntimeError("ply_path must be set")

        # Генерируем grid (и PGM+YAML)
        grid_msg = self.generate_map_and_grid(
            ply_path=Path(ply_path),
            resolution=resolution,
            min_points_per_cell=min_points,
            output_prefix=Path(output_prefix),
            frame_id=map_frame,
        )

        # QoS для "латченного" топика (аналог latched в ROS1)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, publish_topic, qos)

        self.grid_msg = grid_msg
        # Публикуем карту периодически, чтобы RViz/Nav2 всегда могли её получить
        self.timer = self.create_timer(1.0, self.publish_map)

        self.get_logger().info(
            f"Static OccupancyGrid ready, publishing on '{publish_topic}' "
            f"in frame '{map_frame}'"
        )

    def publish_map(self):
        # Обновляем stamp
        self.grid_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.grid_msg)

    def generate_map_and_grid(self, ply_path: Path, resolution: float,
                              min_points_per_cell: int, output_prefix: Path,
                              frame_id: str) -> OccupancyGrid:
        self.get_logger().info(f"Loading point cloud from {ply_path} ...")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        if len(pcd.points) == 0:
            self.get_logger().error("Point cloud is empty")
            raise RuntimeError("Point cloud is empty")

        pts = np.asarray(pcd.points, dtype=np.float64)  # N x 3
        xs = pts[:, 0]
        ys = pts[:, 2]

        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()

        res = resolution
        width = int(math.ceil((max_x - min_x) / res)) + 1
        height = int(math.ceil((max_y - min_y) / res)) + 1

        self.get_logger().info(
            f"Map size: {width} x {height} cells, resolution={res} m/cell"
        )
        self.get_logger().info(
            f"Bounds X: [{min_x:.2f}, {max_x:.2f}], Y: [{min_y:.2f}, {max_y:.2f}]"
        )

        counts = np.zeros((height, width), dtype=np.int32)

        ix = ((xs - min_x) / res).astype(int)
        iy = ((ys - min_y) / res).astype(int)
        ix = np.clip(ix, 0, width - 1)
        iy = np.clip(iy, 0, height - 1)

        for x_i, y_i in zip(ix, iy):
            counts[y_i, x_i] += 1

        occupied = counts >= min_points_per_cell

        # Небольшое "надувание" препятствий
        dilated = occupied.copy()
        h, w = occupied.shape
        for y in range(h):
            for x in range(w):
                if not occupied[y, x]:
                    continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = y + dy
                        xx = x + dx
                        if 0 <= yy < h and 0 <= xx < w:
                            dilated[yy, xx] = True
        occupied = dilated

        # -------- OccupancyGrid --------
        grid = OccupancyGrid()
        grid.header = Header()
        grid.header.frame_id = frame_id

        grid.info.resolution = res
        grid.info.width = width
        grid.info.height = height

        origin = Pose()
        origin.position.x = float(min_x)
        origin.position.y = float(min_y)
        origin.position.z = 0.0
        origin.orientation.w = 1.0  # без поворота
        grid.info.origin = origin

        # data: row-major, от (0,0) снизу-слева.
        # counts/occupied у нас уже в такой системе (iy растёт с Y).
        # data: row-major, от (0,0) снизу-слева.
        # counts/occupied у нас уже в такой системе (iy растёт с Y).
        data = np.full((height, width), 0, dtype=np.int8)  # 0 = free
        data[occupied] = 100  # 100 = occupied

        # Преобразуем в список обычных int
        flat = data.flatten()
        grid.data = [int(v) for v in flat]

        # -------- Генерация PGM+YAML для map_server/Nav2 (опционально) --------
        out_img_path = output_prefix.with_suffix(".pgm")
        out_yaml_path = output_prefix.with_suffix(".yaml")
        out_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Для PGM надо перевернуть по Y (map_server ожидает (0,0) внизу)
        img = np.full((height, width), 254, dtype=np.uint8)
        img[occupied] = 0
        img_pgm = np.flipud(img)

        self.get_logger().info(f"Writing image map to {out_img_path} ...")
        Image.fromarray(img_pgm).save(out_img_path)

        yaml_text = f"""image: {out_img_path.name}
resolution: {res}
origin: [{min_x:.6f}, {min_y:.6f}, 0.0]
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
