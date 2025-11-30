import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge


class PreviewRenderer(Node):
    def __init__(self):
        super().__init__('preview_renderer')

        self.bridge = CvBridge()
        self.width = 640
        self.height = 480

        self.subscription = self.create_subscription(
            PoseStamped,
            'camera/pose',
            self.pose_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            'camera/image_preview',
            10
        )

    def pose_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        # Простая картинка-заглушка
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        text = f"x={x:.2f}, y={y:.2f}, z={z:.2f}"
        cv2.putText(
            img,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Конвертация в sensor_msgs/Image
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        img_msg.header = msg.header  # унаследуем stamp и frame_id

        self.publisher.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PreviewRenderer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
