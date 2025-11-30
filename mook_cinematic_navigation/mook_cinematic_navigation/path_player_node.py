import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class CircularPathPlayer(Node):
    def __init__(self):
        super().__init__('circular_path_player')

        self.publisher = self.create_publisher(PoseStamped, 'camera/pose', 10)

        # Параметры «орбиты»
        self.radius = 3.0
        self.center_x = 0.0
        self.center_y = 0.0
        self.height = 1.5
        self.angular_speed = 0.2  # рад/с
        self.fps = 15.0

        self.dt = 1.0 / self.fps
        self.t = 0.0

        self.timer = self.create_timer(self.dt, self.timer_callback)

    def timer_callback(self):
        self.t += self.dt
        angle = self.angular_speed * self.t

        x = self.center_x + self.radius * math.cos(angle)
        y = self.center_y + self.radius * math.sin(angle)
        z = self.height

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'

        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z

        # Пока без «взгляда в центр» — просто единичный кватернион
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0

        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CircularPathPlayer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
