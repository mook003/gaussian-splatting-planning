from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scene_ply = LaunchConfiguration('scene_ply')
    camera_path_out = LaunchConfiguration('camera_path_out')
    image_width = LaunchConfiguration('image_width')
    image_height = LaunchConfiguration('image_height')
    horizontal_fov_deg = LaunchConfiguration('horizontal_fov_deg')

    return LaunchDescription([
        DeclareLaunchArgument(
            'scene_ply',
            default_value='src/data/ConferenceHall.ply',
            description='Path to Gaussian-splatting PLY scene'
        ),
        DeclareLaunchArgument(
            'camera_path_out',
            default_value='camera_path.json',
            description='Where to save camera_path.json'
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='1280',
            description='Render width'
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='720',
            description='Render height'
        ),
        DeclareLaunchArgument(
            'horizontal_fov_deg',
            default_value='70.0',
            description='Horizontal FOV in degrees'
        ),

        # Твоя нода траектории (можно оставить как раньше)
        Node(
            package='mook_cinematic_navigation',
            executable='path_player',
            name='path_player',
            parameters=[
                {'radius': 3.0},
                {'height': 1.5},
                {'angular_speed': 0.2},
                {'fps': 30.0},
            ]
        ),

        # Превью в реальном времени через Open3D
        Node(
            package='mook_scene_renderer',
            executable='scene_renderer',
            name='scene_renderer',
            parameters=[
                {'scene_ply_path': scene_ply},
                {'image_width': image_width},
                {'image_height': image_height},
                {'horizontal_fov_deg': horizontal_fov_deg},
                {'camera_frame_id': 'camera'},
            ]
        ),

        # Запись траектории в camera_path.json
        Node(
            package='mook_scene_renderer',
            executable='camera_path_recorder',
            name='camera_path_recorder',
            parameters=[
                {'output_path': camera_path_out},
                {'sample_hz': 30.0},
                {'image_width': image_width},
                {'image_height': image_height},
                {'horizontal_fov_deg': horizontal_fov_deg},
            ]
        ),
    ])
