from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition

def static_tf(name, x,y,z, r,p,yaw, parent, child):
    return Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name=name,
        arguments=[
            '--x', str(x), '--y', str(y), '--z', str(z),
            '--roll', str(r), '--pitch', str(p), '--yaw', str(yaw),
            '--frame-id', parent, '--child-frame-id', child
        ]
    )

def generate_launch_description():

    map_to_camera = static_tf('nad', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 'map', 'camera')

    map_file = DeclareLaunchArgument(
        "map_file",
        default_value="",
    )
    declare_slam = DeclareLaunchArgument(
        "slam",
        default_value="False",
        description="Whether run a SLAM",
    )
    rviz_cfg = PathJoinSubstitution([FindPackageShare('mook_gaussian_bringup'), 'rviz', 'common.rviz'])
    common_cfg = PathJoinSubstitution([FindPackageShare('mook_gaussian_bringup'), 'config', 'renderer.yaml'])
    param_file = DeclareLaunchArgument(
        "params_file",
        default_value=PathJoinSubstitution(
            [
                FindPackageShare("mook_cinematic_navigation"),
                "config",
                "nav2_params.yaml",
            ]
        ),
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        "use_sim_time", default_value="False", description="Use simulation (Gazebo) clock if true"
    )

    path_planner = Node(
        package= 'mook_cinematic_navigation',
        executable= 'path_player',
        name='path_player',
        #arguments=['-d', ],
    )

    scene_renderer = Node(
        package= 'mook_scene_renderer',
        executable= 'scene_renderer',
        name='scene_renderer',
        parameters=[common_cfg],
    )

    video_recorder = Node(
        package= 'mook_scene_renderer',
        executable= 'video_recorder',
        name='video_recorder',
        parameters=[common_cfg],
    )

    ply_to_occupancy = Node(
        package= 'mook_cinematic_navigation',
        executable= 'ply_to_occupancy',
        name='ply_to_occupancy',
        parameters=[common_cfg],
    )

    nav2_bringup = IncludeLaunchDescription(
        PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "bringup_launch.py"]),
        launch_arguments=[
            ("use_sim_time", LaunchConfiguration("use_sim_time")),
            ("params_file", LaunchConfiguration("params_file")),
            ("map", LaunchConfiguration("map_file")),
            ("slam", LaunchConfiguration("slam")),
        ],
    )

    rviz2 = Node(
        package= 'rviz2',
        executable= 'rviz2',
        arguments=['-d', rviz_cfg],
    )

    ld = LaunchDescription(
        [
            map_to_camera,
            declare_slam,
            map_file,
            param_file,
            declare_use_sim_time_cmd,
            #path_planner,
            scene_renderer,
            video_recorder,
            ply_to_occupancy,
            nav2_bringup,
            rviz2,
        ]
    )


    return ld
