"""Gazebo (Ignition Fortress) 上に FR3・キューブ・カメラを起動する．

    ros2 launch franka_vla_sim sim.launch.py            # GUIあり
    ros2 launch franka_vla_sim sim.launch.py gui:=false # GUIなし（軽い）
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('franka_vla_sim')
    franka_share = get_package_share_directory('franka_description')
    world = os.path.join(pkg, 'worlds', 'push_cube.sdf')
    xacro_file = os.path.join(pkg, 'urdf', 'fr3_vla.urdf.xacro')

    # package://franka_description/... のメッシュをGazeboが見つけられるようにする
    env = {
        'IGN_GAZEBO_RESOURCE_PATH': os.path.dirname(franka_share),
        'IGN_GAZEBO_SYSTEM_PLUGIN_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
    }

    gui = LaunchConfiguration('gui')
    gazebo_gui = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', world], additional_env=env, output='screen',
        condition=IfCondition(gui))
    gazebo_headless = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', '-s', '--headless-rendering', world],
        additional_env=env, output='screen', condition=UnlessCondition(gui))

    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description}], output='screen')

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', '/robot_description', '-name', 'fr3', '-z', '0.0'])

    # ロボットがGazeboに出現してからコントローラを起動する
    # （PCが遅いとシミュレーションの立ち上がりに時間がかかるので，待ち時間を長めにとる）
    spawner = Node(
        package='controller_manager', executable='spawner', output='screen',
        arguments=['joint_state_broadcaster', 'arm_controller', 'hand_controller',
                   '--controller-manager-timeout', '120', '--switch-timeout', '60'])
    controllers = RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[spawner]))

    # Gazebo <-> ROS 2 のトピック中継
    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
            '/cameras/front/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/cameras/wrist/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/model/red_cube/pose@geometry_msgs/msg/Pose[ignition.msgs.Pose',
        ])

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo GUIを表示するか'),
        SetParameter(name='use_sim_time', value=True),
        gazebo_gui,
        gazebo_headless,
        robot_state_publisher,
        bridge,
        TimerAction(period=3.0, actions=[spawn]),
        controllers,
    ])
