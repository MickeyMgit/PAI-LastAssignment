"""推論ノード（vla_client）と実行ノード（action_executor）をまとめて起動する．

    ros2 launch franka_vla_sim policy.launch.py server_url:=http://host.docker.internal:8000
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('server_url', description='推論サーバのURL（Mac 本体なら http://host.docker.internal:8000）'),
        DeclareLaunchArgument('n_action_steps', default_value='10',
                              description='1回の推論結果のうち実行するステップ数'),
        SetParameter(name='use_sim_time', value=True),
        Node(package='franka_vla_sim', executable='vla_client', output='screen',
             parameters=[{'server_url': LaunchConfiguration('server_url')}]),
        Node(package='franka_vla_sim', executable='action_executor', output='screen',
             parameters=[{'n_action_steps': LaunchConfiguration('n_action_steps')}]),
    ])
