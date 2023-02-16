from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='py_server_3',
            executable='task1'
        ),
        Node(
            package='py_server_3',
            executable='task2'
        )
    ])
