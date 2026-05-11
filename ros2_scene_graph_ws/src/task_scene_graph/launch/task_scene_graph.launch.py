"""
launch/task_scene_graph.launch.py
==================================
Launches all four nodes of the Task-Driven Scene Graph system.

Usage:
  ros2 launch task_scene_graph task_scene_graph.launch.py
  ros2 launch task_scene_graph task_scene_graph.launch.py auto_demo:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    auto_demo_arg = DeclareLaunchArgument(
        'auto_demo',
        default_value='true',
        description='Auto-cycle through tasks for demo/recording (true/false)',
    )

    sim_node = Node(
        package='task_scene_graph',
        executable='scene_simulator',
        name='scene_simulator',
        output='screen',
        parameters=[{'publish_rate_hz': 10.0}],
    )

    detector_node = Node(
        package='task_scene_graph',
        executable='dynamic_detector',
        name='dynamic_detector',
        output='screen',
    )

    graph_node = Node(
        package='task_scene_graph',
        executable='scene_graph_node',
        name='scene_graph_node',
        output='screen',
    )

    task_node = Node(
        package='task_scene_graph',
        executable='task_query_node',
        name='task_query_node',
        output='screen',
        parameters=[{'auto_demo': LaunchConfiguration('auto_demo')}],
    )

    return LaunchDescription([
        auto_demo_arg,
        LogInfo(msg='=== Task-Driven Semantic Scene Graph ==='),
        LogInfo(msg='Launching: SceneSimulator, DynamicDetector, SceneGraphNode, TaskQueryNode'),
        sim_node,
        detector_node,
        graph_node,
        task_node,
    ])