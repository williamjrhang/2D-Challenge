"""Observe-only wall pins (QR codes, optional printed pictures). SLAM + camera must be up.

  ros2 launch lite3_wall_pins wall_pins.launch.py
  ros2 launch lite3_wall_pins wall_pins.launch.py map_id:=arena_run2 rviz:=true
  ros2 launch lite3_wall_pins wall_pins.launch.py enable_pictures:=true   # EXPERIMENTAL
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('lite3_wall_pins')
    cfg = os.path.join(share, 'config')
    L = LaunchConfiguration

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=os.path.join(cfg, 'wall_pins.yaml')),
        DeclareLaunchArgument('map_id', default_value='default',
                              description='pins are saved per map id'),
        DeclareLaunchArgument('pins_file', default_value='',
                              description="'' = ~/.ros/lite3_wall_pins/<map_id>.json"),
        DeclareLaunchArgument('enable_pictures', default_value='false',
                              description='EXPERIMENTAL YOLO-World printed-picture pins'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),

        Node(package='lite3_wall_pins', executable='wall_pins', name='wall_pins',
             output='screen', emulate_tty=True,
             parameters=[L('params_file'), {
                 'map_id': ParameterValue(L('map_id'), value_type=str),
                 'pins_file': ParameterValue(L('pins_file'), value_type=str),
                 'enable_pictures': ParameterValue(L('enable_pictures'), value_type=bool),
                 'use_sim_time': ParameterValue(L('use_sim_time'), value_type=bool),
             }]),
        Node(package='rviz2', executable='rviz2', name='wall_pins_rviz',
             arguments=['-d', os.path.join(cfg, 'wall_pins.rviz')],
             parameters=[{'use_sim_time': ParameterValue(L('use_sim_time'), value_type=bool)}],
             condition=IfCondition(L('rviz'))),
    ])
