"""Start our whole mission stack (robot launch with SLAM + Nav2 must already be running).

  ros2 launch lite3_explore mission.launch.py                 # explorer in DRY RUN
  ros2 launch lite3_explore mission.launch.py live:=true      # robot drives itself
  ros2 launch lite3_explore mission.launch.py live:=true publish_obstacles:=true   # FULL RUN
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lite3_explore'), 'config')
    live = LaunchConfiguration('live')
    mission = LaunchConfiguration('mission_file')
    dry_run = PythonExpression(["'", live, "' != 'true'"])

    def node(name, params=(), condition=None):
        return Node(package='lite3_explore', executable=name, output='screen',
                    parameters=list(params), condition=condition, emulate_tty=True)

    return LaunchDescription([
        DeclareLaunchArgument('live', default_value='false',
                              description='true = explorer drives the robot'),
        DeclareLaunchArgument('mission_file', default_value=os.path.join(cfg, 'mission.yaml')),
        DeclareLaunchArgument('use_vlm', default_value='true'),
        DeclareLaunchArgument('use_qr', default_value='true'),
        DeclareLaunchArgument('use_yolo', default_value='true'),
        DeclareLaunchArgument('publish_obstacles', default_value='false',
                              description='true = YOLO detections trigger crossings'),

        node('explorer', [os.path.join(cfg, 'explorer.yaml'), {'dry_run': dry_run}]),
        node('special_obstacles', [os.path.join(cfg, 'special.yaml')]),
        node('mission_controller', [{'mission_file': mission}]),
        node('reporter', [{'mission_file': mission}]),
        node('vlm_watcher', [{'mission_file': mission}],
             IfCondition(LaunchConfiguration('use_vlm'))),
        node('qr_reader', [], IfCondition(LaunchConfiguration('use_qr'))),
        node('obstacle_detector', [os.path.join(cfg, 'obstacle.yaml'), {'publish_events': PythonExpression(["'", LaunchConfiguration('publish_obstacles'), "' == 'true'"])}],
             IfCondition(LaunchConfiguration('use_yolo'))),
    ])
