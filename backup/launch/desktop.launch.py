"""ON THE DESKTOP (192.168.10.105): scene understanding, QR, obstacle detection, reporter, RViz.

  ros2 launch lite3_explore_pose desktop.launch.py                        # log only
  ros2 launch lite3_explore_pose desktop.launch.py live:=true publish_obstacles:=true
Robot side: robot.launch.py (bringup + link_relay [+ mission:=true]). Same ROS_DOMAIN_ID.

Issue 9 placement (decided 26/11): vlm_watcher, qr_reader, obstacle_detector, reporter and
RViz here, so the Lite3's onboard computer stays light. Nothing here subscribes to the raw
depth cloud: the camera nodes get link_relay's throttled image (image_topic) and thinned
cloud (cloud_topic), which link_relay stops sending while the robot's /livox/imu or lidar
rate is too low. If this machine or the link goes down, the robot keeps exploring
(mission_controller investigate_timeout_s, explorer room_look_timeout_s).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('lite3_explore_pose')
    return LaunchDescription([
        DeclareLaunchArgument('depth_cloud', default_value='true',
                              description='true = depth from link_relay\'s thinned cloud '
                                          '(decided 26/11); false = ray placement only'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config',
                              default_value=os.path.join(share, 'config', 'maze_light.rviz')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'mission.launch.py')),
            launch_arguments={'role': 'desktop',
                              'depth_cloud': LaunchConfiguration('depth_cloud')}.items()),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', LaunchConfiguration('rviz_config')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
