"""ON THE LITE3: bringup (FAST-LIO, slam_toolbox, Nav2 from lite3_launcher, unchanged),
turn_hold, link_relay and, with mission:=true, the robot's mission nodes.

  ros2 launch lite3_explore_pose robot.launch.py                          # bringup + relay
  ros2 launch lite3_explore_pose robot.launch.py mission:=true            # + explorer (DRY RUN)
  ros2 launch lite3_explore_pose robot.launch.py mission:=true live:=true publish_obstacles:=true
On the desktop (192.168.10.105):  ros2 launch lite3_explore_pose desktop.launch.py live:=true ...
Same ROS_DOMAIN_ID on both machines; check with tools/check_link.sh.

Issue 9 placement: robot = FAST-LIO, slam_toolbox, Nav2, turn_hold, explorer,
mission_controller, special_obstacles (gait owner), link_relay, camera drivers.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            OpaqueFunction, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue

from lite3_explore_pose.launch_util import profile_problem


def dds_profile(context):
    # One NIC for DDS (see config/fastdds_one_nic.xml): with both NICs the bridge sent every
    # message twice and /livox/imu starved (25/11). Only where its address exists (issue 9:
    # on the Lite3 itself group5's 192.168.1.75 does not exist). The profile is unchanged.
    path = LaunchConfiguration('fastdds_profile').perform(context)
    why = profile_problem(path)
    if why is None:
        return [SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', path)]
    if path:
        get_logger('lite3_explore_pose').warning(f'Fast DDS profile NOT used: {why}')
    return []


def generate_launch_description():
    bringup = Path(get_package_share_directory('lite3_launcher')) / 'launch' / 'lite3.launch.py'
    share = Path(get_package_share_directory('lite3_explore_pose'))
    slam = LaunchConfiguration('slam')
    navigation = LaunchConfiguration('navigation')
    hold = LaunchConfiguration('post_turn_hold_s')
    return LaunchDescription([
        DeclareLaunchArgument('fastdds_profile',
                              default_value=str(share / 'config' / 'fastdds_one_nic.xml')),
        OpaqueFunction(function=dds_profile),
        DeclareLaunchArgument('slam', default_value='True'),
        DeclareLaunchArgument('navigation', default_value='True'),
        DeclareLaunchArgument('post_turn_hold_s', default_value='3.0',
                              description='Seconds stationary after a Nav2 turn. Zero bypasses the hold.'),
        DeclareLaunchArgument('relay', default_value='true',
                              description='link_relay: throttled image + thinned depth for the desktop'),
        DeclareLaunchArgument('mission', default_value='false',
                              description='true = also the robot mission nodes (mission.launch.py role:=robot)'),
        GroupAction([
            # Match the original names, before the launcher's own remap rules.
            SetRemap(src='velocity_smoother:cmd_vel_smoothed', dst='/nav_cmd_vel_raw'),
            SetRemap(src='behavior_server:cmd_vel', dst='/nav_cmd_vel_raw'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(bringup)),
                launch_arguments={'slam': slam, 'navigation': navigation}.items()),
        ]),
        Node(package='lite3_explore_pose', executable='turn_hold', output='screen',
             parameters=[{'post_turn_hold_s': ParameterValue(hold, value_type=float)}],
             condition=IfCondition(navigation)),
        Node(package='lite3_explore_pose', executable='link_relay', output='screen',
             parameters=[str(share / 'config' / 'link_relay.yaml')],
             condition=IfCondition(LaunchConfiguration('relay'))),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(str(share), 'launch', 'mission.launch.py')),
            launch_arguments={'role': 'robot'}.items(),
            condition=IfCondition(LaunchConfiguration('mission'))),
    ])
