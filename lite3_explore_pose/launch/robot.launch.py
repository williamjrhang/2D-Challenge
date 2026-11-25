"""Start the Lite3 bringup with a pause between Nav2 turns and translation."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    bringup = Path(get_package_share_directory('lite3_launcher')) / 'launch' / 'lite3.launch.py'
    slam = LaunchConfiguration('slam')
    navigation = LaunchConfiguration('navigation')
    hold = LaunchConfiguration('post_turn_hold_s')
    profile = Path(get_package_share_directory('lite3_explore_pose')) / 'config' / 'fastdds_one_nic.xml'
    return LaunchDescription([
        # One NIC for DDS (see config/fastdds_one_nic.xml): with both NICs the bridge sent
        # every message twice and /livox/imu starved (25/11). fastdds_profile:=<file> to change.
        DeclareLaunchArgument('fastdds_profile', default_value=str(profile)),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE',
                               LaunchConfiguration('fastdds_profile')),
        DeclareLaunchArgument('slam', default_value='True'),
        DeclareLaunchArgument('navigation', default_value='True'),
        DeclareLaunchArgument('post_turn_hold_s', default_value='3.0',
                              description='Seconds stationary after a Nav2 turn. Zero bypasses the hold.'),
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
    ])
