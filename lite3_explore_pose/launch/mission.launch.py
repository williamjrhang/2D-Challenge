"""Start our whole mission stack (robot launch with SLAM + Nav2 must already be running).

  ros2 launch lite3_explore_pose mission.launch.py                 # explorer in DRY RUN
  ros2 launch lite3_explore_pose mission.launch.py live:=true      # robot drives itself
  ros2 launch lite3_explore_pose mission.launch.py live:=true publish_obstacles:=true   # FULL RUN
  ... grid_explore:=true       # + visit the arena cell by cell (grid_bounds_cells)
  ... room_check:=true         # + face each room's left/right wall from its doorway
  ... use_wall_pins:=true      # + lite3_wall_pins: QR / picture pins on the map (RViz only,
                               #   reports still come from qr_reader). New map id every launch.
"""
import os
from datetime import datetime, timezone
from pathlib import Path
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessIO
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def record_decisions(context, sources):
    path = Path(LaunchConfiguration('decision_log').perform(context)).expanduser().resolve()
    logger = get_logger('lite3_explore_pose')
    ansi_color = re.compile(rb'\x1b\[[0-9;]*m')
    failed = False

    def append(data):
        nonlocal failed
        if failed:
            return
        try:
            with path.open('ab') as log:
                log.write(data)
        except OSError as exc:
            failed = True
            logger.warning(f'Decision log unavailable at {path}: {exc}')

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning(f'Cannot create decision log directory {path.parent}: {exc}')
        return []
    live = LaunchConfiguration('live').perform(context)
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    append(f'\n=== Lite3 mission started {started} (live={live}) ===\n'.encode())
    if failed:
        return []
    logger.info(f'Live decision log: {path}')

    def capture(name):
        pending = b''
        prefix = f'[{name}] '.encode()

        def on_output(event):
            nonlocal pending
            *lines, pending = (pending + event.text).split(b'\n')
            if lines:
                append(b''.join(prefix + ansi_color.sub(b'', line.rstrip(b'\r')) + b'\n'
                                for line in lines))

        return on_output

    return [RegisterEventHandler(OnProcessIO(target_action=action,
                                             on_stdout=capture(name), on_stderr=capture(name)))
            for name, action in sources.items()]


def wall_pins(context):
    """lite3_wall_pins with a fresh map id (it refuses a pins file from another map)."""
    if LaunchConfiguration('use_wall_pins').perform(context).lower() != 'true':
        return []
    try:
        share = get_package_share_directory('lite3_wall_pins')
    except Exception:  # noqa: BLE001
        get_logger('lite3_explore_pose').warning('use_wall_pins: lite3_wall_pins not installed')
        return []
    map_id = (LaunchConfiguration('wall_pins_map_id').perform(context)
              or datetime.now().strftime('explore_%Y%m%d_%H%M%S'))
    get_logger('lite3_explore_pose').info(f'wall_pins map_id={map_id}')
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'wall_pins.launch.py')),
        launch_arguments={'map_id': map_id, 'rviz': 'false'}.items())]


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lite3_explore_pose'), 'config')
    live = LaunchConfiguration('live')
    mission = LaunchConfiguration('mission_file')
    dry_run = PythonExpression(["'", live, "' != 'true'"])
    depth = PythonExpression(["'", LaunchConfiguration('depth_cloud'), "' == 'true'"])

    def node(name, params=(), condition=None):
        return Node(package='lite3_explore_pose', executable=name, output='screen',
                    parameters=list(params), condition=condition, emulate_tty=True)

    decision_nodes = {
        'explorer': node('explorer', [os.path.join(cfg, 'explorer.yaml'),
                                      {'dry_run': dry_run, 'mission_file': mission,
                                       'room_check': PythonExpression(
                                           ["'", LaunchConfiguration('room_check'),
                                            "' == 'true'"]),
                                       'grid_explore': PythonExpression(
                                           ["'", LaunchConfiguration('grid_explore'),
                                            "' == 'true'"])}]),
        'special_obstacles': node('special_obstacles', [os.path.join(cfg, 'special.yaml')]),
        'mission_controller': node('mission_controller', [
            os.path.join(cfg, 'controller.yaml'),
            {'mission_file': mission, 'require_localization_ready': True}],
            IfCondition(live)),
    }

    return LaunchDescription([
        # Same one-NIC DDS profile as robot.launch.py (camera topics also cross the bridge).
        DeclareLaunchArgument('fastdds_profile',
                              default_value=os.path.join(cfg, 'fastdds_one_nic.xml')),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE',
                               LaunchConfiguration('fastdds_profile')),
        DeclareLaunchArgument('live', default_value='false',
                              description='true = explorer drives the robot'),
        DeclareLaunchArgument('mission_file', default_value=os.path.join(cfg, 'mission.yaml')),
        DeclareLaunchArgument('use_vlm', default_value='true'),
        DeclareLaunchArgument('use_qr', default_value='true'),
        DeclareLaunchArgument('use_yolo', default_value='true'),
        DeclareLaunchArgument('depth_cloud', default_value='false',
                              description='true = camera nodes may fetch /camera/depth/points. '
                                          'Off: 4.4 MB clouds over the robot link starve '
                                          '/livox/imu and FAST-LIO (measured 25/11)'),
        DeclareLaunchArgument('publish_obstacles', default_value='false',
                              description='true = YOLO detections trigger crossings'),
        DeclareLaunchArgument('grid_explore', default_value='false',
                              description='true = visit arena cells one by one (bounded)'),
        DeclareLaunchArgument('room_check', default_value='false',
                              description='true = explorer checks rooms (QR walls up close)'),
        DeclareLaunchArgument('use_wall_pins', default_value='false',
                              description='true = also run lite3_wall_pins (QR pins in RViz)'),
        DeclareLaunchArgument('wall_pins_map_id', default_value='',
                              description="'' = new id per launch (explore_<date_time>)"),
        DeclareLaunchArgument('decision_log', default_value='log/lite3_explore_pose_decisions.log',
                              description='File to append live exploration decisions to'),

        OpaqueFunction(function=record_decisions, args=[decision_nodes]),
        OpaqueFunction(function=wall_pins),
        *decision_nodes.values(),
        node('reporter', [os.path.join(cfg, 'reporter.yaml'), {'mission_file': mission}]),
        node('vlm_watcher', [{'mission_file': mission, 'use_depth_cloud': depth}],
             IfCondition(LaunchConfiguration('use_vlm'))),
        node('qr_reader', [{'use_depth_cloud': depth}], IfCondition(LaunchConfiguration('use_qr'))),
        node('obstacle_detector', [os.path.join(cfg, 'obstacle.yaml'),
                                  {'use_depth_cloud': depth,
                                   'publish_events': PythonExpression([
                                      "'", live, "' == 'true' and '",
                                      LaunchConfiguration('publish_obstacles'), "' == 'true'"])}],
             IfCondition(LaunchConfiguration('use_yolo'))),
    ])
