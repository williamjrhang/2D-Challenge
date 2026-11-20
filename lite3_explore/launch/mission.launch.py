"""Start our whole mission stack (robot launch with SLAM + Nav2 must already be running).

  ros2 launch lite3_explore mission.launch.py                 # explorer in DRY RUN
  ros2 launch lite3_explore mission.launch.py live:=true      # robot drives itself
  ros2 launch lite3_explore mission.launch.py live:=true publish_obstacles:=true   # FULL RUN
"""
import os
from datetime import datetime, timezone
from pathlib import Path
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessIO
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def record_decisions(context, sources):
    path = Path(LaunchConfiguration('decision_log').perform(context)).expanduser().resolve()
    logger = get_logger('lite3_explore')
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


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lite3_explore'), 'config')
    live = LaunchConfiguration('live')
    mission = LaunchConfiguration('mission_file')
    dry_run = PythonExpression(["'", live, "' != 'true'"])

    def node(name, params=(), condition=None):
        return Node(package='lite3_explore', executable=name, output='screen',
                    parameters=list(params), condition=condition, emulate_tty=True)

    decision_nodes = {
        'explorer': node('explorer', [os.path.join(cfg, 'explorer.yaml'),
                                      {'dry_run': dry_run, 'mission_file': mission}]),
        'special_obstacles': node('special_obstacles', [os.path.join(cfg, 'special.yaml')]),
        'mission_controller': node('mission_controller', [{'mission_file': mission}]),
    }

    return LaunchDescription([
        DeclareLaunchArgument('live', default_value='false',
                              description='true = explorer drives the robot'),
        DeclareLaunchArgument('mission_file', default_value=os.path.join(cfg, 'mission.yaml')),
        DeclareLaunchArgument('use_vlm', default_value='true'),
        DeclareLaunchArgument('use_qr', default_value='true'),
        DeclareLaunchArgument('use_yolo', default_value='true'),
        DeclareLaunchArgument('publish_obstacles', default_value='false',
                              description='true = YOLO detections trigger crossings'),
        DeclareLaunchArgument('decision_log', default_value='log/lite3_explore_decisions.log',
                              description='File to append live exploration decisions to'),

        OpaqueFunction(function=record_decisions, args=[decision_nodes]),
        *decision_nodes.values(),
        node('reporter', [{'mission_file': mission}]),
        node('vlm_watcher', [{'mission_file': mission}],
             IfCondition(LaunchConfiguration('use_vlm'))),
        node('qr_reader', [], IfCondition(LaunchConfiguration('use_qr'))),
        node('obstacle_detector', [os.path.join(cfg, 'obstacle.yaml'), {'publish_events': PythonExpression(["'", LaunchConfiguration('publish_obstacles'), "' == 'true'"])}],
             IfCondition(LaunchConfiguration('use_yolo'))),
    ])
