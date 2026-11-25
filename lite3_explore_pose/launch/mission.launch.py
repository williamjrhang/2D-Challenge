"""Start our mission stack (robot launch with SLAM + Nav2 must already be running).

  ros2 launch lite3_explore_pose mission.launch.py                 # explorer in DRY RUN
  ros2 launch lite3_explore_pose mission.launch.py live:=true      # robot drives itself
  ros2 launch lite3_explore_pose mission.launch.py live:=true publish_obstacles:=true   # FULL RUN
  ... grid_explore:=true       # + visit the arena cell by cell (grid_bounds_cells)
  ... room_check:=true         # + face each room's left/right wall from its doorway
  ... use_wall_pins:=true      # + lite3_wall_pins: QR / picture pins on the map (RViz only,
                               #   reports still come from qr_reader). New map id every launch.

role (issue 9, two machines):
  role:=all      everything on this machine (as before)
  role:=robot    explorer, special_obstacles, mission_controller (the Lite3)
  role:=desktop  vlm_watcher, qr_reader, obstacle_detector, reporter (192.168.10.105):
                 colour image from link_relay's throttled topic, depth from link_relay's
                 thinned cloud (never the raw 640x480 cloud over the link, 24-25/11)
Normally started by robot.launch.py mission:=true / desktop.launch.py.
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
from launch.event_handlers import OnProcessIO
from launch.logging import get_logger
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from lite3_explore_pose.launch_util import profile_problem

ROLES = ('all', 'robot', 'desktop')


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
    role = LaunchConfiguration('role').perform(context)
    started = datetime.now(timezone.utc).isoformat(timespec='seconds')
    append(f'\n=== Lite3 mission started {started} (live={live}, role={role}) ===\n'.encode())
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


def dds_profile(context):
    """Use the one-NIC Fast DDS profile only where its whitelisted address exists."""
    path = LaunchConfiguration('fastdds_profile').perform(context)
    why = profile_problem(path)
    if why is None:
        return [SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', path)]
    if path:
        get_logger('lite3_explore_pose').warning(f'Fast DDS profile NOT used: {why}')
    return []


def setup(context):
    cfg = os.path.join(get_package_share_directory('lite3_explore_pose'), 'config')
    arg = lambda n: LaunchConfiguration(n).perform(context)            # noqa: E731
    on = lambda n: arg(n).lower() == 'true'                           # noqa: E731
    role = arg('role')
    if role not in ROLES:
        raise RuntimeError(f'role:={role} (use one of {ROLES})')
    live, mission = on('live'), arg('mission_file')
    depth = on('depth_cloud')
    robot, desktop = role in ('all', 'robot'), role in ('all', 'desktop')
    get_logger('lite3_explore_pose').info(
        f'mission launch role={role}: '
        + ', '.join(n for n, ok in (('robot nodes', robot), ('desktop nodes', desktop)) if ok))

    def node(name, params=(), remappings=()):
        return Node(package='lite3_explore_pose', executable=name, output='screen',
                    parameters=list(params), remappings=list(remappings), emulate_tty=True)

    # Desktop: camera topics from link_relay (issue 9); all: the raw topics as before.
    image = {} if role == 'all' else {'image_topic': arg('image_topic')}
    cloud_remap = [] if role == 'all' else [('/camera/depth/points', arg('cloud_topic'))]
    decision_nodes = {}
    if robot:
        decision_nodes['explorer'] = node('explorer', [
            os.path.join(cfg, 'explorer.yaml'),
            {'dry_run': not live, 'mission_file': mission, 'room_check': on('room_check'),
             'grid_explore': on('grid_explore')}])
        decision_nodes['special_obstacles'] = node('special_obstacles',
                                                   [os.path.join(cfg, 'special.yaml')])
        if live:
            decision_nodes['mission_controller'] = node('mission_controller', [
                os.path.join(cfg, 'controller.yaml'),
                {'mission_file': mission, 'require_localization_ready': True}])
    others = []
    if desktop:
        others.append(node('reporter', [os.path.join(cfg, 'reporter.yaml'),
                                        {'mission_file': mission}]))
        if on('use_vlm'):
            others.append(node('vlm_watcher', [{'mission_file': mission,
                                                'use_depth_cloud': depth, **image}],
                               cloud_remap))
        if on('use_qr'):
            others.append(node('qr_reader', [{'use_depth_cloud': depth, **image}], cloud_remap))
        if on('use_yolo'):
            det = {'use_depth_cloud': depth, 'publish_events': live and on('publish_obstacles'),
                   **image}
            if role != 'all':      # thinned cloud streams at ~2 Hz: keep one subscription
                det.update(cloud_topic=arg('cloud_topic'), cloud_on_demand=False)
            others.append(node('obstacle_detector', [os.path.join(cfg, 'obstacle.yaml'), det]))
    actions = record_decisions(context, {**decision_nodes,
                                         **{n.node_executable: n for n in others}})
    if desktop:
        actions += wall_pins(context)
    return actions + list(decision_nodes.values()) + others


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('lite3_explore_pose'), 'config')
    return LaunchDescription([
        # Same one-NIC DDS profile as robot.launch.py, only where its address exists.
        DeclareLaunchArgument('fastdds_profile',
                              default_value=os.path.join(cfg, 'fastdds_one_nic.xml')),
        OpaqueFunction(function=dds_profile),
        DeclareLaunchArgument('role', default_value='all',
                              description='all | robot | desktop (issue 9)'),
        DeclareLaunchArgument('live', default_value='false',
                              description='true = explorer drives the robot'),
        DeclareLaunchArgument('mission_file', default_value=os.path.join(cfg, 'mission.yaml')),
        DeclareLaunchArgument('use_vlm', default_value='true'),
        DeclareLaunchArgument('use_qr', default_value='true'),
        DeclareLaunchArgument('use_yolo', default_value='true'),
        DeclareLaunchArgument('depth_cloud', default_value='false',
                              description='true = camera nodes may fetch depth. role all: the '
                                          'raw cloud (4.4 MB over the robot link starved '
                                          '/livox/imu, 25/11); role desktop: link_relay\'s '
                                          'thinned cloud'),
        DeclareLaunchArgument('image_topic',
                              default_value='/camera/color/image_throttled/compressed',
                              description='role desktop: colour image from link_relay'),
        DeclareLaunchArgument('cloud_topic', default_value='/camera/depth/points_small',
                              description='role desktop: depth from link_relay'),
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
        OpaqueFunction(function=setup),
    ])
