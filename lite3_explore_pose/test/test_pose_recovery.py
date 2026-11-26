from unittest.mock import Mock, patch

from geometry_msgs.msg import TransformStamped
from nav2_simple_commander.robot_navigator import TaskResult
from std_msgs.msg import Bool, String
from tf2_ros import TransformException
from visualization_msgs.msg import Marker

from lite3_explore_pose.cells import CellGrid
from lite3_explore_pose.explorer_node import FrontierExplorer
from lite3_explore_pose.mission_config import Arena
from lite3_explore_pose.mission_controller import MissionController
from lite3_explore_pose.pose_health import PoseHealth


class Transforms:
    def __init__(self):
        self.stamps = {}
        self.x = {}

    def lookup_transform(self, parent, child, time):
        stamp = self.stamps.get((parent, child))
        if stamp is None:
            raise TransformException('missing transform')
        transform = TransformStamped()
        transform.header.stamp.sec = stamp
        transform.transform.translation.x = self.x.get((parent, child), 0.0)
        return transform

    def update(self, odom, slam, pose):
        self.stamps.update({('odom', 'base_link'): odom, ('map', 'odom'): slam,
                            ('map', 'base_link'): pose})


def test_localization_requires_stable_tf_and_advancing_odometry():
    transforms = Transforms()
    health = PoseHealth(transforms, 'map', 'odom', 'base_link', 2.5, 5.0, 1.0)
    assert health.sample(100)[1] == 'odom -> base_link unavailable'
    transforms.update(100, 99, 99)
    assert health.sample(100)[0] is None
    assert health.sample(101)[0] is None  # The same odometry sample is insufficient.
    transforms.update(101, 100, 100)
    assert health.sample(101)[0] is not None
    assert health.pose_stamp == 100_000_000_000
    assert 'odom -> base_link timestamp age' in health.sample(104)[1]
    transforms.update(104, 98, 98)
    assert 'map -> odom timestamp age' in health.sample(104)[1]
    transforms.update(104, 104, 98)
    assert 'map -> base_link timestamp age' in health.sample(104)[1]
    transforms.update(104, 0, 104)
    assert health.sample(104)[0] is None
    transforms.update(105, 0, 105)
    assert health.sample(105)[0] is not None


def explorer_for_test():
    explorer = object.__new__(FrontierExplorer)
    explorer.P = {
        'dry_run': False, 'coverage_enabled': False, 'pose_result_wait_s': 3.0,
        'pose_goal_tolerance_m': 0.35, 'pose_goal_cooldown_s': 15.0,
        'cell_visit_edge_margin_m': 0.1,
    }
    explorer.nav = Mock()
    explorer.nav.isTaskComplete.return_value = True
    explorer.nav.getResult.return_value = TaskResult.SUCCEEDED
    explorer.pose_health = Mock(pose_stamp=10, odom_stamp=10)
    explorer.goal_pose_stamp = 10
    explorer.goal_odom_stamp = 10
    explorer.goal_xy = (2.0, 0.0)
    explorer.goal_yaw = 0.0
    explorer.goal_kind = 'cell'
    explorer.goal_prioritizes_room = False
    explorer.goal_is_frontier = False
    arena = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0, 0],
                             'start_heading_deg': 0}})
    explorer.cells = CellGrid(arena, bounds_cells=[0, -2, 5, 2])
    explorer.cell_attempts = {(2, 0): 1}
    explorer.goal_was_retried = explorer.pose_retry_pending = explorer.pose_cancel_pending = False
    explorer.success_seen_at = None
    explorer.now_s = Mock(return_value=100.0)
    explorer.get_logger = Mock()
    explorer.publish_cmd = Mock()
    explorer.set_state = lambda state: setattr(explorer, 'state', state)
    explorer.send_goal = Mock(return_value=True)
    explorer.update_cells = Mock()
    explorer.localization_pub = Mock()
    explorer.room_priority_pub = Mock()
    explorer.has_room_priority = Mock(return_value=False)
    explorer.grid = Mock()
    explorer.robot_pose = Mock(return_value=None)
    explorer.pose_problem = 'odom -> base_link unavailable'
    explorer.deferred_goals = {}
    explorer.blacklist = []
    explorer.enabled = True
    explorer.state = 'NAVIGATE'
    return explorer


def test_dropout_cancels_goal_and_waits_for_cancellation_and_localization_before_retry():
    explorer = explorer_for_test()
    explorer.tick()
    explorer.nav.cancelTask.assert_called_once()
    assert explorer.state == 'WAIT_POSE'
    assert explorer.pose_retry_pending
    explorer.robot_pose.return_value = (0.0, 0.0, 0.0)
    explorer.nav.isTaskComplete.return_value = False
    explorer.tick()
    explorer.send_goal.assert_not_called()
    explorer.nav.isTaskComplete.return_value = True
    explorer.tick()
    explorer.send_goal.assert_called_once_with((2.0, 0.0), (0.0, 0.0), False, 'cell', 0.0,
                                               retry=True)
    assert explorer.state == 'NAVIGATE'
    assert explorer.cell_attempts[(2, 0)] == 1
    assert not explorer.blacklist


def test_fresh_odometry_jump_is_rejected_and_stays_rejected_after_settle_reset():
    transforms = Transforms()
    health = PoseHealth(transforms, 'map', 'odom', 'base_link', 2.5, 5.0, 1.0)
    transforms.update(100, 100, 100)
    health.sample(100)
    transforms.update(101, 101, 101)
    assert health.sample(101)[0] is not None
    transforms.update(102, 102, 102)
    transforms.x[('odom', 'base_link')] = 4.5
    pose, reason = health.sample(102)
    assert pose is None and 'odom -> base_link jumped 4.50 m' in reason
    health.reset()
    assert health.sample(104) == (None, reason)


def test_visit_requires_advancing_pose_inside_cell_and_draws_without_map_classification():
    explorer = explorer_for_test()
    explorer.P.update(global_frame='map', cell_overlay=True)
    explorer.get_clock = Mock()
    explorer.get_clock.return_value.now.return_value.to_msg.return_value = (
        TransformStamped().header.stamp)
    explorer.cells_pub = Mock()
    explorer.map_seq = explorer.cells_seq = 1
    explorer.last_cells_pub = 100.0
    explorer.cell_info = {}
    explorer.room_status = {}
    explorer.visited_cells = set()
    explorer.visit_sample = None
    explorer.wall_memory, explorer.walls_seen = {}, set()
    pose = (1.39, 0.0, 0.0)  # Inside cell (1, 0), outside its old central square.
    for sample in (pose, pose):
        FrontierExplorer.update_cells(explorer, sample)
    assert not explorer.visited_cells
    explorer.pose_health.pose_stamp += 1
    FrontierExplorer.update_cells(explorer, pose)
    assert explorer.visited_cells == {(1, 0)}
    markers = explorer.cells_pub.publish.call_args.args[0].markers
    cubes = next(marker for marker in markers if marker.type == Marker.CUBE_LIST)
    assert cubes.color.a > 0
    assert cubes.colors[0].g > cubes.colors[0].r


def test_mission_disallows_crossings_until_localization_and_stops_on_lost_heartbeat():
    mission = object.__new__(MissionController)
    mission.P = {'require_localization_ready': True, 'pose_jump_m': 0.0}
    mission.state = 'EXPLORE'
    mission.pending_special = None
    mission.localization_ready_at = None
    mission.obstacles = []
    mission.now_s = Mock(return_value=100.0)
    mission.get_logger = Mock()
    mission.special_pub = Mock()
    mission.cancel_nav = Mock()
    mission.enter = Mock()
    event = String(data='rubble_ahead')
    mission.on_event(event)
    mission.tick()
    assert mission.pending_special is None
    mission.enter.assert_not_called()

    mission.on_localization_ready(Bool(data=True))
    mission.last_special_end = -100.0
    mission.last_special_kind = None
    mission.on_event(event)
    assert mission.pending_special == 'rubble'
    mission.state = 'TRAVERSE'
    mission.now_s.return_value = 103.0
    mission.tick()
    mission.special_pub.publish.assert_called_once_with(String(data='stop'))
    mission.enter.assert_called_once_with('DONE')


def test_success_stays_unverified_until_a_newer_pose_is_available():
    explorer = explorer_for_test()
    explorer.start_arrival_hold = Mock(return_value=True)
    pose = (*explorer.goal_xy, 0.0)
    explorer.do_navigate(pose)
    explorer.start_arrival_hold.assert_not_called()
    explorer.pose_health.pose_stamp += 1
    explorer.do_navigate(pose)
    explorer.start_arrival_hold.assert_not_called()
    explorer.pose_health.odom_stamp += 1
    explorer.do_navigate(pose)
    explorer.start_arrival_hold.assert_called_once()


def test_inconsistent_success_on_retry_defers_goal_without_using_an_attempt():
    explorer = explorer_for_test()
    explorer.goal_was_retried = True
    explorer.pose_health.pose_stamp += 1
    explorer.pose_health.odom_stamp += 1
    wrong_pose = (0.0, 0.0, 0.0)
    explorer.do_navigate(wrong_pose)
    explorer.now_s.return_value = 103.0
    explorer.do_navigate(wrong_pose)
    assert explorer.state == 'WAIT_POSE'
    assert explorer.deferred_goals == {explorer.goal_xy: 118.0}
    assert explorer.cell_attempts[(2, 0)] == 0
    assert not explorer.blacklist


def test_localization_retry_rechecks_that_frontier_still_exists():
    explorer = explorer_for_test()
    explorer.goal_kind = 'frontier'
    explorer.goal_is_frontier = True
    explorer.P['free_max'] = 25
    explorer.grid = Mock()
    explorer.info = Mock()
    explorer.wait_for_pose('missing TF', retry_goal=True)
    with patch('lite3_explore_pose.explorer_node.still_frontier', return_value=False):
        explorer.do_wait_pose((0.0, 0.0, 0.0))
    explorer.send_goal.assert_not_called()
    assert explorer.state == 'SELECT'
    assert not explorer.blacklist
