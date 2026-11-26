"""Issue 1: a frontier goal is not cancelled when it gets mapped; it is done only once the
robot is inside the cell the frontier leads into. Pose-jump latch recovers."""
from unittest.mock import Mock

import numpy as np
from geometry_msgs.msg import TransformStamped
from nav2_simple_commander.robot_navigator import TaskResult
from tf2_ros import TransformException

from lite3_explore_pose.cells import CellGrid
from lite3_explore_pose.explorer_node import FrontierExplorer
from lite3_explore_pose.frontier_goals import (cell_seen_fraction, frontier_step, target_cell,
                                               unknown_side_xy)
from lite3_explore_pose.frontier_record import FrontierRecord
from lite3_explore_pose.frontiers import MapInfo
from lite3_explore_pose.mission_config import Arena
from lite3_explore_pose.pose_health import PoseHealth

RES = 0.05


def grid_world(unknown_from_x=None):
    info = MapInfo(resolution=RES, origin_x=-1.0, origin_y=-2.0, width=int(6 / RES),
                   height=int(4 / RES))
    grid = np.zeros((info.height, info.width), dtype=np.int16)
    if unknown_from_x is not None:
        c = info.world_to_cell(unknown_from_x, 0.0)[1]
        grid[:, c:] = -1
    return grid, info


def cells():
    arena = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0, 0],
                             'start_heading_deg': 0}})
    return CellGrid(arena)


# ---------------- pure logic ----------------

def test_target_cell_is_the_cell_behind_the_frontier():
    g = cells()
    grid, info = grid_world(unknown_from_x=1.55)
    fr = [info.world_to_cell(1.52, y) for y in np.arange(-0.3, 0.3, RES)]
    side = unknown_side_xy(grid, info, fr)
    assert side[0] > 1.52
    assert target_cell(g.cell_of, (1.52, 0.0), (0.0, 0.0), side) == (2, 0)
    # no unknown side known: fall back to the robot -> goal direction
    assert target_cell(g.cell_of, (1.52, 0.0), (0.0, 0.0), None) == (2, 0)


def test_mapped_frontier_converts_instead_of_cancelling():
    assert frontier_step(False, (0, 0), (2, 0), None) == 'convert'
    assert frontier_step(False, (0, 0), (2, 0), None, converted=True) == 'continue'
    assert frontier_step(True, (1, 0), (2, 0), None) == 'continue'
    assert frontier_step(False, (2, 0), (2, 0), None) == 'done'
    assert frontier_step(False, None, (2, 0), 0.95) == 'done'       # camera saw it all
    assert frontier_step(False, None, (2, 0), 0.5) == 'convert'


def test_cell_seen_fraction():
    grid, info = grid_world()
    seen = np.zeros_like(grid, dtype=bool)
    assert cell_seen_fraction(seen, grid, info, (2.04, 0.0), 0.4) == 0.0
    r0, c0 = info.world_to_cell(1.64, -0.4)
    r1, c1 = info.world_to_cell(2.44, 0.4)
    seen[r0:r1 + 1, c0:c1 + 1] = True
    assert cell_seen_fraction(seen, grid, info, (2.04, 0.0), 0.4) == 1.0


# ---------------- explorer acceptance ----------------

def explorer(grid, info):
    e = object.__new__(FrontierExplorer)
    e.P = {'dry_run': False, 'coverage_enabled': False, 'free_max': 25,
           'recheck_period_s': 2.0, 'time_limit_base_s': 20.0, 'time_limit_per_m_s': 8.0,
           'progress_window_s': 10.0, 'min_progress_m': 0.3, 'frontier_enter_cell': True,
           'frontier_seen_done_frac': 0.9, 'cell_visit_edge_margin_m': 0.1,
           'pose_result_wait_s': 3.0, 'pose_goal_tolerance_m': 0.35}
    e.grid, e.info = grid, info
    e.cells = cells()
    e.cell_info = {}
    e.visited_cells = set()
    e.blacklist = []
    e.nav = Mock()
    e.nav.isTaskComplete.return_value = False
    e.nav.getFeedback.return_value = None
    e.turn_hold_active = False
    e.get_logger = Mock()
    e.now_s = Mock(return_value=100.0)
    e.set_state = lambda st: setattr(e, 'state', st)
    e.last_cells_pub = 0.0
    sent = []

    def send_goal(xy, robot_xy, is_frontier=True, kind=None, yaw=None, retry=False,
                  target=None, converted=False):
        sent.append((xy, target, converted))
        e.goal_xy, e.goal_target_cell, e.goal_converted = xy, target, converted
        e.goal_is_frontier, e.goal_kind = is_frontier, kind or 'frontier'
        e.goal_start = e.last_recheck = e.now_s()
        e.goal_time_limit = 60.0
        e.window_remaining = None
        return True
    e.send_goal = send_goal
    e.sent = sent
    e.frontier_rec = FrontierRecord()
    e.frontier_rec.add((1.52, 0.0), (2, 0), 100.0)
    e.goal_record_xy = (1.52, 0.0)
    send_goal((1.52, 0.0), (-0.5, 0.0), target=(2, 0))
    e.state = 'NAVIGATE'
    return e


def test_frontier_mapped_two_metres_away_is_not_cancelled_and_completes_only_inside_cell():
    grid, info = grid_world()           # fully mapped: the frontier is already gone
    e = explorer(grid, info)
    e.now_s.return_value = 103.0
    e.do_navigate((-0.5, 0.0, 0.0))     # 2 m from the frontier
    e.nav.cancelTask.assert_not_called()
    assert e.state == 'NAVIGATE'
    assert e.sent[-1] == (e.cells.centre((2, 0)), (2, 0), True)   # now 'enter cell (2, 0)'

    e.now_s.return_value = 106.0
    e.do_navigate((1.50, 0.0, 0.0))     # on the border of cell (1, 0)/(2, 0): not inside yet
    e.nav.cancelTask.assert_not_called()
    assert e.state == 'NAVIGATE'

    e.now_s.return_value = 109.0
    e.do_navigate((2.0, 0.05, 0.0))     # inside cell (2, 0)
    e.nav.cancelTask.assert_called_once()
    assert e.state == 'SELECT'
    assert (2, 0) in e.visited_cells
    assert e.frontier_rec.entries[0]['status'] == 'explored'   # issue 2 record


def test_nav2_success_at_the_frontier_edge_converts_to_enter_the_cell():
    grid, info = grid_world(unknown_from_x=1.55)
    e = explorer(grid, info)
    e.nav.isTaskComplete.return_value = True
    e.nav.getResult.return_value = TaskResult.SUCCEEDED
    e.pose_health = Mock(pose_stamp=20, odom_stamp=20)
    e.goal_pose_stamp = e.goal_odom_stamp = 10
    e.success_seen_at = None
    e.do_navigate((1.52, 0.0, 0.0))
    assert e.sent[-1][1:] == ((2, 0), True)
    assert e.state == 'NAVIGATE'


# ---------------- pose-jump latch ----------------

class Tf:
    def __init__(self):
        self.t, self.x = 0, 0.0

    def lookup_transform(self, parent, child, time):
        if self.t is None:
            raise TransformException('x')
        tr = TransformStamped()
        tr.header.stamp.sec = self.t
        tr.transform.translation.x = self.x if (parent, child) == ('odom', 'base_link') else 0.0
        return tr


def test_jump_latch_recovers_after_stable_period_but_hard_jump_does_not():
    tf = Tf()
    h = PoseHealth(tf, 'map', 'odom', 'base_link', 2.5, 5.0, 1.0,
                   jump_recover_s=10.0, hard_jump_m=5.0)
    for t in (100, 101):
        tf.t = t
        h.sample(t)
    tf.t, tf.x = 102, 1.3 + 0.6         # 1.9 m in 1 s > 0.5 + 1 s slack
    pose, why = h.sample(102)
    assert pose is None and 'jumped' in why
    for t in range(103, 112):
        tf.t = t
        assert h.sample(t)[0] is None   # still latched
    tf.t = 112
    h.sample(112)
    assert h.jump is None and h.recovered
    tf.t = 113
    h.sample(113)
    tf.t = 114
    assert h.sample(114)[0] is not None    # healthy again

    tf.t, tf.x = 115, 9.0               # hard jump
    assert 'hard limit' in h.sample(115)[1]
    for t in range(116, 140):
        tf.t = t
        assert h.sample(t)[0] is None


def test_new_jump_restarts_the_recovery_clock():
    tf = Tf()
    h = PoseHealth(tf, 'map', 'odom', 'base_link', 2.5, 5.0, 1.0,
                   jump_recover_s=10.0, hard_jump_m=5.0)
    for t in (100, 101):
        tf.t = t
        h.sample(t)
    tf.t, tf.x = 102, 2.0
    h.sample(102)
    tf.t, tf.x = 108, 4.0 + 6.0         # second jump (2 m + 6 s slack exceeded? 8 m > 6.5)
    tf.x = 2.0 + 7.0
    h.sample(108)
    tf.t = 115
    h.sample(115)
    assert h.jump is not None           # 7 s since the second jump < 10 s
