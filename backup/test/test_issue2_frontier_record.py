"""Issue 2: frontier record (persisted), final retry list, wall seen mask, wall items."""
import math

import numpy as np

from lite3_explore_pose.cells import CellGrid
from lite3_explore_pose.explorer_node import failure_key
from lite3_explore_pose.frontier_record import FrontierRecord, record_path
from lite3_explore_pose.frontiers import MapInfo
from lite3_explore_pose.mission_config import Arena
from lite3_explore_pose.reporter import merge_wall_item

RES = 0.05


def test_record_statuses_reasons_and_persistence(tmp_path):
    r = FrontierRecord(0.6)
    r.add((1.0, 0.0), (1, 0), 10.0)
    r.add((3.0, 1.0), (3, 1), 11.0)
    r.failed((3.05, 1.0), 12.0, 'Nav2 ABORTED')
    r.add((3.0, 1.0), (3, 1), 30.0)
    r.failed((3.0, 1.0), 31.0, 'Nav2 ABORTED')
    r.explored((1.0, 0.1), 13.0)
    assert r.entries[0]['status'] == 'explored'
    assert r.entries[1]['reason'] == 'unreachable: Nav2 ABORTED x2'
    assert r.entries[1]['tries'] == 2
    r.note((9.0, 9.0), (9, 9), 40.0, 'failed', 'outside bounds')
    # retry candidates: failed, not 'outside bounds', old enough, oldest first
    assert [e['id'] for e in r.retry_candidates(now=60.0, min_age_s=20.0)] == [1]
    assert r.retry_candidates(now=40.0, min_age_s=20.0) == []
    path = record_path(str(tmp_path / 'coverage_x.npz'))
    assert path.endswith('coverage_x.frontiers.json')
    r.save(path)
    back = FrontierRecord.load(path)
    assert back.entries == r.entries
    assert any('outside bounds' in ln for ln in back.lines())


def test_explored_is_never_downgraded_by_a_late_failure():
    r = FrontierRecord()
    r.add((0.0, 0.0), None, 0.0)
    r.explored((0.0, 0.0), 1.0)
    r.failed((0.0, 0.0), 2.0, 'Nav2 ABORTED')
    assert r.entries[0]['status'] == 'explored'


def test_failure_key():
    assert failure_key('Nav2 result ABORTED') == 'Nav2 ABORTED'
    assert failure_key('goal time limit (44 s)') == 'time limit'
    assert failure_key('progress 0.10 m in 10.2 s, need 0.30 m') == 'no progress'


def test_wall_seen_frac_from_camera_mask():
    arena = Arena({'arena': {'cell_size_m': 1.02, 'start_cell_centre': [0, 0],
                             'start_heading_deg': 0}})
    g = CellGrid(arena)
    info = MapInfo(RES, -2.0, -2.0, int(4 / RES), int(4 / RES))
    grid = np.zeros((info.height, info.width), dtype=np.int16)
    # wall on the +y side of cell (0, 0) at y = 0.51
    for x in np.arange(-0.5, 0.5, RES / 2):
        r, c = info.world_to_cell(x, 0.51)
        grid[r, c] = 100
    seen = np.zeros_like(grid, dtype=bool)
    assert g.wall_seen_frac(grid, info, seen, (0, 0), '+y') == 0.0
    seen[grid >= 65] = True
    assert g.wall_seen_frac(grid, info, seen, (0, 0), '+y') == 1.0
    assert g.wall_seen_frac(grid, info, seen, (0, 0), '-y') == 0.0   # no wall there


def test_wall_items_merge_and_move():
    items = []
    a = merge_wall_item(items, 'qr', 'HELLO', (1.0, 0.5), 0.5)
    b = merge_wall_item(items, 'qr', 'HELLO', (1.2, 0.5), 0.5)
    assert a is b and math.isclose(a['x'], 1.1) and a['n'] == 2
    c = merge_wall_item(items, 'hazard', 'radioactive', (1.1, 0.5), 0.5)
    assert c['id'] == 1 and len(items) == 2


def test_final_retry_unblacklists_and_resends_each_failed_frontier_once():
    from unittest.mock import Mock
    from lite3_explore_pose.explorer_node import FrontierExplorer
    e = object.__new__(FrontierExplorer)
    e.P = {'frontier_retry_enabled': True, 'frontier_retry_delay_s': 20.0, 'dry_run': False,
           'blacklist_radius_m': 0.6, 'idle_warn_period_s': 5.0}
    e.bounds = None
    e.frontier_rec = FrontierRecord()
    e.frontier_rec.add((3.0, 1.0), (3, 1), 10.0)
    e.frontier_rec.failed((3.0, 1.0), 11.0, 'Nav2 ABORTED')
    e.blacklist = [(3.0, 1.0)]
    e.deferred_goals = {}
    e.retry_queue, e.final_retry_done = None, False
    e.now_s = Mock(return_value=20.0)
    e.get_logger = Mock()
    e.state = 'SELECT'
    e.set_state = lambda st: setattr(e, 'state', st)
    e.send_goal = Mock(return_value=True)
    e.save_record = Mock()
    assert e.start_final_retry((0.0, 0.0, 0.0)) is True        # waiting: failed 9 s ago
    e.send_goal.assert_not_called()
    e.now_s.return_value = 40.0
    assert e.start_final_retry((0.0, 0.0, 0.0)) is True
    e.send_goal.assert_called_once_with((3.0, 1.0), (0.0, 0.0), is_frontier=True, target=(3, 1))
    assert e.blacklist == [] and e.state == 'NAVIGATE'
    assert e.start_final_retry((0.0, 0.0, 0.0)) is False       # nothing left: list logged
    assert e.final_retry_done
    assert e.start_final_retry((0.0, 0.0, 0.0)) is False       # only once
