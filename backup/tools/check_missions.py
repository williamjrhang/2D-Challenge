#!/usr/bin/env python3
"""Check what the reporter will look for, BEFORE the run (no ROS needed).

  python3 ~/challenge_ws/lite3_explore_pose/tools/check_missions.py
  python3 .../check_missions.py --qr "My friend is in the market"     (what M05 would target)

Missions come from config/mission.yaml `missions:`, overridden line by line by
~/challenge_ws/missions_today.txt (one "M03: survivor kitchen" per line) if it exists.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lite3_explore_pose.mission_config import (load_mission, load_missions, parse_mission,  # noqa
                                          room_from_text)

ap = argparse.ArgumentParser()
ap.add_argument('--mission-file', default=os.path.join(os.path.dirname(__file__), '..',
                                                       'config', 'mission.yaml'))
ap.add_argument('--qr', default=None, help='a QR text to test')
args = ap.parse_args()

cfg = load_mission(args.mission_file)
override = os.path.expanduser(cfg.get('missions_file', ''))
print(f'group_id: {cfg["group_id"]}    send_reports: {cfg.get("send_reports")}    '
      f'time_limit_s: {cfg.get("time_limit_s")}')
print(f'override file: {override} ({"USED" if os.path.isfile(override) else "not present"})\n')
bad = 0
for tid, text in sorted(load_missions(cfg, override).items()):
    m = parse_mission(text, cfg)
    ok = m['task_type'] is not None
    bad += not ok
    print(f'  {tid}: {text!r:55} -> {m["task_type"] or "NOT UNDERSTOOD"} {m["target"] or ""}')
if args.qr is not None:
    print(f'\nQR {args.qr!r} -> room {room_from_text(args.qr, cfg)}')
print('\nOK' if not bad else f'\n{bad} mission(s) NOT understood: write them in canonical form')
sys.exit(1 if bad else 0)
