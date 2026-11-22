"""Static guarantee: this package never commands the robot.

Scans the code (docstrings and comments removed) for any movement / navigation / control
interface, and checks every publisher lives under /wall_pins/.
"""
import ast
import pathlib
import re

SRC = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(list((SRC / 'lite3_wall_pins').glob('*.py')) + list((SRC / 'launch').glob('*.py')))

FORBIDDEN = [
    r'\bTwist', r'cmd_vel', r'ActionClient', r'NavigateToPose', r'FollowWaypoints',
    r'nav2', r'BasicNavigator', r'simple_cmd', r'MotionSimpleCMD', r'transfer_interfaces',
    r'goal_pose', r'/perception/', r'/exploration/', r'/special/', r'/mission/',
    r'create_client', r'initialpose',
]


def code_only(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0] = ast.Pass()                      # drop docstring
    return ast.unparse(tree)                          # comments are dropped too


def test_no_control_interfaces():
    hits = [f'{f.name}: {pat}' for f in FILES for pat in FORBIDDEN
            if re.search(pat, code_only(f))]
    assert not hits, hits


def test_only_wall_pins_topics_published():
    code = code_only(SRC / 'lite3_wall_pins' / 'pin_node.py')
    pubs = re.findall(r"create_publisher\(([^)]*)\)", code)
    topics = [re.search(r"'([^']+)'", p).group(1) for p in pubs]
    assert len(topics) == 3 and all(t.startswith('/wall_pins/') for t in topics), topics


def test_rviz_config_has_no_goal_tools():
    rviz = (SRC / 'config' / 'wall_pins.rviz').read_text()
    assert 'SetGoal' not in rviz and 'SetInitialPose' not in rviz and 'PublishPoint' not in rviz
