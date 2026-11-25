"""Build our ROS nodes without rclpy.init / DDS (no traffic on the robot's domain).
Parameters come from the node's own defaults, then the YAML file, then overrides."""
import os
from contextlib import contextmanager
from unittest.mock import Mock, patch

import yaml
from rclpy.node import Node

CFG = os.path.join(os.path.dirname(__file__), '..', 'config')


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class Param:
    def __init__(self, value):
        self.value = value


@contextmanager
def fake_ros(params):
    pubs = {}

    def declare(self, name, default=None):
        params.setdefault(name, default)
        return Param(params[name])

    def create_publisher(self, typ, topic, qos):
        pubs[topic] = Mock(name=topic)
        pubs[topic].get_subscription_count.return_value = 0
        return pubs[topic]

    with patch.multiple(Node, __init__=lambda self, *a, **k: None,
                        declare_parameter=declare,
                        has_parameter=lambda self, n: n in params,
                        get_parameter=lambda self, n: Param(params[n]),
                        create_subscription=lambda self, *a, **k: Mock(),
                        create_publisher=create_publisher,
                        create_timer=lambda self, *a, **k: Mock(),
                        get_logger=lambda self: Mock(),
                        get_clock=lambda self: Mock()):
        yield pubs


def make(cls, yaml_file=None, key=None, **overrides):
    params = {}
    if yaml_file:
        with open(os.path.join(CFG, yaml_file)) as f:
            params.update(yaml.safe_load(f)[key]['ros__parameters'])
    params.update(overrides)
    with fake_ros(params) as pubs:
        node = cls()
    node.pubs = pubs
    node.now_s = Clock()
    node.log = Mock()
    node.get_logger = lambda: node.log
    return node
