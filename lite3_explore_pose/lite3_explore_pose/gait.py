"""Send a gait code on /simple_cmd so the robot really changes gait in AUTO mode.

Default: the code is published directly (worked in auto mode for potholes on 17/11).
Optional fallback (gait_via_manual: true), only if direct codes are ignored:
    manual (0x21010C02) -> wait -> gait code -> wait -> auto (0x21010C03)
The robot is standing still whenever this is called (crossing SETTLE / RESTORE, explorer
start). AI enter/exit codes (stairs) are sent directly: they work in auto mode (Xy, 18/11).
"""
import time

from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

MANUAL = 0x21010C02    # 553716738
AUTO = 0x21010C03      # 553716739
AI_CODES = (0x21010528, 0x2101052B)

# The /simple_cmd subscriber requests transient-local durability. Expire the last
# command quickly so a subscriber that reconnects later cannot replay an old gait.
GAIT_COMMAND_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    lifespan=Duration(seconds=1),
)


def send_gait(pub, msg_cls, code, logger, via_manual=False, wait_s=0.3):
    code = int(code)
    if code == 0 or pub is None or msg_cls is None:
        return
    if not via_manual or code in AI_CODES:
        pub.publish(msg_cls(cmd_code=code, type=0))
        logger.info(f'Gait command sent: {hex(code)}')
        return
    for c in (MANUAL, code, AUTO):
        pub.publish(msg_cls(cmd_code=c, type=0))
        if c != AUTO:
            time.sleep(wait_s)
    logger.info(f'Gait command sent: manual -> {hex(code)} -> auto')


# ================= one gait table for every node (issue 4) =================
# Codes come ONLY from config/gait.yaml. A code that is not in TESTED_CODES is refused
# (sent as 0 = no switch): wrong codes on /simple_cmd can make the robot fall over.
AI_ENTER, AI_EXIT = AI_CODES
TESTED_CODES = {
    0x21010300: 'Flat gait, Slow gear (tested, team 26/11)',
    0x21010407: 'RUG gait, H-Step gear (tested, team 26/11)',
    MANUAL: 'manual mode (tested)',
    AUTO: 'auto mode (tested)',
    AI_ENTER: 'Enter AI (tested 18/11, stairs only)',
    AI_EXIT: 'Exit AI (tested 18/11)',
}
GAIT_NAMES = ('flat', 'high_step')          # gaits a node may request
MODE_NAMES = ('ai_enter', 'ai_exit', 'auto', 'manual')


def parse_code(value):
    """'0x21010300' / '553714432' / 553714432 -> int."""
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    return int(s, 16) if s.startswith('0x') else int(s)


class GaitTable:
    """Names -> /simple_cmd codes, and terrain kind -> gait name."""

    def __init__(self, codes=None, terrain=None):
        self.codes = dict(codes or {})
        self.terrain = dict(terrain or {})

    @classmethod
    def from_dict(cls, data):
        """Returns (table, errors). Untested or unreadable codes become 0 (= no switch)."""
        errors, codes = [], {}
        for name, value in (data.get('codes') or {}).items():
            try:
                code = parse_code(value)
            except (TypeError, ValueError):
                errors.append(f'gait code {name}={value!r} is not a number: disabled')
                codes[name] = 0
                continue
            if code and code not in TESTED_CODES:
                errors.append(f'gait code {name}={hex(code)} is NOT in the tested list: disabled')
                code = 0
            codes[name] = code
        terrain = {}
        for kind, name in (data.get('terrain') or {}).items():
            if name not in codes:
                errors.append(f'terrain {kind} -> unknown gait {name!r}: ignored')
                continue
            terrain[kind] = name
        return cls(codes, terrain), errors

    @classmethod
    def load(cls, path):
        import yaml
        with open(path, encoding='utf-8') as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    def code(self, name):
        return int(self.codes.get(name, 0) or 0)

    def for_terrain(self, kind):
        """Gait name for a terrain kind ('rubble' -> 'high_step'), None = no special gait."""
        return self.terrain.get(kind)


def default_gait_file():
    try:
        from ament_index_python.packages import get_package_share_directory
        import os
        return os.path.join(get_package_share_directory('lite3_explore_pose'), 'config',
                            'gait.yaml')
    except Exception:  # noqa: BLE001
        return ''


class GaitState:
    """Who owns the gait: the last switch, so /gait/current is always the truth we sent."""

    def __init__(self):
        self.current = 'unknown'
        self.history = []          # (t, old, new, why)

    def switch(self, name, why, t):
        """Record a switch; returns True if it changed."""
        old = self.current
        self.history.append((t, old, name, why))
        self.current = name
        return old != name
