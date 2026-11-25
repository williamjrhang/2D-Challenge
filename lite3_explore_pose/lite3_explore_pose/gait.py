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
