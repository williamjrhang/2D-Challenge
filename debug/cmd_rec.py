import rclpy, time, sys
from geometry_msgs.msg import Twist
from std_msgs.msg import String
rclpy.init(); n = rclpy.create_node('cmd_recorder'); t0 = time.time(); f = open(sys.argv[1], 'w')
def log(s): f.write(f'{time.time()-t0:8.2f} {s}\n'); f.flush()
for topic in ['/cmd_vel', '/nav_cmd_vel', '/joy_cmd_vel']:
    n.create_subscription(Twist, topic, lambda m, t=topic: log(f'{t} vx={m.linear.x:+.2f} wz={m.angular.z:+.2f}'), 50)
for topic in ['/special/state', '/mission/state']:
    n.create_subscription(String, topic, lambda m, t=topic: log(f'{t} {m.data}'), 10)
try:
    from rclpy.qos import qos_profile_sensor_data
    import rosidl_runtime_py.utilities as u
    import subprocess
    ty = subprocess.run(['ros2', 'topic', 'type', '/us_publisher/ultrasound_distance'], capture_output=True, text=True, timeout=10).stdout.strip()
    if ty:
        cls = u.get_message(ty)
        n.create_subscription(cls, '/us_publisher/ultrasound_distance', lambda m: log(f'/ultrasound {m}'[:160]), qos_profile_sensor_data)
        log(f'ultrasound type {ty}')
except Exception as e:
    log(f'ultrasound not recorded: {e!r}')
while time.time() - t0 < 1200: rclpy.spin_once(n, timeout_sec=0.1)
