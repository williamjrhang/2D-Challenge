"""Helpers shared by the launch files (no ROS imports)."""
import re
import socket


def local_ipv4():
    ips = set()
    try:
        import psutil
        for addrs in psutil.net_if_addrs().values():
            ips.update(a.address for a in addrs if a.family == socket.AF_INET)
    except Exception:  # noqa: BLE001
        pass
    return ips


def profile_problem(path):
    """The one-NIC Fast DDS profile whitelists group5's address (192.168.1.75). On another
    machine (the Lite3, the desktop) that interface does not exist and Fast DDS would have
    NO usable UDP transport ('User transport failed to register'). Returns a reason not to
    use the profile on this machine, or None. The profile itself is never changed."""
    if not path:
        return 'no profile given'
    try:
        text = open(path, encoding='utf-8').read()
    except OSError as e:
        return f'cannot read {path}: {e}'
    wanted = set(re.findall(r'<address>\s*([\d.]+)\s*</address>', text))
    missing = wanted - local_ipv4()
    if wanted and missing:
        return f'{path} whitelists {sorted(missing)}, which this machine does not have'
    return None
