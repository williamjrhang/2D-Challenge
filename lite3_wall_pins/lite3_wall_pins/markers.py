"""PinStore -> visualization_msgs/MarkerArray (map frame)."""

import math

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

COLOURS = {'qr': (0.0, 0.75, 1.0), 'picture': (1.0, 0.55, 0.0)}
VIEW_COLOUR = (0.2, 0.9, 0.2)


def _marker(frame, stamp, ns, mid, mtype, rgb, alpha=1.0):
    m = Marker()
    m.header.frame_id = frame
    m.header.stamp = stamp
    m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
    m.pose.orientation.w = 1.0
    m.color.r, m.color.g, m.color.b = rgb
    m.color.a = alpha
    return m


def build(store, stamp, show_view_pose=True, text_height=0.18):
    arr = MarkerArray()
    clear = Marker()
    clear.header.frame_id = store.frame_id
    clear.header.stamp = stamp
    clear.action = Marker.DELETEALL
    arr.markers.append(clear)
    f = store.frame_id
    for i, p in enumerate(store.pins):
        rgb = COLOURS.get(p.kind, (1.0, 1.0, 1.0))

        s = _marker(f, stamp, 'wall_pins', i, Marker.SPHERE, rgb,
                    0.6 if p.experimental else 1.0)
        s.pose.position.x, s.pose.position.y, s.pose.position.z = p.x, p.y, p.z
        s.scale.x = s.scale.y = s.scale.z = 0.18
        arr.markers.append(s)

        # pole down to the floor (z = 0) so the pin is easy to see from the top view
        pole = _marker(f, stamp, 'wall_pin_poles', i, Marker.LINE_LIST, rgb, 0.8)
        pole.scale.x = 0.03
        pole.points = [Point(x=p.x, y=p.y, z=0.0), Point(x=p.x, y=p.y, z=p.z)]
        arr.markers.append(pole)

        t = _marker(f, stamp, 'wall_pin_labels', i, Marker.TEXT_VIEW_FACING, (1.0, 1.0, 1.0))
        t.pose.position.x, t.pose.position.y = p.x, p.y
        t.pose.position.z = p.z + 0.3
        t.scale.z = text_height
        label = p.label if len(p.label) <= 40 else p.label[:37] + '...'
        t.text = f'{p.id}  {label}  (x{p.observations})'
        arr.markers.append(t)

        if show_view_pose and p.view is not None:
            v = p.view
            a = _marker(f, stamp, 'view_poses', i, Marker.ARROW, VIEW_COLOUR, 0.9)
            a.pose.position.x, a.pose.position.y, a.pose.position.z = v.x, v.y, 0.05
            a.pose.orientation.z = math.sin(v.yaw / 2.0)
            a.pose.orientation.w = math.cos(v.yaw / 2.0)
            a.scale.x, a.scale.y, a.scale.z = 0.4, 0.06, 0.06
            arr.markers.append(a)

            ln = _marker(f, stamp, 'view_rays', i, Marker.LINE_LIST, VIEW_COLOUR, 0.5)
            ln.scale.x = 0.015
            ln.points = [Point(x=v.x, y=v.y, z=0.05), Point(x=p.x, y=p.y, z=p.z)]
            arr.markers.append(ln)
    return arr
