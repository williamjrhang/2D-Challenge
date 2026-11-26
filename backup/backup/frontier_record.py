"""Persistent record of every frontier the explorer chose (issue 2). Pure Python, no ROS.

Each entry: goal xy, the arena cell it leads into, status pending / explored / failed,
the reason, how often it was tried, and timestamps. Saved as JSON next to the coverage
.npz (coverage_X.npz -> coverage_X.frontiers.json) and restored with resume_coverage, so
nothing reachable is dropped silently and the final list explains every frontier.
"""
import json
import math
import os

PENDING, EXPLORED, FAILED = 'pending', 'explored', 'failed'


def record_path(coverage_path):
    base = coverage_path[:-4] if coverage_path.endswith('.npz') else coverage_path
    return base + '.frontiers.json'


class FrontierRecord:

    def __init__(self, match_radius_m=0.6):
        self.match_radius = match_radius_m
        self.entries = []          # list of dicts, see add()

    # ---------- updates ----------
    def find(self, xy):
        best, bd = None, self.match_radius
        for e in self.entries:
            d = math.dist(xy, e['goal_xy'])
            if d < bd:
                best, bd = e, d
        return best

    def add(self, xy, cell, t):
        """Entry for a goal about to be sent (matches an earlier one within the radius)."""
        e = self.find(xy)
        if e is None:
            e = {'id': len(self.entries), 'goal_xy': [round(float(xy[0]), 3),
                                                     round(float(xy[1]), 3)],
                 'cell': list(cell) if cell is not None else None, 'status': PENDING,
                 'reason': 'selected', 'tries': 0, 'failures': {}, 'retried': False,
                 'first_t': t, 't': t}
            self.entries.append(e)
        elif cell is not None and e['cell'] is None:
            e['cell'] = list(cell)
        e['tries'] += 1
        if e['status'] != EXPLORED:
            e['status'] = PENDING
        e['t'] = t
        return e

    def explored(self, xy, t, why='reached'):
        e = self.find(xy)
        if e is not None:
            e.update(status=EXPLORED, reason=why, t=t)
        return e

    def failed(self, xy, t, why):
        """why like 'Nav2 ABORTED' -> counted: 'unreachable: Nav2 ABORTED x2'."""
        e = self.find(xy)
        if e is None or e['status'] == EXPLORED:
            return e
        e['failures'][why] = e['failures'].get(why, 0) + 1
        parts = [f'{k} x{v}' for k, v in e['failures'].items()]
        e.update(status=FAILED, reason='unreachable: ' + ', '.join(parts), t=t)
        return e

    def note(self, xy, cell, t, status, reason):
        """Entry that was never sent (e.g. outside bounds), for the final list."""
        e = self.find(xy)
        if e is None:
            e = self.add(xy, cell, t)
            e['tries'] = 0
        if e['status'] != EXPLORED:
            e.update(status=status, reason=reason, t=t)
        return e

    # ---------- final retry ----------
    def retry_candidates(self, now, min_age_s=0.0):
        """Failed / still pending entries not retried yet, oldest first."""
        out = [e for e in self.entries
               if e['status'] in (FAILED, PENDING) and not e['retried']
               and not e['reason'].startswith('outside')
               and now - e['t'] >= min_age_s]
        return sorted(out, key=lambda e: e['t'])

    def summary(self):
        counts = {}
        for e in self.entries:
            counts[e['status']] = counts.get(e['status'], 0) + 1
        return counts

    def lines(self):
        return [f'#{e["id"]} ({e["goal_xy"][0]:.2f}, {e["goal_xy"][1]:.2f}) cell '
                f'{tuple(e["cell"]) if e["cell"] else "?"}: {e["status"]} - {e["reason"]}'
                f'{" (retried)" if e["retried"] else ""}' for e in self.entries]

    # ---------- persistence ----------
    def to_dict(self):
        return {'version': 1, 'entries': self.entries}

    def save(self, path):
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, match_radius_m=0.6):
        rec = cls(match_radius_m)
        with open(os.path.expanduser(path), encoding='utf-8') as f:
            data = json.load(f)
        rec.entries = list(data.get('entries', []))
        return rec
