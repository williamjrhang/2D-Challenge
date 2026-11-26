"""Stop-and-look decisions (issue 5). Pure Python, no ROS.

vlm_watcher is unchanged: it asks the scene server in the background while the robot moves.
If it is already busy when /perception/look arrives, its /perception/look_done closes the
look with the photo taken WHILE MOVING. We cannot see inside vlm_watcher, so we estimate:
    photo time ~= look_done arrival - latency_s (from the latest /perception/scene answer)
If that is before the look was sent, the photo is stale: ask for one more look.
"""


def photo_is_stale(look_sent_t, done_t, latency_s, margin_s=0.3):
    """True if the answered photo was (probably) taken before the look was requested.
    latency_s None (no answer / unknown) -> False: nothing to judge, do not loop."""
    if latency_s is None or look_sent_t is None or done_t is None:
        return False
    return done_t - float(latency_s) < look_sent_t - margin_s


class LookCooldown:
    """Which arena cells were investigated, and when (per-cell cooldown)."""

    def __init__(self, cooldown_s=60.0):
        self.cooldown_s = cooldown_s
        self.last = {}            # cell -> time of the last stop-and-look there
        self.scene_done = set()   # cells already investigated because of a scene answer

    def may_look(self, cell, now):
        if cell is None:
            return True
        t = self.last.get(cell)
        return t is None or now - t >= self.cooldown_s

    def may_scene_look(self, cell, now):
        """Scene answer -> stop only for a cell not investigated yet (and not cooling down)."""
        return cell is not None and cell not in self.scene_done and self.may_look(cell, now)

    def mark(self, cell, now, scene=False):
        if cell is None:
            return
        self.last[cell] = now
        if scene:
            self.scene_done.add(cell)
