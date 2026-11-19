"""manual -> gait -> auto wrapper (live 18/11: gait codes were ignored in AUTO mode)."""
from lite3_explore.gait import AUTO, MANUAL, send_gait


class Msg:
    def __init__(self, cmd_code, type):
        self.cmd_code = cmd_code


class Pub:
    def __init__(self):
        self.sent = []

    def publish(self, m):
        self.sent.append(m.cmd_code)


class Log:
    def info(self, *_):
        pass


def test_gait_goes_through_manual_and_back_to_auto():
    p = Pub()
    send_gait(p, Msg, 0x21010401, Log(), via_manual=True, wait_s=0)
    assert p.sent == [MANUAL, 0x21010401, AUTO]


def test_ai_codes_are_sent_directly():
    p = Pub()
    send_gait(p, Msg, 0x21010528, Log(), via_manual=True, wait_s=0)
    send_gait(p, Msg, 0x2101052B, Log(), wait_s=0)
    assert p.sent == [0x21010528, 0x2101052B]


def test_zero_or_no_publisher_sends_nothing():
    p = Pub()
    send_gait(p, Msg, 0, Log(), wait_s=0)
    send_gait(None, Msg, 0x21010401, Log(), wait_s=0)
    assert p.sent == []


def test_direct_mode_when_disabled():
    p = Pub()
    send_gait(p, Msg, 0x21010401, Log(), via_manual=False, wait_s=0)
    assert p.sent == [0x21010401]
