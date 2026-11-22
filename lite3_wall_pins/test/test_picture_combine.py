from lite3_wall_pins.picture_detect import combine, iou

PICS = ['poster of a person on a wall', 'printed photo of a person']


def test_plain_person_is_never_a_picture():
    dets = [('person', 0.95, (100, 100, 200, 400))]
    assert combine(dets, PICS, 'person', 0.3) == []


def test_picture_prompt_with_person_score_attached():
    dets = [('person', 0.84, (100, 100, 200, 300)),
            ('poster of a person on a wall', 0.94, (98, 102, 202, 298)),
            ('printed photo of a person', 0.40, (101, 100, 199, 301))]
    out = combine(dets, PICS, 'person', 0.3)
    assert len(out) == 1
    assert out[0].picture_score == 0.94 and out[0].person_score == 0.84


def test_low_picture_score_dropped():
    dets = [('poster of a person on a wall', 0.06, (0, 0, 50, 50))]
    assert combine(dets, PICS, 'person', 0.3) == []


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
