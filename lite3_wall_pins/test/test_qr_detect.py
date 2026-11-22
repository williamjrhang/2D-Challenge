import cv2
import numpy as np

from lite3_wall_pins.qr_detect import QrDetector


def qr_image(text, px):
    q = cv2.QRCodeEncoder.create().encode(text)
    q = cv2.resize(q, (px, px), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(q, px // 8, px // 8, px // 8, px // 8, cv2.BORDER_CONSTANT,
                              value=255)


def scene(codes, size=(480, 640)):
    img = np.full(size + (3,), 170, np.uint8)
    for text, px, (x, y) in codes:
        q = cv2.cvtColor(qr_image(text, px), cv2.COLOR_GRAY2BGR)
        img[y:y + q.shape[0], x:x + q.shape[1]] = q
    return img


def test_decodes_text_and_corners():
    img = scene([('ROOM-7', 120, (300, 150))])
    dets = QrDetector().detect(img)
    assert [d.text for d in dets] == ['ROOM-7']
    c = dets[0].corners
    assert c.shape == (4, 2)
    # corners = the dark code area, measured independently from the pixels
    dark = np.argwhere(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) < 60)
    (y0, x0), (y1, x1) = dark.min(axis=0), dark.max(axis=0) + 1
    assert abs(c[:, 0].min() - x0) < 3 and abs(c[:, 0].max() - x1) < 3
    assert abs(c[:, 1].min() - y0) < 3 and abs(c[:, 1].max() - y1) < 3


def test_two_codes_side_by_side():
    img = scene([('LEFT', 100, (60, 150)), ('RIGHT', 100, (380, 150))])
    texts = sorted(d.text for d in QrDetector().detect(img))
    assert texts == ['LEFT', 'RIGHT']


def test_perspective_view_of_a_side_wall():
    q = cv2.cvtColor(qr_image('SIDE-WALL', 160), cv2.COLOR_GRAY2BGR)
    h, w = q.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[200, 120], [330, 160], [330, 300], [200, 340]])   # ~40 deg yaw
    M = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(q, M, (640, 480), borderValue=(170, 170, 170))
    assert [d.text for d in QrDetector().detect(img)] == ['SIDE-WALL']


def test_nothing_in_empty_image():
    assert QrDetector().detect(np.full((480, 640, 3), 128, np.uint8)) == []
