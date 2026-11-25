#!/usr/bin/env python3
"""Try a trained terrain model on a laptop webcam (no robot, no ROS).

Shows live boxes with class name and confidence. Tests RECOGNITION only: a webcam has no
depth, so distances / triggers (which the robot gets from the depth camera) are not tested.
Hold the laptop low (~30-40 cm above the floor, tilted slightly down) like the robot camera.

Setup on the laptop (Windows / Mac / Linux, Python 3.9+):
    pip install ultralytics
Run:
    python webcam_test.py terrain_v1.pt               # default webcam
    python webcam_test.py terrain_v1.pt --cam 1       # another camera
    python webcam_test.py terrain_v1.pt --conf 0.4    # stricter
    python webcam_test.py terrain_v1.pt --images photos/   # a folder of photos instead
    python webcam_test.py yolov8s-worldv2.pt          # the robot's CURRENT model + its prompts
Keys: q = quit, s = save the current frame (to label it later if it was wrong).

The robot's detector uses the class names directly: they must be exactly
rubble / potholes / stairs / curtain (anything else is ignored on the robot).
"""
import argparse
import glob
import os
import time

import cv2
from ultralytics import YOLO

ROBOT_CLASSES = {'rubble', 'potholes', 'stairs', 'curtain', 'person'}
NEGATIVE_CLASSES = {'wall'}      # labelled on purpose, never triggers anything


def draw(frame, result, names):
    for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                              result.boxes.conf.tolist()):
        name = names[int(cls)]
        if name in ROBOT_CLASSES:
            colour, note = (0, 200, 0), ''                 # green = triggers on the robot
        elif name in NEGATIVE_CLASSES:
            colour, note = (160, 160, 160), ' (not an obstacle)'
        else:
            colour, note = (0, 0, 255), ' (NAME IGNORED ON ROBOT)'
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(frame, f'{name} {conf:.2f}' + note,
                    (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--cam', type=int, default=0)
    ap.add_argument('--conf', type=float, default=0.25)    # same as the robot's 'conf'
    ap.add_argument('--imgsz', type=int, default=320)      # same as the robot
    ap.add_argument('--images', help='folder of photos instead of the webcam')
    a = ap.parse_args()

    if 'world' in os.path.basename(a.model).lower():
        # Same prompts -> kinds as the robot's config/obstacle.yaml (YOLO-World, no training).
        from ultralytics import YOLOWorld
        prompts = ['metal staircase', 'stairs with handrails', 'white fabric curtain',
                   'cloth curtain in a doorway', 'plain wooden wall', 'white wall panel']
        kinds = ['stairs', 'stairs', 'curtain', 'curtain', 'wall', 'wall']
        model = YOLOWorld(a.model)
        model.set_classes(prompts)
        names = dict(enumerate(kinds))
    else:
        model = YOLO(a.model)
        names = model.names
    print('Classes in this model:', list(names.values()))
    bad = [repr(n) for n in names.values() if n not in ROBOT_CLASSES | NEGATIVE_CLASSES]
    if bad:
        print('WARNING: the robot ignores these class names:', bad)

    if a.images:
        files = sorted(glob.glob(os.path.join(a.images, '*.jpg'))
                       + glob.glob(os.path.join(a.images, '*.png')))
        for f in files:
            frame = cv2.imread(f)
            r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False,
                              agnostic_nms=True)[0]
            cv2.imshow('model test (any key = next, q = quit)', draw(frame, r, names))
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        return

    print(f'Opening camera {a.cam} ...', flush=True)
    # Windows: the default camera backend (MSMF) can hang for a long time -> DirectShow.
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    cap = cv2.VideoCapture(a.cam, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(a.cam)                      # fall back to the default
    if not cap.isOpened():
        raise SystemExit(f'Cannot open camera {a.cam} (try --cam 1, and close Teams/Zoom/Camera app)')
    print('Camera open. Window: "model test" (check the taskbar). q = quit, s = save frame.',
          flush=True)
    os.makedirs('saved_frames', exist_ok=True)
    t = time.time()
    first = True
    while True:
        ok, frame = cap.read()
        if not ok:
            print('Camera gave no picture (try --cam 1).', flush=True)
            break
        if first:
            print('First picture received, showing window...', flush=True)
            first = False
        r = model.predict(frame, imgsz=a.imgsz, conf=a.conf, verbose=False,
                          agnostic_nms=True)[0]
        fps = 1.0 / max(time.time() - t, 1e-3)
        t = time.time()
        out = draw(frame, r, names)
        cv2.putText(out, f'{fps:.1f} fps', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow('model test (q = quit, s = save frame)', out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            path = os.path.join('saved_frames', f'frame_{int(time.time())}.jpg')
            cv2.imwrite(path, frame)
            print('saved', path)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
