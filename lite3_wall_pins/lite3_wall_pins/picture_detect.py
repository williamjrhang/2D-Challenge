"""EXPERIMENTAL: printed pictures of people with the local YOLO-World model.

What the model can and cannot tell apart
----------------------------------------
YOLO-World (yolov8s-worldv2.pt + CLIP ViT-B/32 text encoder, both in
~/challenge_ws/models) is an open-vocabulary box detector: it scores image regions against
text prompts. It has NO trained "printed vs. real" class. Bench test (see README):

  * prompt "person" fires strongly on real people AND on printed photos of people;
  * prompt "poster of a person on a wall" scored 0.38-0.94 on a synthetic printed photo on a
    plain wall and <= 0.06 on real people in two stock photos.

That is a useful hint, not a guarantee, so a detection only becomes a picture candidate when
  (a) a *picture* prompt (not plain "person") scores >= picture_conf, and
  (b) the node's depth check finds the region FLAT (a sheet on a wall), which a real person
      standing in front of a wall is not.
Real people pressed flat against a wall, TV screens, mirrors and posters of non-people are
known blind spots. Only validated offline on synthetic images; needs a robot test.

Torch / ultralytics are imported lazily, only when the feature is enabled.
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class PictureDetection:
    box: tuple                   # (x1, y1, x2, y2) pixels in the decoded image
    picture_score: float         # best score of a picture prompt for this box
    person_score: float          # best score of the plain person prompt overlapping it
    prompt: str


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def combine(dets, picture_prompts, person_prompt, picture_conf, overlap_iou=0.5):
    """dets: list of (prompt, score, box). Picture-prompt boxes >= picture_conf become
    candidates; the overlapping 'person' score is attached for logging. Pure function."""
    pics = [d for d in dets if d[0] in picture_prompts and d[1] >= picture_conf]
    persons = [d for d in dets if d[0] == person_prompt]
    pics.sort(key=lambda d: -d[1])
    out = []
    for prompt, score, box in pics:
        if any(iou(box, o.box) > overlap_iou for o in out):
            continue                                       # same picture, other prompt
        ps = max([s for _, s, b in persons if iou(box, b) > overlap_iou], default=0.0)
        out.append(PictureDetection(tuple(float(x) for x in box), float(score), float(ps),
                                    prompt))
    return out


class PictureDetector:

    def __init__(self, model_path, model_dir, picture_prompts, person_prompt='person',
                 picture_conf=0.30, imgsz=480, threads=2, nice=10, logger=None):
        self.picture_prompts = list(picture_prompts)
        self.person_prompt = person_prompt
        self.picture_conf = float(picture_conf)
        self.imgsz = int(imgsz)
        self.prompts = self.picture_prompts + [person_prompt]
        self.log = logger

        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            os.environ.setdefault(v, str(threads))
        os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
        import torch
        torch.set_num_threads(int(threads))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass                                           # already set in this process
        try:
            os.nice(int(nice))
        except OSError:
            pass
        from ultralytics import YOLOWorld

        if not os.path.isabs(model_path):
            model_path = os.path.join(model_dir, model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        # ultralytics resolves the CLIP weights as ./weights/clip/ViT-B-32.pt relative to the
        # current directory; run from model_dir so the local copy is used (no download).
        if not os.path.isfile(os.path.join(model_dir, 'weights', 'clip', 'ViT-B-32.pt')):
            raise FileNotFoundError(f'{model_dir}/weights/clip/ViT-B-32.pt (CLIP text encoder)')
        cwd = os.getcwd()
        os.chdir(model_dir)
        try:
            self.model = YOLOWorld(model_path)
            self.model.set_classes(self.prompts)
        finally:
            os.chdir(cwd)

    def detect(self, img):
        low = min(self.picture_conf, 0.2)
        r = self.model.predict(img, imgsz=self.imgsz, conf=low, verbose=False)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for c, s, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                               r.boxes.xyxy.cpu().numpy()):
                dets.append((self.prompts[int(c)], float(s), np.asarray(b, dtype=float)))
        return combine(dets, self.picture_prompts, self.person_prompt, self.picture_conf)
