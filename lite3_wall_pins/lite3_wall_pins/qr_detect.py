"""QR code detection with OpenCV only (no AI). Pure functions, no ROS.

Returns the decoded text AND the four image corners, so the node can measure the code's
3D position from depth inside exactly that quadrilateral.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class QrDetection:
    text: str
    corners: np.ndarray          # (4, 2) float, pixel coords in the decoded image

    @property
    def size_px(self):
        c = self.corners
        return float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))


class QrDetector:

    def __init__(self, try_upscale=True, min_size_px=12.0):
        # Both OpenCV detectors run and their results are merged by payload: in tests the
        # ArUco-based one found 2 codes side by side where the classic one found 1, but it
        # failed on slightly blurred codes that the classic one read.
        self.dets = [cv2.QRCodeDetectorAruco(), cv2.QRCodeDetector()]
        self.try_upscale = try_upscale
        self.min_size_px = min_size_px

    def detect(self, img):
        """img: BGR or gray image -> list[QrDetection] (only decoded codes)."""
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = []
        for det in self.dets:
            try:
                ok, texts, points, _ = det.detectAndDecodeMulti(gray)
            except cv2.error:
                ok, texts, points = False, (), None
            if not ok or points is None:
                continue
            for text, pts in zip(texts, points):
                corners = np.asarray(pts, dtype=float).reshape(4, 2)
                if not text and self.try_upscale:
                    text = self._decode_crop(gray, corners)
                if text:
                    d = QrDetection(text=text, corners=corners)
                    if d.size_px >= self.min_size_px:
                        out.append(d)
        # identical payload found twice in one frame (rare): keep the larger view
        best = {}
        for d in out:
            if d.text not in best or d.size_px > best[d.text].size_px:
                best[d.text] = d
        return list(best.values())

    def _decode_crop(self, gray, corners):
        """Codes that are found but too small/blurred to decode: retry on an upscaled,
        sharpened / binarised crop with both decoders. Measured on rendered 12 cm codes
        (real Lite3 focal length, blur + JPEG): extends reliable reading from ~0.5 m to
        ~1.0 m. Only runs for a found-but-undecoded code, so it costs nothing otherwise."""
        h, w = gray.shape[:2]
        x1, y1 = corners.min(axis=0)
        x2, y2 = corners.max(axis=0)
        pad = 0.3 * max(x2 - x1, y2 - y1) + 4
        x1, y1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        x2, y2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return ''
        crop = gray[y1:y2, x1:x2]
        for s in (2.0, 3.0):
            up = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            sharp = cv2.addWeighted(up, 1.8, cv2.GaussianBlur(up, (0, 0), s), -0.8, 0)
            _, otsu = cv2.threshold(cv2.GaussianBlur(up, (3, 3), 0), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for img in (up, sharp, otsu):
                for det in self.dets:
                    try:
                        text = det.detectAndDecode(img)[0]
                    except cv2.error:
                        continue
                    if text:
                        return text
        return ''
