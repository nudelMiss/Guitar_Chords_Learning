#!/usr/bin/env python3
import cv2
import numpy as np

FINGER_NAMES = ["index", "middle", "ring", "pinky"]

class _EMA:
    def __init__(self, alpha=0.65):
        self.alpha = float(alpha)
        self.state = None

    def __call__(self, xy):
        xy = np.asarray(xy, dtype=np.float32)
        if self.state is None:
            self.state = xy
        else:
            self.state = self.alpha * xy + (1.0 - self.alpha) * self.state
        return self.state


def _cluster_ys(ys, bin_px=6):
    """Cluster y values by binning; return sorted cluster centers."""
    if len(ys) == 0:
        return []
    ys = np.array(sorted(ys), dtype=np.int32)
    clusters = []
    cur = [int(ys[0])]
    for y in ys[1:]:
        if abs(int(y) - cur[-1]) <= bin_px:
            cur.append(int(y))
        else:
            clusters.append(int(np.median(cur)))
            cur = [int(y)]
    clusters.append(int(np.median(cur)))
    return sorted(clusters)


def _detect_strings_y(gray, n_strings=6):
    """
    Detect horizontal string lines and return y positions (in the input image coordinates).
    Works best when the neck is roughly horizontal and strings are visible.
    """
    # Enhance horizontal edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)

    h, w = gray.shape[:2]
    min_len = int(0.45 * w)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=15,
    )

    ys = []
    if lines is None:
        return [], edges

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dy) > 3:  # near-horizontal only
            continue
        length = np.hypot(dx, dy)
        if length < min_len:
            continue
        ys.append(int((y1 + y2) * 0.5))

    # Cluster and keep the best n_strings (spread across the neck)
    clusters = _cluster_ys(ys, bin_px=6)
    if len(clusters) <= n_strings:
        return clusters, edges

    # If too many, choose n_strings that are roughly evenly spaced
    clusters = sorted(clusters)
    idxs = np.linspace(0, len(clusters) - 1, n_strings).round().astype(int)
    chosen = [clusters[i] for i in idxs]
    chosen = sorted(list(dict.fromkeys(chosen)))  # unique, keep order
    return chosen, edges


class HandDetector:
    """
    Classical-CV fingertip detector for fretting hand, no deep learning.
    Drop-in replacement for your existing MediaPipe-based HandDetector.

    Returns 4 fingertips (index/middle/ring/pinky), assigned by left->right order.
    """

    def __init__(
        self,
        n_strings=6,
        y_margin=55,            # ROI padding above/below strings
        x_crop_left=0.10,       # ignore headstock area
        x_crop_right=0.06,      # ignore body/soundhole area
        ema_alpha=0.65,
        min_area=450,
        max_area=18000,
        require_near_string=True,
        near_string_px=22,
    ):
        self.n_strings = int(n_strings)
        self.y_margin = int(y_margin)
        self.x_crop_left = float(x_crop_left)
        self.x_crop_right = float(x_crop_right)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.require_near_string = bool(require_near_string)
        self.near_string_px = int(near_string_px)

        self._ema = {name: _EMA(alpha=ema_alpha) for name in FINGER_NAMES}

        # debug state
        self.last_roi = None
        self.last_mask = None
        self.last_edges = None
        self.last_strings_y = None
        self.last_fingertips_px = None

    def _make_neck_roi(self, frame_bgr, strings_y):
        h, w = frame_bgr.shape[:2]
        x1 = int(self.x_crop_left * w)
        x2 = int((1.0 - self.x_crop_right) * w)

        if len(strings_y) >= 2:
            y1 = max(0, int(min(strings_y) - self.y_margin))
            y2 = min(h, int(max(strings_y) + self.y_margin))
        else:
            # fallback: center band
            y1 = int(0.30 * h)
            y2 = int(0.72 * h)

        return x1, y1, x2, y2

    def _skin_mask_ycrcb(self, roi_bgr):
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)

        # Reasonable default skin range; you may tune these once and keep them.
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)

        mask = cv2.inRange(ycrcb, lower, upper)

        # Clean up
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

        # Slight blur to soften jaggies
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 80, 255, cv2.THRESH_BINARY)
        return mask

    def _extract_fingertips(self, mask, strings_y_local=None):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        tips = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if h < 18 or w < 18:
                continue

            # fingertip candidate: top-most point (min y) on contour hull
            hull = cv2.convexHull(c)
            pts = hull.reshape(-1, 2)
            tip = pts[np.argmin(pts[:, 1])]  # [x, y] with min y
            tx, ty = int(tip[0]), int(tip[1])

            if self.require_near_string and strings_y_local and len(strings_y_local) > 0:
                d = min(abs(ty - sy) for sy in strings_y_local)
                if d > self.near_string_px:
                    continue

            tips.append((tx, ty, area))

        # Keep at most 4 best (largest blobs tend to be fingers)
        tips.sort(key=lambda t: t[2], reverse=True)
        tips = tips[:4]

        # Sort left -> right
        tips.sort(key=lambda t: t[0])
        return [(t[0], t[1]) for t in tips]

    def detect_fingers_and_frets(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        strings_y, edges = _detect_strings_y(gray, n_strings=self.n_strings)
        self.last_strings_y = strings_y
        self.last_edges = edges

        x1, y1, x2, y2 = self._make_neck_roi(frame_bgr, strings_y)
        roi = frame_bgr[y1:y2, x1:x2].copy()
        self.last_roi = (x1, y1, x2, y2)

        mask = self._skin_mask_ycrcb(roi)
        self.last_mask = mask

        # Convert global strings_y to ROI-local y for near-string filtering
        strings_y_local = []
        for sy in strings_y:
            if y1 <= sy < y2:
                strings_y_local.append(int(sy - y1))

        tips_xy = self._extract_fingertips(mask, strings_y_local=strings_y_local)

        # Assign names by left->right order and smooth
        named = []
        for i, (x, y) in enumerate(tips_xy):
            if i >= len(FINGER_NAMES):
                break
            name = FINGER_NAMES[i]
            xs, ys = self._ema[name]([x, y])
            named.append((name, int(xs + x1), int(ys + y1)))  # back to full-frame coords

        self.last_fingertips_px = named
        return len(named), named

    def draw_debug(self, frame_bgr):
        out = frame_bgr.copy()

        # draw ROI
        if self.last_roi is not None:
            x1, y1, x2, y2 = self.last_roi
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw detected strings
        if self.last_strings_y:
            for sy in self.last_strings_y:
                cv2.line(out, (0, sy), (out.shape[1] - 1, sy), (255, 255, 255), 1)

        # draw fingertips
        if self.last_fingertips_px:
            for name, x, y in self.last_fingertips_px:
                cv2.circle(out, (x, y), 8, (255, 0, 0), -1)

                # white text with black outline (readable on dark neck)
                cv2.putText(out, name, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(out, name, (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return out