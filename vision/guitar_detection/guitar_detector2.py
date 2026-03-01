# new try to detect guitar by canny and hough transform
# by Michal and Yuval

import cv2
import numpy as np
import math
from collections import deque

# ===== Instrument mask calibration (from your tuner) =====
LOWER_INST = np.array([0, 57, 60], dtype=np.uint8)
UPPER_INST = np.array([166, 241, 245], dtype=np.uint8)
K_OPEN = 1
K_CLOSE = 21

# ===== Tracking / stability =====
N_STRINGS = 4          # 4 ukulele / 6 guitar (set once)
HISTORY = 8            # number of recent frames to use for voting
Y_BIN = 6              # pixels: y quantization (bigger = more stable, less precise)
MIN_HITS = 3           # bin must appear in at least this many frames to be considered

AUTO_V = True
V_MARGIN_LOW = 55   # how far below median V to allow
V_MARGIN_HIGH = 80  # how far above median V to allow



def angle_deg(x1, y1, x2, y2):
    return abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))


def build_instrument_mask_roi(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lower = LOWER_INST.copy()
    upper = UPPER_INST.copy()

    if AUTO_V:
        # estimate median brightness in ROI (ignore extreme dark pixels)
        v_flat = v[v > 10]
        if v_flat.size > 0:
            med_v = int(np.median(v_flat))
            lower[2] = max(0, med_v - V_MARGIN_LOW)
            upper[2] = min(255, med_v + V_MARGIN_HIGH)

    mask = cv2.inRange(hsv, lower, upper)

    k_open = max(1, K_OPEN | 1)
    k_close = max(1, K_CLOSE | 1)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num > 1:
        best_label = 1
        best_score = -1.0
        MIN_AREA = 2000

        for lbl in range(1, num):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue
            cx = centroids[lbl][0]
            score = area + 0.0008 * cx * area
            if score > best_score:
                best_score = score
                best_label = lbl

        mask = (labels == best_label).astype(np.uint8) * 255

        x = stats[best_label, cv2.CC_STAT_LEFT]
        y = stats[best_label, cv2.CC_STAT_TOP]
        w = stats[best_label, cv2.CC_STAT_WIDTH]
        h = stats[best_label, cv2.CC_STAT_HEIGHT]

        clean = np.zeros_like(mask)
        clean[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
        mask = clean

    return mask


class StringTracker:
    """
    Tracks string y-positions over time using bin-voting.
    Stores candidates per frame, then chooses the most stable N_STRINGS bins.
    """
    def __init__(self, n_strings=4, history=8, y_bin=6, min_hits=3):
        self.n_strings = n_strings
        self.history = history
        self.y_bin = y_bin
        self.min_hits = min_hits
        self.frames = deque(maxlen=history)  # each item: list of (y_bin, line)

    def update(self, lines):
        # lines are [x1,y1,x2,y2] in FULL frame coords
        binned = []
        for x1, y1, x2, y2 in lines:
            y_mean = int(round(0.5 * (y1 + y2)))
            yb = int(round(y_mean / self.y_bin)) * self.y_bin
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            binned.append((yb, length, [x1, y1, x2, y2]))
        self.frames.append(binned)

    def get_stable(self):
        if not self.frames:
            return []

        # vote bins by number of frames they appear in + total length support
        bin_hits = {}     # yb -> set(frame_idx)
        bin_support = {}  # yb -> total length

        for fi, frame_bins in enumerate(self.frames):
            seen_in_frame = set()
            for yb, length, _line in frame_bins:
                seen_in_frame.add(yb)
                bin_support[yb] = bin_support.get(yb, 0.0) + length
            for yb in seen_in_frame:
                bin_hits.setdefault(yb, set()).add(fi)

        candidates = []
        for yb in bin_hits:
            hits = len(bin_hits[yb])
            if hits >= self.min_hits:
                support = bin_support.get(yb, 0.0)
                # score: prefer bins seen in more frames, then higher support
                candidates.append((hits, support, yb))

        if not candidates:
            return []

        # pick top bins
        candidates.sort(reverse=True)  # by hits, support
        selected_bins = sorted([yb for _, _, yb in candidates[:self.n_strings]])

        # for each selected bin, pick the best current line close to it (from latest frame),
        # otherwise fall back to best from history
        latest = self.frames[-1]
        out = []

        for yb in selected_bins:
            # prefer latest frame line near this bin
            near = [(length, line) for (yb2, length, line) in latest if abs(yb2 - yb) <= self.y_bin]
            if near:
                _, best_line = max(near, key=lambda t: t[0])  # longest in latest
                out.append(best_line)
                continue

            # fallback: best from history
            best = None
            best_len = -1
            for frame_bins in reversed(self.frames):
                for yb2, length, line in frame_bins:
                    if abs(yb2 - yb) <= self.y_bin and length > best_len:
                        best_len = length
                        best = line
            if best is not None:
                out.append(best)

        out.sort(key=lambda l: 0.5 * (l[1] + l[3]))
        return out


tracker = StringTracker(n_strings=N_STRINGS, history=HISTORY, y_bin=Y_BIN, min_hits=MIN_HITS)


def detect_strings_bottom(frame, brightness_thresh=80, bottom_fraction=0.5, debug_mask=False):
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))

    bottom_frame = frame[start_row:, :]

    inst_mask = build_instrument_mask_roi(bottom_frame)

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)

    _, thresh = cv2.threshold(sobely, brightness_thresh, 255, cv2.THRESH_BINARY)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h)

    thresh = cv2.bitwise_and(thresh, inst_mask)

    if debug_mask:
        cv2.imshow("Instrument Mask ROI", inst_mask)
        cv2.imshow("Thresh (masked)", thresh)

    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=120,
        maxLineGap=10
    )

    horizontal_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            a = angle_deg(x1, y1, x2, y2)
            a = min(a, 180 - a)
            if a < 12 and abs(x2 - x1) > 40:
                horizontal_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    horizontal_lines.sort(key=lambda l: l[1])

    # ===== temporal stabilization =====
    tracker.update(horizontal_lines)
    stable = tracker.get_stable()

    # If we don't have enough stable lines yet (startup), return raw
    if len(stable) >= 2:
        return stable
    return horizontal_lines


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    brightness_thresh = 80
    bottom_fraction = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        strings = detect_strings_bottom(frame, brightness_thresh, bottom_fraction, debug_mask=False)

        for x1, y1, x2, y2 in strings:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Detected Strings (Masked ROI)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('d'):
            _ = detect_strings_bottom(frame, brightness_thresh, bottom_fraction, debug_mask=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()