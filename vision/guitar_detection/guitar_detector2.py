# new try to detect guitar by canny and hough transform
# by Michal and Yuval

import cv2
import numpy as np
import math
from collections import deque

# ===== Instrument mask calibration =====
# CALIBRATION: These values are a starting point for a typical ukulele. 
# You may need to adjust them based on your lighting and instrument color.
LOWER_INST = np.array([0, 57, 60], dtype=np.uint8)
UPPER_INST = np.array([166, 241, 245], dtype=np.uint8)
K_OPEN = 1
K_CLOSE = 21

# ==== Auto V calibration =====
AUTO_V = True
V_MARGIN_LOW = 55
V_MARGIN_HIGH = 80

# ===== Tracking / stability =====
N_STRINGS = 5          # 4 for ukulele / 6 for guitar
HISTORY = 15           # INCREASED: Keep more frames in memory for better stability
Y_BIN = 9              # INCREASED: Larger bins to allow for more variation in string position across frames
MIN_HITS = 7          # INCREASED: Require the string to appear more times to be considered stable
MIN_GOOD_LENGTH = 180  # minimum string length to consider "good"



last_stable = []


def angle_deg(x1, y1, x2, y2):
    # Calculate the angle between two points
    return abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))


def build_instrument_mask_roi(bgr_roi):
    # Create a mask to isolate the instrument
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lower = LOWER_INST.copy()
    upper = UPPER_INST.copy()

    if AUTO_V:
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

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

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

    return mask


def estimate_x_limits_from_mask(mask_roi, left_q=0.05, right_q=0.98):
    # Estimate the left and right boundaries of the instrument
    xs = np.where(mask_roi > 0)[1]
    if xs.size == 0:
        return None, None
    x_left = int(np.quantile(xs, left_q))
    x_right = int(np.quantile(xs, right_q))
    if x_right <= x_left:
        return None, None
    return x_left, x_right


def clip_line_to_x(line, x_left, x_right):
    # Clip lines so they don't draw outside the instrument body
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < 1e-6:
        y = int(0.5 * (y1 + y2))
        return [x_left, y, x_right, y]

    m = dy / dx
    y_left = int(round(y1 + m * (x_left - x1)))
    y_right = int(round(y1 + m * (x_right - x1)))

    return [x_left, y_left, x_right, y_right]


def is_good_detection(lines, n_strings, min_len):
    # Verify if we found enough lines that match the required length
    if len(lines) < n_strings:
        return False

    lengths = []
    for x1, y1, x2, y2 in lines:
        lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

    good = sum(1 for L in lengths if L >= min_len)
    return good >= n_strings


class StringTracker:
    def __init__(self, n_strings, history, y_bin, min_hits):
        self.n_strings = n_strings
        self.history = history
        self.y_bin = y_bin
        self.min_hits = min_hits
        self.frames = deque(maxlen=history)

    def update(self, lines):
        binned = []
        for x1, y1, x2, y2 in lines:
            y_mid = int(0.5 * (y1 + y2))
            yb = int(round(y_mid / self.y_bin)) * self.y_bin
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            binned.append((yb, length, [x1, y1, x2, y2]))
        self.frames.append(binned)

    def get_stable(self):
        if not self.frames:
            return []

        bin_hits = {}
        bin_support = {}

        for fi, frame_bins in enumerate(self.frames):
            seen = set()
            for yb, length, _ in frame_bins:
                seen.add(yb)
                bin_support[yb] = bin_support.get(yb, 0) + length
            for yb in seen:
                bin_hits.setdefault(yb, set()).add(fi)

        candidates = []
        for yb in bin_hits:
            hits = len(bin_hits[yb])
            if hits >= MIN_HITS:
                support = bin_support.get(yb, 0)
                candidates.append((hits, support, yb))

        if not candidates:
            return []

        candidates.sort(reverse=True)
        selected = sorted([yb for _, _, yb in candidates[:self.n_strings]])

        latest = self.frames[-1]
        out = []

        for yb in selected:
            near = [(length, line) for (yb2, length, line) in latest if abs(yb2 - yb) <= Y_BIN]
            if near:
                _, best = max(near, key=lambda t: t[0])
                out.append(best)

        out.sort(key=lambda l: 0.5 * (l[1] + l[3]))
        return out


tracker = StringTracker(N_STRINGS, HISTORY, Y_BIN, MIN_HITS)


def detect_strings_bottom(frame, brightness_thresh=50, bottom_fraction=0.5):
    global last_stable

    height = frame.shape[0]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    inst_mask = build_instrument_mask_roi(bottom_frame)
    x_left, x_right = estimate_x_limits_from_mask(inst_mask)

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. CHANGED: Switched from horizontal Sobel to Canny Edge Detection
    # Canny detects edges at ANY angle perfectly.
    thresh = cv2.Canny(gray, brightness_thresh, brightness_thresh * 3)

    # 2. CHANGED: Replaced the horizontal kernel (31, 1) with a square one (5, 5)
    # This prevents angled strings from being broken up by the morphology operation.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=80,    # Slightly reduced to catch angled strings easier
        maxLineGap=20
    )

    horizontal_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            a = angle_deg(x1, y1, x2, y2)
            a = min(a, 180 - a)
            
            # 3. CHANGED: Increased the allowed angle from 12 to 35 degrees!
            # Now the algorithm won't ignore strings when you tilt the ukulele.
            if a < 35:
                horizontal_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    horizontal_lines.sort(key=lambda l: l[1])

    if is_good_detection(horizontal_lines, N_STRINGS, MIN_GOOD_LENGTH):
        tracker.update(horizontal_lines)
        stable = tracker.get_stable()
        if len(stable) == N_STRINGS:
            last_stable = stable

    out = []
    if last_stable and len(last_stable) == N_STRINGS:
        out = last_stable

    if x_left is not None and x_right is not None and out:
        out = [clip_line_to_x(line, x_left, x_right) for line in out]

    return out


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        strings = detect_strings_bottom(frame)

        # CHANGED: Draw lines only if we have exactly the required number, and make them THICKER (5 instead of 2)
        if len(strings) == N_STRINGS:
            for x1, y1, x2, y2 in strings:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv2.imshow("Stable Strings", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()