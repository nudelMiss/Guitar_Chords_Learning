import cv2
import numpy as np
import math
from collections import deque

# ===== Instrument mask calibration =====
# Base color limits for the instrument
LOWER_INST = np.array([0, 79, 138], dtype=np.uint8)
UPPER_INST = np.array([37, 248, 191], dtype=np.uint8)
K_OPEN = 1
K_CLOSE = 101

# ==== Auto V calibration =====
AUTO_V = True
V_MARGIN_LOW = 55
V_MARGIN_HIGH = 80

# ===== Tracking / stability =====
N_STRINGS = 6 
HISTORY = 15
Y_BIN = 9         
MIN_HITS = 3       
MIN_GOOD_LENGTH = 100  
# Live-tunable Hough/mask params (trackbars will update these)
MIN_LINE_LEN = 369
HOUGH_THRESH = 50

last_stable = []

""" Calculate the absolute angle between two points """
def angle_deg(x1, y1, x2, y2):
    return abs(math.degrees(math.atan2((y2 - y1), (x2 - x1))))

""" Isolate the instrument body from the background
        input: BGR image of the region of interest (bottom part of frame)
        output: binary mask of the instrument body """
def build_instrument_mask(imageBGR):
    hsv = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lower = LOWER_INST.copy()
    upper = UPPER_INST.copy()

    # Auto-calibrate of value range based on the median of the brighter pixels
    if AUTO_V:
        v_flat = v[v > 10]
        if v_flat.size > 0:
            med_v = int(np.median(v_flat))
            lower[2] = max(0, med_v - V_MARGIN_LOW)
            upper[2] = min(255, med_v + V_MARGIN_HIGH)

    mask = cv2.inRange(hsv, lower, upper)

    # Basic morphology to remove small specks
    # make sure kernel sizes are odd and at least 1
    k_open = max(1, int(K_OPEN))
    if k_open % 2 == 0:
        k_open += 1
    k_close = max(1, int(K_CLOSE))
    if k_close % 2 == 0:
        k_close += 1

    # Open to remove small noise, then Close to fill holes and connect the instrument body
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Find external contours and keep the largest one
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Select the largest contour by area
        cnt = max(contours, key=cv2.contourArea)
        MIN_AREA = 1000
        if cv2.contourArea(cnt) >= MIN_AREA:
            # Use convex hull to produce a clean filled mask of the instrument
            hull = cv2.convexHull(cnt)
            refined = np.zeros_like(mask)
            cv2.drawContours(refined, [hull], -1, 255, -1)

            # Slight closing to smooth edges and include thin neck regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)

            return refined

    return mask

def create_controls():
    # Create a small controls window with sliders for live tuning
    cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Controls', 400, 300)

    cv2.createTrackbar('LH', 'Controls', int(LOWER_INST[0]), 179, lambda x: None)
    cv2.createTrackbar('LS', 'Controls', int(LOWER_INST[1]), 255, lambda x: None)
    cv2.createTrackbar('LV', 'Controls', int(LOWER_INST[2]), 255, lambda x: None)

    cv2.createTrackbar('UH', 'Controls', int(UPPER_INST[0]), 179, lambda x: None)
    cv2.createTrackbar('US', 'Controls', int(UPPER_INST[1]), 255, lambda x: None)
    cv2.createTrackbar('UV', 'Controls', int(UPPER_INST[2]), 255, lambda x: None)

    cv2.createTrackbar('K_OPEN', 'Controls', K_OPEN, 31, lambda x: None)
    cv2.createTrackbar('K_CLOSE', 'Controls', K_CLOSE, 101, lambda x: None)

    cv2.createTrackbar('Brightness', 'Controls', 80, 255, lambda x: None)
    cv2.createTrackbar('HoughMinLen', 'Controls', MIN_LINE_LEN, 400, lambda x: None)
    cv2.createTrackbar('HoughThresh', 'Controls', HOUGH_THRESH, 500, lambda x: None)
    cv2.createTrackbar('MinHits', 'Controls', MIN_HITS, 15, lambda x: None)
    cv2.createTrackbar('MinGoodLen', 'Controls', MIN_GOOD_LENGTH, 300, lambda x: None)

""" Find the left and right boundaries of the instrument 
        input: binary mask of the instrument body in the region of interest and quantiles to ignore outliers
        output: x-coordinates of the left and right edges of the instrument (or None if not found) """
def estimate_x_limits_from_mask(mask_roi, left_q=0.05, right_q=0.98):
    pixels = np.where(mask_roi > 0)[1]
    if pixels.size == 0:
        return None, None
    x_left = int(np.quantile(pixels, left_q))
    x_right = int(np.quantile(pixels, right_q))
    if x_right <= x_left:
        return None, None
    return x_left, x_right

""" Ensure lines do not draw outside the detected instrument body
        input: line coordinates and x-limits
        output: line coordinates clipped to the x-limits """
def clip_line_to_x(line, x_left, x_right):
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

""" Check if the detected lines meet the minimum length requirements
        input: list of line coordinates, expected number of strings, minimum length for a line to be considered good
        output: boolean indicating if the detection is good enough to update the tracker """
def is_good_detection(lines, n_strings, min_len):
    if len(lines) < n_strings:
        return False

    lengths = []
    for x1, y1, x2, y2 in lines:
        lengths.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)

    good = sum(1 for L in lengths if L >= min_len)
    return good >= n_strings

class StringTracker:
    # Memory tracker to ensure strings are stable over multiple frames
    def __init__(self, n_strings, history, y_bin, min_hits):
        self.n_strings = n_strings
        self.history = history
        self.y_bin = y_bin
        self.min_hits = min_hits
        self.frames = deque(maxlen=history)

    """ Update the tracker with new detected lines
        aligning them to y-bins for stability analysis"""
    def update_lines(self, lines):
        aligned_lines = []
        for x1, y1, x2, y2 in lines:
            y_mid = int(0.5 * (y1 + y2))
            yb = int(round(y_mid / self.y_bin)) * self.y_bin
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            aligned_lines.append((yb, length, [x1, y1, x2, y2]))
        self.frames.append(aligned_lines)

    """ Analyze the history of detected lines to find stable candidates
        that appear consistently over multiple frames"""
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
            if hits >= self.min_hits:
                support = bin_support.get(yb, 0)
                candidates.append((hits, support, yb))

        if not candidates:
            return []

        candidates.sort(reverse=True)
        selected = sorted([yb for _, _, yb in candidates[:self.n_strings]])

        latest = self.frames[-1]
        out = []

        for yb in selected:
            near = [(length, line) for (yb2, length, line) in latest if abs(yb2 - yb) <= self.y_bin]
            if near:
                _, best = max(near, key=lambda t: t[0])
                out.append(best)

        out.sort(key=lambda l: 0.5 * (l[1] + l[3]))
        return out


tracker = StringTracker(N_STRINGS, HISTORY, Y_BIN, MIN_HITS)

""" Main function to detect strings in the bottom part of the frame using
    input: full camera frame, brightness threshold for edge detection, fraction of the frame to analyze from the bottom, and whether to update the tracker
    output: list of stable string line coordinates, thresholded debug image, and all detected horizontal line candidates for debugging """
def detect_strings_bottom(frame, brightness_thresh=43, bottom_fraction=0.5, update_tracker=True):
    global last_stable

    height = frame.shape[0]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    # Extract mask and body boundaries
    inst_mask = build_instrument_mask(bottom_frame)
    x_left, x_right = estimate_x_limits_from_mask(inst_mask)

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    
    # Apply the mask to black out the background completely
    gray = cv2.bitwise_and(gray, gray, mask=inst_mask)

    # 1. Switch to Sobel Y: This ONLY detects horizontal edges, ignoring vertical noise like frets
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    abs_sobely = cv2.convertScaleAbs(sobely)

    # 2. Thresholding
    _, thresh = cv2.threshold(abs_sobely, brightness_thresh, 255, cv2.THRESH_BINARY)

    # 3. Morphological CLOSE: Connect broken string segments into continuous horizontal lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

    # 4. Morphological OPEN: Erase small dots, wood texture noise, and leftover fret pieces
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

    # Hough Lines detection based on the cleaned image
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESH,
        minLineLength=MIN_LINE_LEN,    
        maxLineGap=20         
    )

    horizontal_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            a = angle_deg(x1, y1, x2, y2)
            a = min(a, 180 - a)
            
            # Keep only strictly horizontal lines (less than 10 degrees)
            if a < 10:
                horizontal_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    # Filter candidates to those that lie within the instrument mask and x-limits
    if horizontal_lines and inst_mask is not None:
        filtered = []
        h_mask, w_mask = inst_mask.shape
        for (x1, y1, x2, y2) in horizontal_lines:
            mid_x = int(round(0.5 * (x1 + x2)))
            mid_y = int(round(0.5 * (y1 + y2)))
            rel_y = mid_y - start_row
            if rel_y < 0 or rel_y >= h_mask or mid_x < 0 or mid_x >= w_mask:
                continue
            if inst_mask[rel_y, mid_x] == 0:
                continue
            if x_left is not None and x_right is not None:
                if mid_x < x_left or mid_x > x_right:
                    continue
            filtered.append([x1, y1, x2, y2])
        horizontal_lines = filtered

    horizontal_lines.sort(key=lambda l: l[1])

    if is_good_detection(horizontal_lines, N_STRINGS, MIN_GOOD_LENGTH):
        if update_tracker:
            tracker.update_lines(horizontal_lines)
        stable = tracker.get_stable()
        if len(stable) >= N_STRINGS:
            last_stable = stable[:N_STRINGS]

    out = []
    if last_stable and len(last_stable) == N_STRINGS:
        out = last_stable

    if x_left is not None and x_right is not None and out:
        out = [clip_line_to_x(line, x_left, x_right) for line in out]

    # Return the final stable lines, the thresholded debug image,
    # and all detected horizontal line candidates for debugging
    return out, thresh, horizontal_lines

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    locked_stable = None
    # Create interactive controls for live tuning
    create_controls()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Read trackbar values and update global params
        lh = cv2.getTrackbarPos('LH', 'Controls')
        ls = cv2.getTrackbarPos('LS', 'Controls')
        lv = cv2.getTrackbarPos('LV', 'Controls')
        uh = cv2.getTrackbarPos('UH', 'Controls')
        us = cv2.getTrackbarPos('US', 'Controls')
        uv = cv2.getTrackbarPos('UV', 'Controls')

        LOWER_INST[0] = np.uint8(lh)
        LOWER_INST[1] = np.uint8(ls)
        LOWER_INST[2] = np.uint8(lv)
        UPPER_INST[0] = np.uint8(uh)
        UPPER_INST[1] = np.uint8(us)
        UPPER_INST[2] = np.uint8(uv)

        kopen = cv2.getTrackbarPos('K_OPEN', 'Controls')
        kclose = cv2.getTrackbarPos('K_CLOSE', 'Controls')
        # enforce minimum and odd sizes for kernels
        global K_OPEN, K_CLOSE
        K_OPEN = max(1, kopen)
        if K_OPEN % 2 == 0:
            K_OPEN += 1
        K_CLOSE = max(1, kclose)
        if K_CLOSE % 2 == 0:
            K_CLOSE += 1

        # update Hough / threshold globals
        global MIN_LINE_LEN, HOUGH_THRESH, MIN_HITS, MIN_GOOD_LENGTH
        MIN_LINE_LEN = max(3, cv2.getTrackbarPos('HoughMinLen', 'Controls'))
        HOUGH_THRESH = max(1, cv2.getTrackbarPos('HoughThresh', 'Controls'))
        brightness = max(1, cv2.getTrackbarPos('Brightness', 'Controls'))
        MIN_HITS = max(1, cv2.getTrackbarPos('MinHits', 'Controls'))
        MIN_GOOD_LENGTH = max(20, cv2.getTrackbarPos('MinGoodLen', 'Controls'))

        # When locked, do not update the tracker so the locked lines remain fixed
        strings, debug_thresh, detected_lines = detect_strings_bottom(frame, brightness_thresh=brightness, update_tracker=(locked_stable is None))

        # Draw all detected horizontal candidate lines (blue)
        for x1, y1, x2, y2 in detected_lines:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw the lines if tracking is stable (green). If user locked lines,
        # draw the locked set instead and don't update the tracker.
        to_draw = locked_stable if locked_stable is not None else strings
        if to_draw and len(to_draw) == N_STRINGS:
            for x1, y1, x2, y2 in to_draw:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Overlay detection counts for quick debugging
        cv2.putText(frame, f'Detected candidates: {len(detected_lines)}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
        stable_count = len(to_draw) if to_draw else 0
        lock_state = 'LOCKED' if locked_stable is not None else 'AUTO'
        cv2.putText(frame, f'Stable strings: {stable_count} [{lock_state}]', (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f'MinHits: {MIN_HITS}, MinLen: {MIN_GOOD_LENGTH}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

        # Show the main camera
        cv2.imshow("Stable Strings Tracker", frame)
        
        # Show the debug view in a separate window
        cv2.imshow("DEBUG: What the computer sees", debug_thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # 's' = lock current stable strings; 'r' = release (auto)
        if key == ord('s'):
            if strings and len(strings) == N_STRINGS:
                locked_stable = [list(l) for l in strings]
                print(f"✓ LOCKED {len(strings)} strings")
            else:
                print(f"✗ Cannot lock: need exactly {N_STRINGS} strings, got {len(strings)}")
        if key == ord('r'):
            locked_stable = None
            print("✓ UNLOCKED - resuming auto-tracking")
        # 'p' = print current settings to terminal
        if key == ord('p'):
            print("\n" + "="*60)
            print("CURRENT SETTINGS")
            print("="*60)
            print(f"LOWER_INST = np.array([{lh}, {ls}, {lv}], dtype=np.uint8)")
            print(f"UPPER_INST = np.array([{uh}, {us}, {uv}], dtype=np.uint8)")
            print(f"K_OPEN = {K_OPEN}")
            print(f"K_CLOSE = {K_CLOSE}")
            print(f"brightness_thresh = {brightness}")
            print(f"MIN_LINE_LEN = {MIN_LINE_LEN}")
            print(f"HOUGH_THRESH = {HOUGH_THRESH}")
            print("="*60 + "\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()