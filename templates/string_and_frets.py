import cv2
import numpy as np


# --- 1. Classical Vision: Fret Detection (Improved Threshold Stability) ---
def detect_frets_bottom(frame, brightness_thresh=120, bottom_fraction=0.4):
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)

    # Detect vertical edges (frets)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    # === CHANGE: adaptive threshold using Otsu instead of fixed brightness_thresh ===
    _, thresh = cv2.threshold(
        sobelx,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect line segments
    lines = cv2.HoughLinesP(
        thresh,
        1,
        np.pi / 180,
        30,
        minLineLength=15,
        maxLineGap=5
    )

    vertical_lines = []

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]

            # Keep near-vertical lines only
            if abs(x2 - x1) < 5 and (y2 - y1) > 10:
                vertical_lines.append([
                    x1,
                    y1 + start_row,
                    x2,
                    y2 + start_row
                ])

    return vertical_lines


# --- 2. Classical Vision: Neck Bounds (Untouched) ---
def detect_guitar_neck_bounds(frame, bottom_fraction=0.4):
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    gray = cv2.cvtColor(frame[start_row:, :], cv2.COLOR_BGR2GRAY)

    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    _, thresh = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=width // 4, maxLineGap=50)

    y_coords = []
    if lines is not None:
        for l in lines:
            if abs(l[0][3] - l[0][1]) < 15:
                y_coords.append(((l[0][1] + l[0][3]) // 2) + start_row)

    if len(y_coords) >= 2:
        y_coords.sort()
        return y_coords[0], y_coords[-1]
    return None, None


def merge_vertical_lines(lines, x_threshold=20):
    if not lines: return []
    lines.sort(key=lambda l: l[0])
    groups = []
    if len(lines) > 0:
        curr = [lines[0]]
        for i in range(1, len(lines)):
            if abs(lines[i][0] - curr[-1][0]) <= x_threshold:
                curr.append(lines[i])
            else:
                groups.append(curr)
                curr = [lines[i]]
        groups.append(curr)
    return [[int(np.mean([l[0] for l in g])), min(l[1] for l in g),
             int(np.mean([l[0] for l in g])), max(l[3] for l in g)] for g in groups]


# --- 3. String Detection Logic ---
def detect_strings_in_neck(frame, locked_model):
    """
    Analyzes the horizontal intensity profile between neck bounds
    to find the 6 guitar strings.
    """
    y_t, y_b = locked_model['y_t'], locked_model['y_b']
    x_min, x_max = locked_model['x_min'], locked_model['x_max']

    neck_roi = frame[y_t:y_b, x_min:x_max]
    if neck_roi.size == 0: return []

    gray_neck = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
    gray_neck = cv2.GaussianBlur(gray_neck, (5, 5), 0)

    # Calculate average intensity per row
    projection = cv2.reduce(gray_neck, 1, cv2.REDUCE_AVG).flatten()
    num_rows = len(projection)
    step = num_rows // 7
    string_y_rel = []

    # Search for dark peaks (valleys) in expected regions
    for i in range(1, 7):
        center = i * step
        search_range = 12
        low, high = max(0, center - search_range), min(num_rows, center + search_range)
        search_area = projection[low:high]
        if len(search_area) > 0:
            local_min = np.argmin(search_area) + low
            string_y_rel.append(local_min / num_rows)

    return string_y_rel


# --- 4. Main Application ---
def main():
    cap = cv2.VideoCapture(0)

    # State Variables
    is_tracking = False
    show_string_error = False
    tracking_pts = None
    initial_pts = None
    fret_model_rel = []
    string_model_rel = []
    last_gray = None
    locked_model = {}

    print("--- Guitar Tracker Loaded ---")
    print("Commands: [c] Calibrate Neck, [s] Detect Strings, [x] Reset Strings, [r] Reset All, [q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        # --- Keyboard Logic ---
        if key == ord('q'): break

        if key == ord('r'):  # Reset everything
            is_tracking = False
            string_model_rel = []
            show_string_error = False
            print("System Reset.")

        if key == ord('x'):  # Reset strings only
            string_model_rel = []
            show_string_error = False
            print("Strings Cleared.")

        # Attempt string detection (Only if tracking is active)
        if key == ord('s') and is_tracking:
            detected = detect_strings_in_neck(frame, locked_model)
            if len(detected) == 6:
                string_model_rel = detected
                show_string_error = False
                print("Success: 6 Strings Calibrated.")
            else:
                string_model_rel = []
                show_string_error = True
                print(f"Failed: Only {len(detected)} strings found. Need exactly 6.")

        if is_tracking and tracking_pts is not None:
            if is_tracking and tracking_pts is not None:
                # --- TRACKING ENGINE (Updated for Stability) ---
                new_pts, status, _ = cv2.calcOpticalFlowPyrLK(last_gray, gray, tracking_pts, None)
                good_new = new_pts[status.flatten() == 1]
                good_old = initial_pts[status.flatten() == 1]

                # Initialize a persistent matrix in the model to avoid flickering
                if 'last_matrix' not in locked_model:
                    locked_model['last_matrix'] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
                # Try to update the transformation only if enough points are tracked
                if len(good_new) >= 6:
                    # Use RANSAC to find the neck's global movement and filter out hand motion
                    matrix, inliers = cv2.estimateAffine2D(good_old, good_new,
                                                                  method=cv2.RANSAC,
                                                                  ransacReprojThreshold=3)

                    if matrix is not None:
                        # Consensus Check: How many points agree on this specific transformation?
                        inlier_ratio = np.sum(inliers) / len(good_new)

                        # Only update the scale/position if more than 50% of points move together
                        # This prevents the hand from distorting the fretboard scale
                        if inlier_ratio > 0.5:
                            locked_model['last_matrix'] = matrix
                            tracking_pts = cv2.transform(initial_pts, matrix)
                            last_gray = gray.copy()

                # Always use the last validated matrix for drawing (prevents grid from disappearing)
                curr_matrix = locked_model['last_matrix']

                # Transform neck corners
                corners = np.array([
                    [locked_model['x_min'], locked_model['y_t']],
                    [locked_model['x_max'], locked_model['y_t']],
                    [locked_model['x_min'], locked_model['y_b']],
                    [locked_model['x_max'], locked_model['y_b']]
                ], dtype=np.float32).reshape(-1, 1, 2)

                tracked_corners = cv2.transform(corners, curr_matrix).reshape(-1, 2)
                tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

                # 1. Draw Frets (Green)
                for rel_x in fret_model_rel:
                    p1 = (tl + rel_x * (tr - tl)).astype(int)
                    p2 = (bl + rel_x * (br - bl)).astype(int)
                    cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 0), 2)

                # 2. Draw Strings (Yellow)
                for rel_y in string_model_rel:
                    p1 = (tl + rel_y * (bl - tl)).astype(int)
                    p2 = (tr + rel_y * (br - tr)).astype(int)
                    cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 255), 1)

                # 3. Draw Neck Boundaries (Blue)
                cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)

            # Display pop-up alert if string count is wrong
            if show_string_error:
                cv2.putText(display_frame, "STRINGS FAILED - TRY AGAIN (S)", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # --- PREVIEW/CALIBRATION MODE ---
            raw_f = detect_frets_bottom(frame)
            y_t, y_b = detect_guitar_neck_bounds(frame)

            if y_t is not None:
                cv2.line(display_frame, (0, y_t), (width, y_t), (0, 0, 255), 1)
                cv2.line(display_frame, (0, y_b), (width, y_b), (0, 0, 255), 1)

            # Lock Neck Calibration
            if key == ord('c') and y_t is not None and len(raw_f) > 2:
                stable_f = merge_vertical_lines(raw_f)
                x_min, x_max = stable_f[0][0], stable_f[-1][0]

                # Create point grid for LK Tracking
                grid_x = np.linspace(x_min, x_max, 10)
                grid_y = np.linspace(y_t, y_b, 4)
                temp_pts = [[gx, gy] for gx in grid_x for gy in grid_y]

                initial_pts = np.array(temp_pts, dtype=np.float32).reshape(-1, 1, 2)
                tracking_pts = initial_pts.copy()

                locked_model = {'x_min': x_min, 'x_max': x_max, 'y_t': y_t, 'y_b': y_b}
                fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]

                last_gray = gray.copy()
                is_tracking = True
                print("Model Locked! Press 's' to calibrate strings.")

        cv2.imshow("Robust Guitar Fret & String Tracker", display_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()