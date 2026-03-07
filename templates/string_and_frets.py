import cv2
import numpy as np


# --- 1. Fret Detection (Vertical Edges) ---
def detect_frets_bottom(frame, bottom_fraction=0.4):
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)

    # Use Sobel X to find vertical edges (frets)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    # Adaptive thresholding using Otsu's method
    _, thresh = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cleanup noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=5)

    vertical_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Filter for near-vertical lines
            if abs(x2 - x1) < 5 and (y2 - y1) > 10:
                vertical_lines.append([x1, y1 + start_row, x2, y2 + start_row])
    return vertical_lines


# --- 2. Neck Bounds Detection (Horizontal Edges) ---
def detect_guitar_neck_bounds(frame, bottom_fraction=0.4):
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    gray = cv2.cvtColor(frame[start_row:, :], cv2.COLOR_BGR2GRAY)

    # Use Sobel Y to find horizontal edges (neck edges)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)
    _, thresh = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 50, minLineLength=width // 4, maxLineGap=50)

    y_coords = []
    if lines is not None:
        for l in lines:
            if abs(l[0][3] - l[0][1]) < 15:  # Filter for horizontal lines
                y_coords.append(((l[0][1] + l[0][3]) // 2) + start_row)

    if len(y_coords) >= 2:
        y_coords.sort()
        return y_coords[0], y_coords[-1]
    return None, None


# --- 3. Merge overlapping fret lines ---
def merge_vertical_lines(lines, x_threshold=20):
    if not lines: return []
    lines.sort(key=lambda l: l[0])
    groups = []
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


# --- 4. String Detection (Advanced Edge Projection) ---
def detect_strings_in_neck(frame, locked_model):
    """
    Finds strings by analyzing vertical gradients (Sobel Y) across the neck.
    Robust against changes in absolute brightness.
    """
    y_t, y_b = int(locked_model['y_t']), int(locked_model['y_b'])
    x_min, x_max = int(locked_model['x_min']), int(locked_model['x_max'])

    neck_roi = frame[y_t:y_b, x_min:x_max]
    if neck_roi.size == 0: return []

    gray_neck = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)

    # Sobel Y highlights horizontal edges (strings)
    grad_y = cv2.Sobel(gray_neck, cv2.CV_16S, 0, 1, ksize=3)
    grad_y = cv2.convertScaleAbs(grad_y)
    grad_y = cv2.GaussianBlur(grad_y, (3, 3), 0)

    # Projection: Sum of gradients across each row
    projection = cv2.reduce(grad_y, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F).flatten()

    num_rows = len(projection)
    # Search bands for the 6 strings
    step = num_rows // 7
    string_y_rel = []
    search_range = 15

    for i in range(1, 7):
        center = i * step
        low, high = max(0, center - search_range), min(num_rows, center + search_range)
        search_area = projection[low:high]

        if len(search_area) > 0:
            # Strings are Peaks in the gradient projection
            local_max = np.argmax(search_area) + low
            string_y_rel.append(local_max / num_rows)

    return string_y_rel


# --- 5. Main Application ---
def main():
    cap = cv2.VideoCapture(0)

    # State Variables
    is_tracking = False
    tracking_pts = None
    initial_pts = None
    fret_model_rel = []
    string_model_rel = []
    last_gray = None
    locked_model = {}

    # Smoothing parameters for the visual grid
    smoothed_corners = None
    ALPHA = 0.4  # Lower = smoother but slower tracking. Higher = faster but jittery.

    print("--- Pro Guitar Tracker ---")
    print("Commands: [c] Calibrate, [s] Strings, [r] Reset, [q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): break
        if key == ord('r'):
            is_tracking = False
            string_model_rel = []
            smoothed_corners = None
            print("System Reset.")

        # --- TRACKING ENGINE ---
        if is_tracking and tracking_pts is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(last_gray, gray, tracking_pts, None)

            if new_pts is not None:
                good_new = new_pts[status.flatten() == 1]
                good_old = initial_pts[status.flatten() == 1]

                if len(good_new) >= 6:
                    # Robust transform calculation using RANSAC
                    matrix, inliers = cv2.estimateAffine2D(good_old, good_new, method=cv2.RANSAC,
                                                           ransacReprojThreshold=3)

                    if matrix is not None:
                        inlier_ratio = np.sum(inliers) / len(good_new)
                        if inlier_ratio > 0.4:
                            locked_model['last_matrix'] = matrix
                            tracking_pts = cv2.transform(initial_pts, matrix)
                            last_gray = gray.copy()

                curr_matrix = locked_model.get('last_matrix', np.eye(2, 3, dtype=np.float32))

                # Corner definitions for current model
                corners = np.array([
                    [locked_model['x_min'], locked_model['y_t']],
                    [locked_model['x_max'], locked_model['y_t']],
                    [locked_model['x_min'], locked_model['y_b']],
                    [locked_model['x_max'], locked_model['y_b']]
                ], dtype=np.float32).reshape(-1, 1, 2)

                # Apply smoothing to corners to prevent visual jitter
                raw_corners = cv2.transform(corners, curr_matrix).reshape(-1, 2)
                if smoothed_corners is None:
                    smoothed_corners = raw_corners
                else:
                    smoothed_corners = ALPHA * raw_corners + (1 - ALPHA) * smoothed_corners

                tl, tr, bl, br = smoothed_corners

                # String Calibration Trigger (During tracking)
                if key == ord('s'):
                    # Create a temporary model of the current tracked state
                    current_view_model = {
                        'x_min': min(tl[0], bl[0]), 'x_max': max(tr[0], br[0]),
                        'y_t': min(tl[1], tr[1]), 'y_b': max(bl[1], br[1])
                    }
                    detected = detect_strings_in_neck(frame, current_view_model)
                    if len(detected) == 6:
                        string_model_rel = detected
                        print("Strings calibrated.")
                    else:
                        print(f"Failed strings: found {len(detected)}")

                # --- DRAWING ---
                # Draw Frets (Green)
                for rel_x in fret_model_rel:
                    p1 = (tl + rel_x * (tr - tl)).astype(int)
                    p2 = (bl + rel_x * (br - bl)).astype(int)
                    cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 0), 2)

                # Draw Strings (Yellow)
                for rel_y in string_model_rel:
                    p1 = (tl + rel_y * (bl - tl)).astype(int)
                    p2 = (tr + rel_y * (br - tr)).astype(int)
                    cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 255), 1)

                # Draw Neck Bounds (Blue)
                cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)

        else:
            # --- PREVIEW / CALIBRATION MODE ---
            raw_f = detect_frets_bottom(frame)
            y_t, y_b = detect_guitar_neck_bounds(frame)

            if y_t is not None:
                cv2.line(display_frame, (0, y_t), (width, y_t), (0, 0, 255), 1)
                cv2.line(display_frame, (0, y_b), (width, y_b), (0, 0, 255), 1)

            if key == ord('c') and y_t is not None and len(raw_f) > 2:
                stable_f = merge_vertical_lines(raw_f)
                x_min, x_max = stable_f[0][0], stable_f[-1][0]

                # Intelligent Point Selection: Track texture/corners instead of a grid
                roi_gray = gray[y_t:y_b, x_min:x_max]
                features = cv2.goodFeaturesToTrack(roi_gray, maxCorners=50, qualityLevel=0.01, minDistance=10)

                if features is not None:
                    # Offset features to full frame coordinates
                    features[:, 0, 0] += x_min
                    features[:, 0, 1] += y_t
                    initial_pts = features.astype(np.float32)
                    tracking_pts = initial_pts.copy()

                    locked_model = {'x_min': x_min, 'x_max': x_max, 'y_t': y_t, 'y_b': y_b}
                    fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]

                    last_gray = gray.copy()
                    is_tracking = True
                    smoothed_corners = None
                    print("Calibration Locked. Now press 's' for strings.")

        cv2.imshow("Advanced Guitar Tracker", display_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()