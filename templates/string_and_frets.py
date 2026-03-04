import cv2
import numpy as np


# --- 1. Classical Vision: Fret Detection (לא נגעתי - עובד מעולה) ---
def detect_frets_bottom(frame, brightness_thresh=120, bottom_fraction=0.4):
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)
    _, thresh = cv2.threshold(sobelx, brightness_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, minLineLength=15, maxLineGap=5)

    vertical_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            if abs(x2 - x1) < 5 and (y2 - y1) > 10:
                vertical_lines.append([x1, y1 + start_row, x2, y2 + start_row])
    return vertical_lines


# --- 2. Classical Vision: Neck Bounds (לא נגעתי) ---
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


# --- 3. String Detection (שיפור קל ליציבות) ---
def detect_strings_in_neck(frame, locked_model):
    y_t, y_b = locked_model['y_t'], locked_model['y_b']
    x_min, x_max = locked_model['x_min'], locked_model['x_max']

    # חיתוך הצוואר ושיפור ניגודיות
    neck_roi = frame[y_t:y_b, x_min:x_max]
    if neck_roi.size == 0: return []

    gray_neck = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
    gray_neck = cv2.GaussianBlur(gray_neck, (5, 5), 0)  # החלקה להורדת רעשים

    # היסטוגרמת בהירות
    projection = cv2.reduce(gray_neck, 1, cv2.REDUCE_AVG).flatten()

    num_rows = len(projection)
    step = num_rows // 7
    string_y_rel = []

    for i in range(1, 7):
        center = i * step
        search_range = 10  # מחפש בטווח קטן סביב המקום המשוער של המיתר
        low = max(0, center - search_range)
        high = min(num_rows, center + search_range)

        search_area = projection[low:high]
        if len(search_area) > 0:
            local_min = np.argmin(search_area) + low
            string_y_rel.append(local_min / num_rows)

    return string_y_rel


# --- 4. Main Loop ---
def main():
    cap = cv2.VideoCapture(0)

    is_tracking = False
    tracking_pts = None
    initial_pts = None
    fret_model_rel = []
    string_model_rel = []
    last_gray = None
    locked_model = {}

    print("Controls: 'c' - Calibrate Neck, 's' - Detect Strings (after 'c'), 'r' - Reset, 'q' - Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        # לוגיקה של מקשים (מחוץ ל-IF/ELSE כדי שיעבדו תמיד)
        if key == ord('q'): break

        if key == ord('r'):
            is_tracking = False
            string_model_rel = []
            print("Resetting...")

        # לחיצה על S תעבוד רק אם אנחנו כבר עוקבים אחרי הצוואר
        if key == ord('s') and is_tracking:
            string_model_rel = detect_strings_in_neck(frame, locked_model)
            if len(string_model_rel) > 0:
                print(f"Strings detected: {len(string_model_rel)}")
            else:
                print("Failed to detect strings. Check lighting.")

        if is_tracking and tracking_pts is not None:
            # --- שלב המעקב (Tracking) ---
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(last_gray, gray, tracking_pts, None)
            good_new = new_pts[status.flatten() == 1]
            good_old = initial_pts[status.flatten() == 1]

            if len(good_new) >= 4:
                matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)

                if matrix is not None:
                    tracking_pts = cv2.transform(initial_pts, matrix)
                    last_gray = gray.copy()

                    # חישוב 4 פינות הצוואר
                    corners = np.array([
                        [locked_model['x_min'], locked_model['y_t']],
                        [locked_model['x_max'], locked_model['y_t']],
                        [locked_model['x_min'], locked_model['y_b']],
                        [locked_model['x_max'], locked_model['y_b']]
                    ], dtype=np.float32).reshape(-1, 1, 2)

                    tracked_corners = cv2.transform(corners, matrix).reshape(-1, 2)
                    tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

                    # ציור סריגים (ירוק)
                    for rel_x in fret_model_rel:
                        fx_t, fy_t = tl + rel_x * (tr - tl)
                        fx_b, fy_b = bl + rel_x * (br - bl)
                        cv2.line(display_frame, (int(fx_t), int(fy_t)), (int(fx_b), int(fy_b)), (0, 255, 0), 2)

                    # ציור מיתרים (צהוב)
                    for rel_y in string_model_rel:
                        sx_l, sy_l = tl + rel_y * (bl - tl)
                        sx_r, sy_r = tr + rel_y * (br - tr)
                        cv2.line(display_frame, (int(sx_l), int(sy_l)), (int(sx_r), int(sy_r)), (0, 255, 255), 1)

                    # גבולות כחולים
                    cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                    cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)
            else:
                cv2.putText(display_frame, "LOST - Press R", (10, 30), 1, 1, (0, 0, 255), 2)

        else:
            # --- שלב התצוגה המקדימה (Preview) ---
            raw_f = detect_frets_bottom(frame)
            y_t, y_b = detect_guitar_neck_bounds(frame)

            if y_t is not None:
                cv2.line(display_frame, (0, y_t), (width, y_t), (0, 0, 255), 1)
                cv2.line(display_frame, (0, y_b), (width, y_b), (0, 0, 255), 1)

            if key == ord('c') and y_t is not None and len(raw_f) > 2:
                stable_f = merge_vertical_lines(raw_f)
                x_min, x_max = stable_f[0][0], stable_f[-1][0]

                grid_x = np.linspace(x_min, x_max, 10)
                grid_y = np.linspace(y_t, y_b, 4)
                temp_pts = [[gx, gy] for gx in grid_x for gy in grid_y]

                initial_pts = np.array(temp_pts, dtype=np.float32).reshape(-1, 1, 2)
                tracking_pts = initial_pts.copy()

                locked_model = {'x_min': x_min, 'x_max': x_max, 'y_t': y_t, 'y_b': y_b}
                fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]

                last_gray = gray.copy()
                is_tracking = True
                print("Locked! Now press 's' for strings.")

        cv2.imshow("Guitar Tracker", display_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()