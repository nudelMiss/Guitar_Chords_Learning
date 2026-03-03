import cv2
import numpy as np


# --- 1. Classical Vision: Fret Detection (Sobel + Hough) ---
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


# --- 2. Classical Vision: Neck Bounds (Sobel Y) ---
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


# --- 3. Main Loop with Rigid Tracking ---
def main():
    cap = cv2.VideoCapture(0)

    # State Memory
    is_tracking = False
    tracking_pts = None  # Current tracked points
    initial_pts = None  # "Anchor" points from the moment of Calibration (C)
    fret_model_rel = []
    last_gray = None

    while True:
        ret, frame = cap.read()
        if not ret: break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()
        key = cv2.waitKey(1) & 0xFF

        # Manual Reset
        if key == ord('r'):
            is_tracking = False
            print("Resetting model...")

        if is_tracking and tracking_pts is not None:
            # LK Optical Flow to track movement
            new_pts, status, err = cv2.calcOpticalFlowPyrLK(last_gray, gray, tracking_pts, None)

            if status is not None and np.sum(status) >= 3:
                # RIGID TRANSFORMATION: Find how the neck moved as a single unit
                # This prevents the "bending" effect you saw
                matrix, inliers = cv2.estimateAffinePartial2D(initial_pts[status.flatten() == 1],
                                                              new_pts[status.flatten() == 1])

                if matrix is not None:
                    # Apply the transformation to the ORIGINAL 4 corners
                    tracked_corners = cv2.transform(initial_pts, matrix)
                    tracking_pts = tracked_corners
                    last_gray = gray.copy()

                    pts = tracked_corners.reshape(-1, 2)
                    tl, tr, bl, br = pts[0], pts[1], pts[2], pts[3]

                    # Draw frets based on the rigid model
                    for rel_x in fret_model_rel:
                        fx_t = tl[0] + rel_x * (tr[0] - tl[0])
                        fy_t = tl[1] + rel_x * (tr[1] - tl[1])
                        fx_b = bl[0] + rel_x * (br[0] - bl[0])
                        fy_b = bl[1] + rel_x * (br[1] - bl[1])

                        p1, p2 = (int(fx_t), int(fy_t)), (int(fx_b), int(fy_b))
                        # Only draw if the fret is visible in frame
                        if (0 <= p1[0] < width and 0 <= p1[1] < height and
                                0 <= p2[0] < width and 0 <= p2[1] < height):
                            cv2.line(display_frame, p1, p2, (0, 255, 0), 2)

                    # Draw boundaries
                    cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                    cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)
            else:
                cv2.putText(display_frame, "LOST - Reposition or press R", (10, 30), 1, 1, (0, 0, 255), 2)

        else:
            # LIVE PREVIEW MODE
            raw_f = detect_frets_bottom(frame)
            y_t, y_b = detect_guitar_neck_bounds(frame)

            if y_t is not None:
                cv2.line(display_frame, (0, y_t), (width, y_t), (0, 0, 255), 1)
                cv2.line(display_frame, (0, y_b), (width, y_b), (0, 0, 255), 1)

            if key == ord('c') and y_t is not None and len(raw_f) > 2:
                stable_f = merge_vertical_lines(raw_f)
                x_min, x_max = stable_f[0][0], stable_f[-1][0]

                # LOCK initial rigid model
                initial_pts = np.array([[x_min, y_t], [x_max, y_t], [x_min, y_b], [x_max, y_b]],
                                       dtype=np.float32).reshape(-1, 1, 2)
                tracking_pts = initial_pts.copy()
                fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]
                last_gray = gray.copy()
                is_tracking = True
                print("Locked Rigid Model!")

        cv2.imshow("Rigid Fret Tracker (Vision Only)", display_frame)
        if key == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()