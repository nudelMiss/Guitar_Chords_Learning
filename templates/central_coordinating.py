import cv2
import numpy as np

import Chords
import string_and_frets as sf


def _current_locked_model_from_tracked_corners(tracked_corners):
    """
    tracked_corners: numpy array shape (4,2) points [tl,tr,bl,br]
    Returns dict: x_min, x_max, y_t, y_b
    """
    xs = tracked_corners[:, 0]
    ys = tracked_corners[:, 1]
    return {
        "x_min": int(np.min(xs)),
        "x_max": int(np.max(xs)),
        "y_t": int(np.min(ys)),
        "y_b": int(np.max(ys)),
    }


def get_dot_coordinates(fret_num, string_num, tracked_corners, fret_model_rel, string_model_rel):
    """
    Convert (fret_num, string_num) -> pixel (x,y) on CURRENT tracked neck.
    tracked_corners: (4,2) points tl,tr,bl,br
    """
    if not fret_model_rel or not string_model_rel:
        return -1, -1
    if string_num < 1 or string_num > 6:
        return -1, -1
    if fret_num < 1 or fret_num > len(fret_model_rel):
        return -1, -1

    tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

    # Y from strings
    rel_y = float(string_model_rel[string_num - 1])

    # X in the middle of the fret cell
    if fret_num == 1:
        rel_x = float(fret_model_rel[0]) / 2.0
    else:
        rel_x = (float(fret_model_rel[fret_num - 1]) + float(fret_model_rel[fret_num - 2])) / 2.0

    top_point = tl + rel_x * (tr - tl)
    bot_point = bl + rel_x * (br - bl)
    pt = top_point + rel_y * (bot_point - top_point)

    return int(pt[0]), int(pt[1])


def main():
    cap = cv2.VideoCapture(0)

    # State
    is_tracking = False
    is_strings_calibrated = False

    tracking_pts = None
    initial_pts = None
    last_gray = None

    locked_model = {}
    fret_model_rel = []
    string_model_rel = []

    # Chord selection (non-blocking)
    available_chords = Chords.list_available_chords()
    available_chords.sort()
    chord_idx = 0
    current_chord_name = None
    current_chord_data = None

    print("--- Central System (Python 3.8 safe) ---")
    print("Commands:")
    print("  [c] Lock Neck")
    print("  [s] Calibrate Strings (requires tracking)")
    print("  [a] Next chord, [z] Prev chord, [x] Clear chord")
    print("  [r] Reset All, [q] Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):
            is_tracking = False
            is_strings_calibrated = False
            tracking_pts = None
            initial_pts = None
            last_gray = None
            locked_model = {}
            fret_model_rel = []
            string_model_rel = []
            current_chord_name = None
            current_chord_data = None
            print("Reset All.")
            cv2.imshow("Guitar Learning - Central System", display_frame)
            continue

        # Chord controls (no input() blocking)
        if key == ord('a'):
            chord_idx = (chord_idx + 1) % len(available_chords)
            current_chord_name = available_chords[chord_idx]
            current_chord_data = Chords.get_chord_data(current_chord_name)
            print("Chord selected:", current_chord_name)

        if key == ord('z'):
            chord_idx = (chord_idx - 1) % len(available_chords)
            current_chord_name = available_chords[chord_idx]
            current_chord_data = Chords.get_chord_data(current_chord_name)
            print("Chord selected:", current_chord_name)

        if key == ord('x'):
            current_chord_name = None
            current_chord_data = None
            print("Chord cleared.")

        # =========================
        # TRACKING MODE
        # =========================
        if is_tracking and tracking_pts is not None and initial_pts is not None and last_gray is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(last_gray, gray, tracking_pts, None)

            if new_pts is not None and status is not None:
                good_new = new_pts[status.flatten() == 1]
                good_old = initial_pts[status.flatten() == 1]
            else:
                good_new = np.array([])
                good_old = np.array([])

            if len(good_new) >= 4:
                matrix, _ = cv2.estimateAffinePartial2D(
                    good_old, good_new,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3
                )

                if matrix is not None:
                    tracking_pts = cv2.transform(initial_pts, matrix)
                    last_gray = gray.copy()

                    # transform calibration-time neck corners
                    corners = np.array(
                        [
                            [locked_model["x_min"], locked_model["y_t"]],
                            [locked_model["x_max"], locked_model["y_t"]],
                            [locked_model["x_min"], locked_model["y_b"]],
                            [locked_model["x_max"], locked_model["y_b"]],
                        ],
                        dtype=np.float32
                    ).reshape(-1, 1, 2)

                    tracked_corners = cv2.transform(corners, matrix).reshape(-1, 2)
                    tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

                    # IMPORTANT FIX: strings calibration uses CURRENT tracked bounds
                    current_locked_model = _current_locked_model_from_tracked_corners(tracked_corners)

                    if key == ord('s'):
                        detected = sf.detect_strings_in_neck(frame, current_locked_model)
                        if len(detected) == 6:
                            string_model_rel = detected
                            is_strings_calibrated = True
                            print("Strings locked (6 detected).")
                        else:
                            string_model_rel = []
                            is_strings_calibrated = False
                            print("Strings failed: detected", len(detected), "need 6. Try again [s].")

                    # Draw frets (green)
                    for rel_x in fret_model_rel:
                        p1 = (tl + rel_x * (tr - tl)).astype(int)
                        p2 = (bl + rel_x * (br - bl)).astype(int)
                        cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 0), 2)

                    # Draw strings (yellow)
                    for rel_y in string_model_rel:
                        p1 = (tl + rel_y * (bl - tl)).astype(int)
                        p2 = (tr + rel_y * (br - tr)).astype(int)
                        cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 255), 1)

                    # Draw neck boundaries (blue)
                    cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                    cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)

                    # Draw chord dots (only if strings calibrated)
                    if current_chord_data is not None and is_strings_calibrated:
                        for fret_num, string_num in current_chord_data["fingers"]:
                            x, y = get_dot_coordinates(
                                fret_num, string_num,
                                tracked_corners,
                                fret_model_rel,
                                string_model_rel
                            )
                            if x >= 0 and y >= 0:
                                cv2.circle(display_frame, (x, y), 12, (255, 120, 0), -1)
                                cv2.circle(display_frame, (x, y), 13, (255, 255, 255), 1)

                    # HUD
                    if current_chord_name:
                        cv2.putText(display_frame, "Chord: {}".format(current_chord_name),
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    if not is_strings_calibrated:
                        cv2.putText(display_frame, "Press [s] to calibrate strings",
                                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # =========================
        # PREVIEW / CALIBRATION MODE
        # =========================
        else:
            raw_frets = sf.detect_frets_bottom(frame)
            y_t, y_b = sf.detect_guitar_neck_bounds(frame)

            if y_t is not None and y_b is not None:
                cv2.line(display_frame, (0, y_t), (width, y_t), (0, 0, 255), 2)
                cv2.line(display_frame, (0, y_b), (width, y_b), (0, 0, 255), 2)

            if key == ord('c') and y_t is not None and len(raw_frets) > 2:
                stable_f = sf.merge_vertical_lines(raw_frets)
                if len(stable_f) >= 2:
                    x_min, x_max = stable_f[0][0], stable_f[-1][0]

                    grid_x = np.linspace(x_min, x_max, 10)
                    grid_y = np.linspace(y_t, y_b, 4)
                    temp_pts = [[gx, gy] for gx in grid_x for gy in grid_y]

                    initial_pts = np.array(temp_pts, dtype=np.float32).reshape(-1, 1, 2)
                    tracking_pts = initial_pts.copy()

                    locked_model = {"x_min": x_min, "x_max": x_max, "y_t": y_t, "y_b": y_b}
                    fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]

                    last_gray = gray.copy()
                    is_tracking = True

                    # reset strings after new lock
                    is_strings_calibrated = False
                    string_model_rel = []

                    print("Neck locked. Now press [s] to calibrate strings.")

            cv2.putText(display_frame, "Preview: press [c] to lock neck",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Guitar Learning - Central System", display_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()