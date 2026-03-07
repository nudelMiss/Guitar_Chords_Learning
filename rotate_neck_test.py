import cv2
import numpy as np
import math


# ----------------------------
# Geometry helpers
# ----------------------------
def angle_from_line(x1, y1, x2, y2):
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180
    return angle


def circular_mean_deg(angles_deg, weights=None):
    if not angles_deg:
        return None

    if weights is None:
        weights = [1.0] * len(angles_deg)

    doubled = [math.radians(2.0 * a) for a in angles_deg]
    c = sum(w * math.cos(a) for a, w in zip(doubled, weights))
    s = sum(w * math.sin(a) for a, w in zip(doubled, weights))

    if abs(c) < 1e-8 and abs(s) < 1e-8:
        return None

    mean2 = math.atan2(s, c)
    mean = math.degrees(mean2) / 2.0

    while mean >= 90:
        mean -= 180
    while mean < -90:
        mean += 180

    return mean


def rotate_image_keep_bounds(image, angle_deg):
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos_val = abs(M[0, 0])
    sin_val = abs(M[0, 1])

    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))

    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return rotated, M


def transform_point(M, x, y):
    pt = np.array([x, y, 1.0], dtype=np.float32)
    out = M @ pt
    return int(out[0]), int(out[1])


def resize_to_width(img, width):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (width, new_h))


def stack_vertical(images, width=1000):
    resized = [resize_to_width(im, width) for im in images]
    return cv2.vconcat(resized)


# ----------------------------
# ROI helpers
# ----------------------------
def rel_roi_to_abs(frame_shape, x0_rel, x1_rel, y0_rel, y1_rel):
    h, w = frame_shape[:2]
    x0 = int(w * x0_rel)
    x1 = int(w * x1_rel)
    y0 = int(h * y0_rel)
    y1 = int(h * y1_rel)
    return x0, y0, x1, y1


def clamp_quad_to_frame(corners, frame_shape):
    h, w = frame_shape[:2]
    out = corners.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


def quad_to_bbox(corners):
    xs = corners[:, 0]
    ys = corners[:, 1]
    return int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))


def draw_quad(img, corners, color=(0, 255, 255), thickness=2):
    out = img.copy()
    c = corners.astype(int)
    cv2.line(out, tuple(c[0]), tuple(c[1]), color, thickness)
    cv2.line(out, tuple(c[1]), tuple(c[3]), color, thickness)
    cv2.line(out, tuple(c[3]), tuple(c[2]), color, thickness)
    cv2.line(out, tuple(c[2]), tuple(c[0]), color, thickness)
    return out


def crop_bbox_from_quad(img, corners):
    x0, y0, x1, y1 = quad_to_bbox(corners)
    h, w = img.shape[:2]
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


# ----------------------------
# Line / angle detection inside ROI
# ----------------------------
def detect_dominant_angle_in_roi(
    roi_bgr,
    canny1=60,
    canny2=150,
    hough_threshold=50,
    min_line_length=80,
    max_line_gap=20,
    angle_limit_deg=25,
):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny1, canny2)

    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    selected_lines = []
    selected_angles = []
    selected_weights = []

    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            length = math.hypot(x2 - x1, y2 - y1)
            angle = angle_from_line(x1, y1, x2, y2)

            if abs(angle) <= angle_limit_deg:
                selected_lines.append((x1, y1, x2, y2))
                selected_angles.append(angle)
                selected_weights.append(length)

    mean_angle = circular_mean_deg(selected_angles, selected_weights)
    return mean_angle, edges, selected_lines


# ----------------------------
# Tracking helpers
# ----------------------------
def make_initial_quad_from_rel_roi(frame_shape, x0_rel, x1_rel, y0_rel, y1_rel):
    x0, y0, x1, y1 = rel_roi_to_abs(frame_shape, x0_rel, x1_rel, y0_rel, y1_rel)
    return np.array(
        [
            [x0, y0],  # tl
            [x1, y0],  # tr
            [x0, y1],  # bl
            [x1, y1],  # br
        ],
        dtype=np.float32,
    )


def sample_grid_points_in_quad(corners, nx=10, ny=4):
    tl, tr, bl, br = corners
    pts = []
    for j in range(ny):
        v = j / max(1, ny - 1)
        left = tl + v * (bl - tl)
        right = tr + v * (br - tr)
        for i in range(nx):
            u = i / max(1, nx - 1)
            p = left + u * (right - left)
            pts.append([p[0], p[1]])
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def estimate_tracked_quad(prev_gray, gray, prev_pts, prev_quad):
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    if new_pts is None or status is None:
        return None, None, False

    status = status.flatten()
    good_old = prev_pts[status == 1]
    good_new = new_pts[status == 1]

    if len(good_old) < 6:
        return None, None, False

    M, inliers = cv2.estimateAffinePartial2D(
        good_old,
        good_new,
        method=cv2.RANSAC,
        ransacReprojThreshold=3
    )

    if M is None:
        return None, None, False

    new_quad = cv2.transform(prev_quad.reshape(-1, 1, 2), M).reshape(-1, 2)
    updated_pts = cv2.transform(prev_pts, M)

    return new_quad, updated_pts, True


def corners_reasonable(corners, frame_shape):
    h, w = frame_shape[:2]
    xs = corners[:, 0]
    ys = corners[:, 1]

    if np.min(xs) < -80 or np.max(xs) > w + 80:
        return False
    if np.min(ys) < -80 or np.max(ys) > h + 80:
        return False

    tl, tr, bl, br = corners
    top_w = np.linalg.norm(tr - tl)
    bot_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)

    avg_w = 0.5 * (top_w + bot_w)
    avg_h = 0.5 * (left_h + right_h)

    if avg_w < 60 or avg_h < 30:
        return False
    if avg_w > w * 1.1 or avg_h > h * 0.7:
        return False

    return True


# ----------------------------
# Main
# ----------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Fixed ROI values you found
    roi_x0 = 0.300
    roi_x1 = 0.880
    roi_y0 = 0.660
    roi_y1 = 0.840

    # Detection params
    canny1 = 60
    canny2 = 150
    hough_threshold = 50
    min_line_length = 80
    max_line_gap = 20
    angle_limit_deg = 25

    # Tracking state
    is_tracking = False
    tracked_quad = None
    tracking_pts = None
    last_gray = None

    print("Commands:")
    print("q = quit")
    print("c = initialize / relock ROI from saved values")
    print("r = reset tracking")
    print("p = print current tracked ROI as relative bbox")
    print("The ROI will move with the guitar after lock.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug_frame = frame.copy()

        # Initialize tracker from saved ROI if not tracking yet
        if not is_tracking:
            tracked_quad = make_initial_quad_from_rel_roi(frame.shape, roi_x0, roi_x1, roi_y0, roi_y1)
            tracking_pts = sample_grid_points_in_quad(tracked_quad, nx=10, ny=4)
            last_gray = gray.copy()
            is_tracking = True

        # Track ROI quad
        else:
            new_quad, new_pts, ok_track = estimate_tracked_quad(last_gray, gray, tracking_pts, tracked_quad)

            if ok_track and new_quad is not None and corners_reasonable(new_quad, frame.shape):
                tracked_quad = new_quad
                tracking_pts = new_pts
                last_gray = gray.copy()
            else:
                cv2.putText(
                    debug_frame,
                    "Tracking weak - press c to relock",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )

        tracked_quad = clamp_quad_to_frame(tracked_quad, frame.shape)

        # Draw current tracked ROI quad
        debug_frame = draw_quad(debug_frame, tracked_quad, color=(0, 255, 255), thickness=2)

        # Crop current ROI from tracked quad
        roi, (bx0, by0, bx1, by1) = crop_bbox_from_quad(frame, tracked_quad)

        mean_angle, roi_edges, roi_lines = detect_dominant_angle_in_roi(
            roi,
            canny1=canny1,
            canny2=canny2,
            hough_threshold=hough_threshold,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            angle_limit_deg=angle_limit_deg,
        )

        # Draw selected lines on full frame
        for lx1, ly1, lx2, ly2 in roi_lines:
            cv2.line(
                debug_frame,
                (bx0 + lx1, by0 + ly1),
                (bx0 + lx2, by0 + ly2),
                (0, 255, 0),
                2
            )

        if mean_angle is not None:
            cv2.putText(
                debug_frame,
                f"Tracked ROI angle: {mean_angle:.2f} deg",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )

            rotated_full, M = rotate_image_keep_bounds(frame, -mean_angle)

            # transform tracked quad into rotated frame
            p1 = transform_point(M, tracked_quad[0][0], tracked_quad[0][1])
            p2 = transform_point(M, tracked_quad[1][0], tracked_quad[1][1])
            p3 = transform_point(M, tracked_quad[2][0], tracked_quad[2][1])
            p4 = transform_point(M, tracked_quad[3][0], tracked_quad[3][1])

            rotated_full_dbg = rotated_full.copy()
            cv2.line(rotated_full_dbg, p1, p2, (255, 255, 0), 2)
            cv2.line(rotated_full_dbg, p2, p4, (255, 255, 0), 2)
            cv2.line(rotated_full_dbg, p4, p3, (255, 255, 0), 2)
            cv2.line(rotated_full_dbg, p3, p1, (255, 255, 0), 2)

            rx = [p1[0], p2[0], p3[0], p4[0]]
            ry = [p1[1], p2[1], p3[1], p4[1]]
            rx0 = max(0, min(rx))
            rx1 = min(rotated_full.shape[1], max(rx))
            ry0 = max(0, min(ry))
            ry1 = min(rotated_full.shape[0], max(ry))

            if rx1 > rx0 and ry1 > ry0:
                rotated_roi = rotated_full[ry0:ry1, rx0:rx1].copy()
            else:
                rotated_roi = np.zeros((200, 400, 3), dtype=np.uint8)

        else:
            rotated_full_dbg = np.zeros_like(frame)
            rotated_roi = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(
                debug_frame,
                "No dominant angle inside tracked ROI",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 0, 255),
                2,
            )

        roi_dbg = roi.copy()
        for lx1, ly1, lx2, ly2 in roi_lines:
            cv2.line(roi_dbg, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

        roi_edges_bgr = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)

        cv2.putText(
            roi_dbg,
            "Tracked ROI",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            roi_edges_bgr,
            (
                f"C1={canny1} C2={canny2} H={hough_threshold} "
                f"minLen={min_line_length} gap={max_line_gap} angLim={angle_limit_deg}"
            ),
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            rotated_roi,
            "Rotated tracked ROI",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        panel = stack_vertical(
            [
                debug_frame,
                roi_dbg,
                roi_edges_bgr,
                rotated_full_dbg,
                rotated_roi,
            ],
            width=1000,
        )

        cv2.imshow("Rotate Neck Test", panel)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        elif key in (ord("c"), ord("C")):
            tracked_quad = make_initial_quad_from_rel_roi(frame.shape, roi_x0, roi_x1, roi_y0, roi_y1)
            tracking_pts = sample_grid_points_in_quad(tracked_quad, nx=10, ny=4)
            last_gray = gray.copy()
            is_tracking = True
            print("ROI relocked from saved values.")

        elif key in (ord("r"), ord("R")):
            is_tracking = False
            tracked_quad = None
            tracking_pts = None
            last_gray = None
            print("Tracking reset.")

        elif key in (ord("p"), ord("P")) and tracked_quad is not None:
            x0, y0, x1, y1 = quad_to_bbox(tracked_quad)
            h, w = frame.shape[:2]
            print("\n==== CURRENT TRACKED ROI (relative bbox) ====")
            print(f"roi_x0 = {x0 / w:.3f}")
            print(f"roi_x1 = {x1 / w:.3f}")
            print(f"roi_y0 = {y0 / h:.3f}")
            print(f"roi_y1 = {y1 / h:.3f}")
            print("===========================================\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()