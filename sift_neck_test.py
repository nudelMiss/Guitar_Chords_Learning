import cv2
import numpy as np


class SIFTNeckTracker:
    def __init__(self):
        self.roi_x0_rel = 0.300
        self.roi_x1_rel = 0.880
        self.roi_y0_rel = 0.560
        self.roi_y1_rel = 0.740

        self.is_locked = False

        self.locked_quad = None
        self.locked_gray = None
        self.locked_crop = None
        self.locked_kp = None
        self.locked_des = None

        self.last_good_matrix = None

        # Search area enlargement around previous estimate
        self.search_scale_x_good = 1.08
        self.search_scale_y_good = 1.18

        # Larger search for reacquisition after fast motion
        self.search_scale_x_lost = 1.70
        self.search_scale_y_lost = 2.00

        self.tracking_ok = False
        self.failed_frames = 0

        # SIFT + matcher
        self.sift = cv2.SIFT_create(nfeatures=900)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        self.locked_top_rel = 0.28
        self.locked_bottom_rel = 0.72
        self.fret_model_rel = []
        self.string_model_rel = [i / 5.0 for i in range(6)]

    def _build_refined_quad_from_roi(self, roi_quad, top_rel, bottom_rel):
        tl, tr, bl, br = roi_quad

        top_left = tl + top_rel * (bl - tl)
        top_right = tr + top_rel * (br - tr)
        bottom_left = tl + bottom_rel * (bl - tl)
        bottom_right = tr + bottom_rel * (br - tr)

        return np.array(
            [top_left, top_right, bottom_left, bottom_right],
            dtype=np.float32
        )

    def _detect_neck_band_in_roi(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.28, 0.72

        h, w = roi_bgr.shape[:2]
        if h < 20 or w < 40:
            return 0.28, 0.72

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        sobely = cv2.convertScaleAbs(sobely)

        x0 = int(0.18 * w)
        x1 = int(0.92 * w)
        band = sobely[:, x0:x1]

        row_score = np.mean(band, axis=1).astype(np.float32)
        row_score = cv2.GaussianBlur(row_score.reshape(-1, 1), (1, 31), 0).flatten()

        margin = max(4, int(0.08 * h))
        mid = h // 2

        top_search = row_score[margin:max(margin + 1, mid)]
        bot_search = row_score[mid:min(h - margin, h)]

        if len(top_search) == 0 or len(bot_search) == 0:
            return 0.28, 0.72

        top_idx = int(np.argmax(top_search)) + margin
        bot_idx = int(np.argmax(bot_search)) + mid

        min_gap = max(12, int(0.18 * h))
        if bot_idx - top_idx < min_gap:
            return 0.28, 0.72

        top_rel = float(np.clip(top_idx / float(h), 0.05, 0.65))
        bottom_rel = float(np.clip(bot_idx / float(h), 0.35, 0.95))

        if bottom_rel - top_rel < 0.15:
            return 0.28, 0.72

        return top_rel, bottom_rel

    def _detect_frets_in_refined_bbox(self, frame, refined_quad):
        x0, y0, x1, y1 = self._quad_to_bbox(refined_quad)
        h, w = frame.shape[:2]

        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return []

        ch, cw = crop.shape[:2]
        if cw < 40 or ch < 12:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)

        _, thresh = cv2.threshold(
            sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        lines = cv2.HoughLinesP(
            thresh,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=max(8, ch // 2),
            maxLineGap=8
        )

        xs = []
        if lines is not None:
            for l in lines:
                x1l, y1l, x2l, y2l = l[0]
                if abs(x2l - x1l) <= 6 and abs(y2l - y1l) >= max(8, ch // 2):
                    xs.append(int((x1l + x2l) / 2))

        if not xs:
            return []

        xs = sorted(xs)
        merged = [[xs[0]]]
        for x in xs[1:]:
            if abs(x - merged[-1][-1]) <= 12:
                merged[-1].append(x)
            else:
                merged.append([x])

        merged_x = [int(np.mean(g)) for g in merged]
        merged_x = [x for x in merged_x if 5 < x < cw - 5]

        if len(merged_x) < 2:
            return []

        return [x / float(cw) for x in merged_x]

    # ---------------------------------
    # Geometry helpers
    # ---------------------------------
    def _rel_roi_to_abs_quad(self, frame_shape):
        h, w = frame_shape[:2]
        x0 = int(w * self.roi_x0_rel)
        x1 = int(w * self.roi_x1_rel)
        y0 = int(h * self.roi_y0_rel)
        y1 = int(h * self.roi_y1_rel)

        return np.array(
            [
                [x0, y0],  # tl
                [x1, y0],  # tr
                [x0, y1],  # bl
                [x1, y1],  # br
            ],
            dtype=np.float32
        )

    def _quad_to_bbox(self, quad):
        xs = quad[:, 0]
        ys = quad[:, 1]
        return int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))

    def _clamp_quad_to_frame(self, quad, frame_shape):
        h, w = frame_shape[:2]
        out = quad.copy()
        out[:, 0] = np.clip(out[:, 0], 0, w - 1)
        out[:, 1] = np.clip(out[:, 1], 0, h - 1)
        return out

    def _crop_bbox_from_quad(self, frame, quad):
        x0, y0, x1, y1 = self._quad_to_bbox(quad)
        h, w = frame.shape[:2]

        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        return frame[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)

    def _scale_quad_about_center(self, quad, sx=1.0, sy=1.0):
        quad = quad.astype(np.float32)
        center = np.mean(quad, axis=0, keepdims=True)
        q = quad - center
        q[:, 0] *= sx
        q[:, 1] *= sy
        return q + center

    def _corners_are_reasonable(self, corners, frame_shape):
        if corners is None:
            return False

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

        if avg_w < 80 or avg_h < 16:
            return False
        if avg_w > w * 1.1 or avg_h > h * 0.6:
            return False

        return True

    def _draw_quad(self, img, quad, color=(0, 255, 255), thickness=2):
        q = quad.astype(int)
        cv2.line(img, tuple(q[0]), tuple(q[1]), color, thickness)
        cv2.line(img, tuple(q[1]), tuple(q[3]), color, thickness)
        cv2.line(img, tuple(q[3]), tuple(q[2]), color, thickness)
        cv2.line(img, tuple(q[2]), tuple(q[0]), color, thickness)

    # ---------------------------------
    # Lock
    # ---------------------------------
    def lock(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quad = self._rel_roi_to_abs_quad(frame.shape)
        quad = self._clamp_quad_to_frame(quad, frame.shape)

        crop, _ = self._crop_bbox_from_quad(gray, quad)
        kp, des = self.sift.detectAndCompute(crop, None)

        if des is None or kp is None or len(kp) < 12:
            print("Lock failed: not enough SIFT features in ROI.")
            self.is_locked = False
            return

        self.locked_quad = quad.copy()

        roi_bgr, _ = self._crop_bbox_from_quad(frame, quad)
        top_rel, bottom_rel = self._detect_neck_band_in_roi(roi_bgr)
        self.locked_top_rel = top_rel
        self.locked_bottom_rel = bottom_rel

        refined_quad = self._build_refined_quad_from_roi(quad, top_rel, bottom_rel)
        self.fret_model_rel = self._detect_frets_in_refined_bbox(frame, refined_quad)
        self.string_model_rel = [i / 5.0 for i in range(6)]

        self.locked_gray = gray.copy()
        self.locked_crop = crop.copy()
        self.locked_kp = kp
        self.locked_des = des
        self.last_good_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.is_locked = True
        self.tracking_ok = True
        self.failed_frames = 0

        print(f"Locked. SIFT keypoints in ROI: {len(kp)}")

    # ---------------------------------
    # Search area
    # ---------------------------------
    def _get_search_quad(self, frame_shape):
        if not self.is_locked or self.locked_quad is None:
            return self._rel_roi_to_abs_quad(frame_shape)

        current_quad = cv2.transform(
            self.locked_quad.reshape(-1, 1, 2),
            self.last_good_matrix
        ).reshape(-1, 2)

        if self.tracking_ok:
            sx = self.search_scale_x_good
            sy = self.search_scale_y_good
        else:
            sx = self.search_scale_x_lost
            sy = self.search_scale_y_lost

        current_quad = self._scale_quad_about_center(current_quad, sx=sx, sy=sy)
        current_quad = self._clamp_quad_to_frame(current_quad, frame_shape)
        return current_quad

    # ---------------------------------
    # Update
    # ---------------------------------
    def update(self, frame):
        display = frame.copy()

        preview_quad = self._get_search_quad(frame.shape)
        self._draw_quad(display, preview_quad, color=(0, 255, 255), thickness=1)

        if not self.is_locked:
            cv2.putText(
                display,
                "Preview: press [c] to lock SIFT ROI",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            return display

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        search_crop, (sx0, sy0, sx1, sy1) = self._crop_bbox_from_quad(gray, preview_quad)
        curr_kp, curr_des = self.sift.detectAndCompute(search_crop, None)

        if curr_des is None or curr_kp is None or len(curr_kp) < 12:
            self.tracking_ok = False
            self.failed_frames += 1

            cv2.putText(
                display,
                "Current frame: not enough SIFT features",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if self.last_good_matrix is not None:
                tracked_quad = cv2.transform(
                    self.locked_quad.reshape(-1, 1, 2),
                    self.last_good_matrix
                ).reshape(-1, 2)
                tracked_quad = self._clamp_quad_to_frame(tracked_quad, frame.shape)

                if self._corners_are_reasonable(tracked_quad, frame.shape):
                    self._draw_quad(display, tracked_quad, color=(255, 0, 0), thickness=2)

                    tracked_refined_quad = self._build_refined_quad_from_roi(
                        tracked_quad,
                        self.locked_top_rel,
                        self.locked_bottom_rel
                    )

                    rq = tracked_refined_quad.astype(int)
                    cv2.line(display, tuple(rq[0]), tuple(rq[1]), (255, 0, 0), 2)
                    cv2.line(display, tuple(rq[1]), tuple(rq[3]), (255, 0, 0), 1)
                    cv2.line(display, tuple(rq[3]), tuple(rq[2]), (255, 0, 0), 2)
                    cv2.line(display, tuple(rq[2]), tuple(rq[0]), (255, 0, 0), 1)

                    tl, tr, bl, br = tracked_refined_quad

                    for rel_x in self.fret_model_rel:
                        p1 = (tl + rel_x * (tr - tl)).astype(int)
                        p2 = (bl + rel_x * (br - bl)).astype(int)
                        cv2.line(display, tuple(p1), tuple(p2), (0, 255, 0), 2)

                    for rel_y in self.string_model_rel:
                        p1 = (tl + rel_y * (bl - tl)).astype(int)
                        p2 = (tr + rel_y * (br - tr)).astype(int)
                        cv2.line(display, tuple(p1), tuple(p2), (0, 255, 255), 1)

            return display

        matches_knn = self.matcher.knnMatch(self.locked_des, curr_des, k=2)

        good_matches = []
        for pair in matches_knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.80 * n.distance:
                good_matches.append(m)

        cv2.putText(
            display,
            f"SIFT matches: {len(good_matches)}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if len(good_matches) < 10:
            self.tracking_ok = False
            self.failed_frames += 1

            cv2.putText(
                display,
                "Too few good SIFT matches",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            if self.last_good_matrix is not None:
                tracked_quad = cv2.transform(
                    self.locked_quad.reshape(-1, 1, 2),
                    self.last_good_matrix
                ).reshape(-1, 2)
                tracked_quad = self._clamp_quad_to_frame(tracked_quad, frame.shape)

                if self._corners_are_reasonable(tracked_quad, frame.shape):
                    self._draw_quad(display, tracked_quad, color=(255, 0, 0), thickness=2)

                    tracked_refined_quad = self._build_refined_quad_from_roi(
                        tracked_quad,
                        self.locked_top_rel,
                        self.locked_bottom_rel
                    )

                    rq = tracked_refined_quad.astype(int)
                    cv2.line(display, tuple(rq[0]), tuple(rq[1]), (255, 0, 0), 2)
                    cv2.line(display, tuple(rq[1]), tuple(rq[3]), (255, 0, 0), 1)
                    cv2.line(display, tuple(rq[3]), tuple(rq[2]), (255, 0, 0), 2)
                    cv2.line(display, tuple(rq[2]), tuple(rq[0]), (255, 0, 0), 1)

                    tl, tr, bl, br = tracked_refined_quad

                    for rel_x in self.fret_model_rel:
                        p1 = (tl + rel_x * (tr - tl)).astype(int)
                        p2 = (bl + rel_x * (br - bl)).astype(int)
                        cv2.line(display, tuple(p1), tuple(p2), (0, 255, 0), 2)

                    for rel_y in self.string_model_rel:
                        p1 = (tl + rel_y * (bl - tl)).astype(int)
                        p2 = (tr + rel_y * (br - tr)).astype(int)
                        cv2.line(display, tuple(p1), tuple(p2), (0, 255, 255), 1)

            return display

        src_pts = []
        dst_pts = []

        for m in good_matches:
            lp = self.locked_kp[m.queryIdx].pt
            cp = curr_kp[m.trainIdx].pt

            lx = lp[0] + self.locked_quad[0][0]
            ly = lp[1] + self.locked_quad[0][1]

            cx = cp[0] + sx0
            cy = cp[1] + sy0

            src_pts.append([lx, ly])
            dst_pts.append([cx, cy])

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        M, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0
        )

        if M is None or inliers is None:
            self.tracking_ok = False
            self.failed_frames += 1

            cv2.putText(
                display,
                "Affine estimation failed",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
            return display

        inlier_ratio = float(np.sum(inliers)) / float(len(inliers))
        cv2.putText(
            display,
            f"Inlier ratio: {inlier_ratio:.2f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if inlier_ratio >= 0.30:
            self.last_good_matrix = M.astype(np.float32)
            self.tracking_ok = True
            self.failed_frames = 0
        else:
            self.tracking_ok = False
            self.failed_frames += 1

        tracked_quad = cv2.transform(
            self.locked_quad.reshape(-1, 1, 2),
            self.last_good_matrix
        ).reshape(-1, 2)

        tracked_quad = self._clamp_quad_to_frame(tracked_quad, frame.shape)

        if self._corners_are_reasonable(tracked_quad, frame.shape):
            self._draw_quad(display, tracked_quad, color=(255, 0, 0), thickness=2)

            tracked_refined_quad = self._build_refined_quad_from_roi(
                tracked_quad,
                self.locked_top_rel,
                self.locked_bottom_rel
            )

            rq = tracked_refined_quad.astype(int)
            cv2.line(display, tuple(rq[0]), tuple(rq[1]), (255, 0, 0), 2)
            cv2.line(display, tuple(rq[1]), tuple(rq[3]), (255, 0, 0), 1)
            cv2.line(display, tuple(rq[3]), tuple(rq[2]), (255, 0, 0), 2)
            cv2.line(display, tuple(rq[2]), tuple(rq[0]), (255, 0, 0), 1)

            tl, tr, bl, br = tracked_refined_quad

            for rel_x in self.fret_model_rel:
                p1 = (tl + rel_x * (tr - tl)).astype(int)
                p2 = (bl + rel_x * (br - bl)).astype(int)
                cv2.line(display, tuple(p1), tuple(p2), (0, 255, 0), 2)

            for rel_y in self.string_model_rel:
                p1 = (tl + rel_y * (bl - tl)).astype(int)
                p2 = (tr + rel_y * (br - tr)).astype(int)
                cv2.line(display, tuple(p1), tuple(p2), (0, 255, 255), 1)
        else:
            cv2.putText(
                display,
                "Tracked quad invalid",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return display


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = SIFTNeckTracker()

    print("Commands:")
    print("c = lock ROI with SIFT")
    print("r = reset")
    print("q = quit")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read frame.")
            break

        vis = tracker.update(frame)

        cv2.imshow("SIFT Neck Test", vis)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q")):
            break
        elif key in (ord("c"), ord("C")):
            tracker.lock(frame)
        elif key in (ord("r"), ord("R")):
            tracker = SIFTNeckTracker()
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()