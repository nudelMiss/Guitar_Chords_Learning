import cv2
import numpy as np


def detect_strings_in_neck(frame, locked_model):
    """
    Geometric string model only.
    Returns 6 evenly-spaced relative string positions between y_t and y_b.
    """
    y_t = locked_model.get("y_t", 0)
    y_b = locked_model.get("y_b", 0)

    if y_b <= y_t:
        return [i / 5.0 for i in range(6)]

    return [i / 5.0 for i in range(6)]


class GuitarNeckTracker:
    def __init__(self):
        # ROI tuned for website camera view
        self.roi_x0_rel = 0.300
        self.roi_x1_rel = 0.880
        self.roi_y0_rel = 0.560
        self.roi_y1_rel = 0.740

        # Tracking state
        self.is_tracking = False

        self.locked_quad = None          # original ROI at lock
        self.preview_quad = None         # current preview / search quad
        self.refined_quad = None         # current fretboard quad

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

        # Neck / fretboard model
        self.locked_top_rel = 0.28
        self.locked_bottom_rel = 0.72
        self.fret_model_rel = []
        self.string_model_rel = []
        self.is_strings_calibrated = False

        # Optional debug info
        self.last_match_count = 0
        self.last_inlier_ratio = 0.0

    # =========================================================
    # Public API
    # =========================================================
    def reset(self):
        self.__init__()

    def clear_strings(self):
        self.string_model_rel = []
        self.is_strings_calibrated = False

    def lock(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quad = self._rel_roi_to_abs_quad(frame.shape)
        quad = self._clamp_quad_to_frame(quad, frame.shape)

        crop, _ = self._crop_bbox_from_quad(gray, quad)
        roi_bgr, _ = self._crop_bbox_from_quad(frame, quad)

        top_rel, bottom_rel = self._detect_neck_band_in_roi(roi_bgr)
        self.locked_top_rel = top_rel
        self.locked_bottom_rel = bottom_rel

        neck_mask = self._make_neck_band_mask(crop.shape, top_rel, bottom_rel)
        kp, des = self.sift.detectAndCompute(crop, neck_mask)

        if des is None or kp is None or len(kp) < 12:
            self.is_tracking = False
            print("Lock failed: not enough SIFT features in neck band.")
            return False

        self.locked_quad = quad.copy()
        self.preview_quad = quad.copy()

        refined_quad = self._build_refined_quad_from_roi(quad, top_rel, bottom_rel)
        self.refined_quad = refined_quad.copy()

        self.fret_model_rel = self._detect_frets_in_refined_bbox(frame, refined_quad)

        self.locked_gray = gray.copy()
        self.locked_crop = crop.copy()
        self.locked_kp = kp
        self.locked_des = des
        self.last_good_matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        self.is_tracking = True
        self.tracking_ok = True
        self.failed_frames = 0
        self.is_strings_calibrated = False
        self.string_model_rel = []

        print(f"Neck locked. SIFT keypoints in ROI: {len(kp)}")
        if len(self.fret_model_rel) >= 2:
            print(f"Detected {len(self.fret_model_rel)} fret lines.")
        else:
            print("Fret detection was weak. Try again while holding still.")

        return True

    def update(self, frame):
        self.preview_quad = self._get_search_quad(frame.shape)

        if not self.is_tracking or self.locked_quad is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        search_crop, (sx0, sy0, sx1, sy1) = self._crop_bbox_from_quad(gray, self.preview_quad)

        search_mask = self._make_neck_band_mask(
            search_crop.shape,
            self.locked_top_rel,
            self.locked_bottom_rel
        )

        curr_kp, curr_des = self.sift.detectAndCompute(search_crop, search_mask)

        if curr_des is None or curr_kp is None or len(curr_kp) < 12:
            self.tracking_ok = False
            self.failed_frames += 1
            return False

        matches_knn = self.matcher.knnMatch(self.locked_des, curr_des, k=2)

        good_matches = []
        for pair in matches_knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.80 * n.distance:
                good_matches.append(m)

        self.last_match_count = len(good_matches)

        if len(good_matches) < 10:
            self.tracking_ok = False
            self.failed_frames += 1
            return False

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
            return False

        inlier_ratio = float(np.sum(inliers)) / float(len(inliers))
        self.last_inlier_ratio = inlier_ratio

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
            self.refined_quad = self._build_refined_quad_from_roi(
                tracked_quad,
                self.locked_top_rel,
                self.locked_bottom_rel
            )
            return True

        return False

    def calibrate_strings(self, frame):
        locked_model = self.get_locked_model()
        detected = detect_strings_in_neck(frame, locked_model)

        if len(detected) == 6:
            self.string_model_rel = detected
            self.is_strings_calibrated = True
            print("Strings locked (6 detected / modeled).")
            return True

        self.string_model_rel = []
        self.is_strings_calibrated = False
        print("Strings failed.")
        return False

    def get_locked_model(self):
        if self.refined_quad is None:
            return {"x_min": 0, "x_max": 0, "y_t": 0, "y_b": 0}

        xs = self.refined_quad[:, 0]
        ys = self.refined_quad[:, 1]
        return {
            "x_min": int(np.min(xs)),
            "x_max": int(np.max(xs)),
            "y_t": int(np.min(ys)),
            "y_b": int(np.max(ys)),
        }

    def get_dot_coordinates(self, fret_num, string_num, mirror_frets=True):
        if self.refined_quad is None:
            return -1, -1
        if not self.fret_model_rel or not self.string_model_rel:
            return -1, -1
        if string_num < 1 or string_num > 6:
            return -1, -1
        if fret_num < 1 or fret_num > len(self.fret_model_rel):
            return -1, -1

        tl, tr, bl, br = self.refined_quad[0], self.refined_quad[1], self.refined_quad[2], self.refined_quad[3]

        rel_y = float(self.string_model_rel[6 - string_num])

        num_frets = len(self.fret_model_rel)
        use_fret = (num_frets - fret_num + 1) if mirror_frets else fret_num

        if use_fret == 1:
            rel_x = float(self.fret_model_rel[0]) / 2.0
        else:
            rel_x = (
                float(self.fret_model_rel[use_fret - 1]) +
                float(self.fret_model_rel[use_fret - 2])
            ) / 2.0

        top_point = tl + rel_x * (tr - tl)
        bot_point = bl + rel_x * (br - bl)
        pt = top_point + rel_y * (bot_point - top_point)

        return int(pt[0]), int(pt[1])

    def draw_debug(self, frame):
        out = frame.copy()

        # Preview mode only: show yellow ROI before lock/tracking
        if not self.is_tracking or self.refined_quad is None:
            preview_quad = self._rel_roi_to_abs_quad(frame.shape)
            preview_quad = self._clamp_quad_to_frame(preview_quad, frame.shape)
            pt = preview_quad.astype(int)

            cv2.line(out, tuple(pt[0]), tuple(pt[1]), (0, 255, 255), 2)
            cv2.line(out, tuple(pt[1]), tuple(pt[3]), (0, 255, 255), 2)
            cv2.line(out, tuple(pt[3]), tuple(pt[2]), (0, 255, 255), 2)
            cv2.line(out, tuple(pt[2]), tuple(pt[0]), (0, 255, 255), 2)
            return out

        # Tracking mode: do NOT draw yellow preview/search box anymore

        rc = self.refined_quad.astype(int)
        cv2.line(out, tuple(rc[0]), tuple(rc[1]), (255, 0, 0), 2)
        cv2.line(out, tuple(rc[1]), tuple(rc[3]), (255, 0, 0), 1)
        cv2.line(out, tuple(rc[3]), tuple(rc[2]), (255, 0, 0), 2)
        cv2.line(out, tuple(rc[2]), tuple(rc[0]), (255, 0, 0), 1)

        tl, tr, bl, br = self.refined_quad

        for rel_x in self.fret_model_rel:
            p1 = (tl + rel_x * (tr - tl)).astype(int)
            p2 = (bl + rel_x * (br - bl)).astype(int)
            cv2.line(out, tuple(p1), tuple(p2), (0, 255, 0), 2)

        for rel_y in self.string_model_rel:
            p1 = (tl + rel_y * (bl - tl)).astype(int)
            p2 = (tr + rel_y * (br - tr)).astype(int)
            cv2.line(out, tuple(p1), tuple(p2), (0, 255, 255), 1)

        return out

    # =========================================================
    # Internal helpers
    # =========================================================
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

    def _merge_x_positions(self, xs, x_threshold=14):
        if not xs:
            return []

        xs = sorted(xs)
        groups = [[xs[0]]]

        for x in xs[1:]:
            if abs(x - groups[-1][-1]) <= x_threshold:
                groups[-1].append(x)
            else:
                groups.append([x])

        return [int(np.mean(g)) for g in groups]

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

        x_candidates = []
        if lines is not None:
            for l in lines:
                lx1, ly1, lx2, ly2 = l[0]
                if abs(lx2 - lx1) <= 6 and abs(ly2 - ly1) >= max(8, ch // 2):
                    x_candidates.append(int((lx1 + lx2) / 2))

        merged_x = self._merge_x_positions(x_candidates, x_threshold=12)
        merged_x = [x for x in merged_x if 5 < x < cw - 5]

        if len(merged_x) < 2:
            return []

        return [x / float(cw) for x in merged_x]

    def _get_search_quad(self, frame_shape):
        if not self.is_tracking or self.locked_quad is None:
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

    
    def _make_neck_band_mask(self, shape, top_rel, bottom_rel, x_margin_rel=0.03):
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        y0 = int(np.clip(top_rel * h, 0, h - 1))
        y1 = int(np.clip(bottom_rel * h, 1, h))
        x0 = int(np.clip(x_margin_rel * w, 0, w - 1))
        x1 = int(np.clip((1.0 - x_margin_rel) * w, 1, w))

        if y1 <= y0:
            return mask

        mask[y0:y1, x0:x1] = 255
        return mask