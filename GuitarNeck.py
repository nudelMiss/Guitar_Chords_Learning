import cv2
import numpy as np



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

        # ORB + matcher (much faster than AKAZE)
        self.detector = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Temporal smoothing for refined_quad
        self.smooth_alpha = 0.6

        # Neck / fretboard model
        self.locked_top_rel = 0.28
        self.locked_bottom_rel = 0.72
        self.fret_model_rel = []
        self.string_model_rel = []
        self.is_strings_calibrated = False

        # Optional debug info
        self.last_match_count = 0
        self.last_inlier_ratio = 0.0

        # Optical flow tracking state
        self.prev_gray = None
        self.locked_track_pts = None
        self.current_track_pts = None

        # Periodic fret re-detection
        self.frames_since_lock = 0

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
        crop_enhanced = self.clahe.apply(crop)
        kp, des = self.detector.detectAndCompute(crop_enhanced, neck_mask)

        if des is None or kp is None or len(kp) < 12:
            self.is_tracking = False
            print("Lock failed: not enough ORB features in neck band.")
            return False

        self.locked_quad = quad.copy()
        self.preview_quad = quad.copy()

        refined_quad = self._build_refined_quad_from_roi(quad, top_rel, bottom_rel)
        self.refined_quad = refined_quad.copy()

        self.fret_model_rel = self._detect_frets_in_refined_bbox(frame, refined_quad)

        # Refine boundaries using detected fret endpoints (trapezoid fit)
        if len(self.fret_model_rel) >= 3:
            tighter_quad = self._refine_boundary_via_frets(
                frame, refined_quad, self.fret_model_rel
            )
            if tighter_quad is not None:
                self.refined_quad = tighter_quad.copy()
                refined_quad = tighter_quad
                # Re-detect frets with the tighter boundary
                new_frets = self._detect_frets_in_refined_bbox(frame, refined_quad)
                if len(new_frets) >= len(self.fret_model_rel):
                    self.fret_model_rel = new_frets

        # Expand the refined_quad slightly (6% vertically) for better string alignment.
        # The boundary refinement tends to be too tight — the detected edge is
        # slightly inside the actual fretboard. This expansion corrects the
        # visual alignment without affecting tracking precision.
        self.refined_quad = self._scale_quad_about_center(
            self.refined_quad, sx=1.0, sy=1.06
        )
        refined_quad = self.refined_quad.copy()

        self.locked_gray = gray.copy()
        self.locked_crop = crop.copy()
        self.locked_kp = kp
        self.locked_des = des
        self.last_good_matrix = np.eye(3, dtype=np.float32)

        # Initialize optical flow tracking points in the fretboard region
        self.prev_gray = gray.copy()
        flow_mask = self._make_refined_mask(gray.shape, refined_quad)
        flow_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=7,
            mask=flow_mask
        )
        if flow_pts is not None and len(flow_pts) >= 6:
            self.current_track_pts = flow_pts.astype(np.float32)
            self.locked_track_pts = flow_pts.copy().astype(np.float32)
        else:
            self.current_track_pts = None
            self.locked_track_pts = None

        self.is_tracking = True
        self.tracking_ok = True
        self.failed_frames = 0
        self.frames_since_lock = 0
        # Strings are NOT auto-calibrated; user must press 's' separately
        self.string_model_rel = []
        self.is_strings_calibrated = False

        print(f"Neck locked. ORB keypoints in ROI: {len(kp)}")
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

        success = False

        # Primary: sparse optical flow (fast, stable for small motion)
        if (self.prev_gray is not None
                and self.current_track_pts is not None
                and len(self.current_track_pts) >= 6):
            success = self._track_optical_flow(gray, frame.shape)

        # Fallback: ORB feature matching (recovery after lost tracking)
        if not success:
            success = self._track_orb_recovery(gray, frame.shape)
            if success:
                self._init_track_points(gray)

        self.prev_gray = gray.copy()

        if not success:
            self.tracking_ok = False
            self.failed_frames += 1
            return False

        return True

    def calibrate_strings(self, frame):
        if self.refined_quad is None:
            print("Strings failed: no fretboard locked.")
            return False

        detected = self._detect_strings_in_refined_bbox(frame, self.refined_quad)

        if len(detected) == 6:
            self.string_model_rel = detected
            self.is_strings_calibrated = True
            print(f"Strings locked (6 detected). Positions: "
                  f"{[f'{v:.3f}' for v in detected]}")
            return True

        # Fallback: evenly spaced model (always works)
        fallback = [i / 5.0 for i in range(6)]
        self.string_model_rel = fallback
        self.is_strings_calibrated = True
        print(f"Strings: detection found {len(detected)}, using evenly-spaced fallback.")
        return True

    def maybe_refine_frets(self, frame):
        """Re-detect frets every 30 successful frames; keep the better result."""
        self.frames_since_lock += 1
        if self.frames_since_lock % 30 != 0:
            return
        if self.refined_quad is None:
            return

        new_frets = self._detect_frets_in_refined_bbox(frame, self.refined_quad)
        if len(new_frets) > len(self.fret_model_rel):
            self.fret_model_rel = new_frets

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

        # Preview mode only: show blue ROI before lock/tracking
        if not self.is_tracking or self.refined_quad is None:
            preview_quad = self._rel_roi_to_abs_quad(frame.shape)
            preview_quad = self._clamp_quad_to_frame(preview_quad, frame.shape)
            pt = preview_quad.astype(int)

            cv2.line(out, tuple(pt[0]), tuple(pt[1]), (255, 0, 0), 2)
            cv2.line(out, tuple(pt[1]), tuple(pt[3]), (255, 0, 0), 2)
            cv2.line(out, tuple(pt[3]), tuple(pt[2]), (255, 0, 0), 2)
            cv2.line(out, tuple(pt[2]), tuple(pt[0]), (255, 0, 0), 2)
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

    def _refine_boundary_via_frets(self, frame, refined_quad, fret_rels):
        """
        Refine fretboard boundary using column-wise intensity gradient
        edge detection with ADAPTIVE thresholding for varying lighting.

        Uses percentile-based thresholds instead of fixed values to handle
        both bright and dim lighting conditions.

        Returns a tighter refined_quad (trapezoid), or None on failure.
        """
        # --- crop an EXPANDED bounding box so search zones reach real edges ---
        x0, y0, x1, y1 = self._quad_to_bbox(refined_quad)
        h, w = frame.shape[:2]

        # Expand vertically so the search zones straddle the actual neck edge
        band_h = y1 - y0
        expand = int(0.40 * band_h)
        y0 = y0 - expand
        y1 = y1 + expand

        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        ch, cw = crop.shape[:2]
        if ch < 20 or cw < 40:
            return None

        # --- ENHANCED preprocessing for lighting robustness ---
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 1. Aggressive CLAHE with higher clipLimit for low-contrast images
        img_std = np.std(gray)
        adaptive_clip = max(2.0, min(6.0, 400.0 / (img_std + 1)))  # stronger for low contrast
        clahe_adaptive = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(8, 8))
        gray = clahe_adaptive.apply(gray)

        # 2. Local contrast normalization (helps with uneven lighting)
        gray_float = gray.astype(np.float32)
        local_mean = cv2.GaussianBlur(gray_float, (51, 51), 0)
        local_std = cv2.GaussianBlur((gray_float - local_mean) ** 2, (51, 51), 0)
        local_std = np.sqrt(local_std) + 1e-6
        gray_norm = ((gray_float - local_mean) / local_std * 40 + 128).clip(0, 255).astype(np.uint8)

        # 3. Bilateral filter on normalized image
        gray = cv2.bilateralFilter(gray_norm, d=7, sigmaColor=40, sigmaSpace=40)

        # --- initial boundary in crop coordinates (for search zones) ---
        tl, tr, bl, br = refined_quad
        origin = np.array([x0, y0], dtype=np.float32)
        tl_c = tl - origin
        tr_c = tr - origin
        bl_c = bl - origin
        br_c = br - origin

        # --- parameters ---
        N_COLS = 24
        STRIP_HW = 4          # half-width → 9px strip
        GRAD_SIGMA = 3.0
        SEARCH_MARGIN = 0.25  # fraction of crop height

        top_points = []
        bot_points = []
        margin_px = int(SEARCH_MARGIN * ch)

        # Collect all gradient values for adaptive thresholding
        all_top_grads = []
        all_bot_grads = []
        col_data = []  # store (x_crop, profile, grad, init_top, init_bot)

        for i in range(N_COLS):
            t = (i + 0.5) / N_COLS
            x_crop = int(t * cw)
            if x_crop < STRIP_HW or x_crop >= cw - STRIP_HW:
                continue

            # vertical intensity profile (averaged over strip)
            strip = gray[:, x_crop - STRIP_HW : x_crop + STRIP_HW + 1]
            profile = np.mean(strip, axis=1).astype(np.float64)
            profile = self._smooth_1d(profile, sigma=GRAD_SIGMA)
            grad = np.abs(np.gradient(profile))

            # interpolate initial boundary at this column
            init_top_y = tl_c[1] + t * (tr_c[1] - tl_c[1])
            init_bot_y = bl_c[1] + t * (br_c[1] - bl_c[1])

            # top edge search zone
            t_lo = max(0, int(init_top_y) - margin_px)
            t_hi = min(ch - 1, int(init_top_y) + margin_px)
            if t_hi - t_lo > 2:
                zone = grad[t_lo:t_hi + 1]
                all_top_grads.extend(zone.tolist())

            # bottom edge search zone
            b_lo = max(0, int(init_bot_y) - margin_px)
            b_hi = min(ch - 1, int(init_bot_y) + margin_px)
            if b_hi - b_lo > 2:
                zone = grad[b_lo:b_hi + 1]
                all_bot_grads.extend(zone.tolist())

            col_data.append((x_crop, grad, init_top_y, init_bot_y))

        # --- ADAPTIVE thresholding based on percentile ---
        # Use 60th percentile of gradient values as threshold (works in any lighting)
        if len(all_top_grads) > 10:
            top_threshold = np.percentile(all_top_grads, 60)
        else:
            top_threshold = 5
        if len(all_bot_grads) > 10:
            bot_threshold = np.percentile(all_bot_grads, 60)
        else:
            bot_threshold = 5

        # Ensure minimum threshold to avoid noise
        top_threshold = max(top_threshold, 3)
        bot_threshold = max(bot_threshold, 3)

        # --- Now find edge points using adaptive thresholds ---
        for x_crop, grad, init_top_y, init_bot_y in col_data:
            t_lo = max(0, int(init_top_y) - margin_px)
            t_hi = min(ch - 1, int(init_top_y) + margin_px)
            if t_hi - t_lo > 2:
                zone = grad[t_lo:t_hi + 1]
                pk = int(np.argmax(zone))
                if zone[pk] >= top_threshold:
                    top_points.append((x_crop, t_lo + pk))

            b_lo = max(0, int(init_bot_y) - margin_px)
            b_hi = min(ch - 1, int(init_bot_y) + margin_px)
            if b_hi - b_lo > 2:
                zone = grad[b_lo:b_hi + 1]
                pk = int(np.argmax(zone))
                if zone[pk] >= bot_threshold:
                    bot_points.append((x_crop, b_lo + pk))

        if len(top_points) < 5 or len(bot_points) < 5:
            return None

        # --- RANSAC line fit ---
        top_pts = np.array(top_points, dtype=np.float64)
        bot_pts = np.array(bot_points, dtype=np.float64)

        a_top, b_top = self._fit_line_ransac(
            top_pts[:, 0], top_pts[:, 1], n_iter=80, inlier_thresh=4.0)
        a_bot, b_bot = self._fit_line_ransac(
            bot_pts[:, 0], bot_pts[:, 1], n_iter=80, inlier_thresh=4.0)

        if a_top is None or a_bot is None:
            return None

        # --- evaluate fitted lines at crop edges ---
        top_y_left  = b_top
        top_y_right = a_top * (cw - 1) + b_top
        bot_y_left  = b_bot
        bot_y_right = a_bot * (cw - 1) + b_bot

        # sanity checks
        if top_y_left >= bot_y_left or top_y_right >= bot_y_right:
            return None
        if (bot_y_left - top_y_left) < ch * 0.15:
            return None
        if (bot_y_right - top_y_right) < ch * 0.15:
            return None
        if abs(a_top) > 0.3 or abs(a_bot) > 0.3:
            return None

        # --- build trapezoid in frame coords ---
        new_tl = np.array([x0, y0 + top_y_left],  dtype=np.float32)
        new_tr = np.array([x1, y0 + top_y_right], dtype=np.float32)
        new_bl = np.array([x0, y0 + bot_y_left],  dtype=np.float32)
        new_br = np.array([x1, y0 + bot_y_right], dtype=np.float32)

        new_quad = np.array([new_tl, new_tr, new_bl, new_br], dtype=np.float32)

        print(f"  Boundary refinement: {len(top_points)} top + {len(bot_points)} bot "
              f"edge points, top slope={a_top:.4f}, bot slope={a_bot:.4f}")

        return new_quad

    def _fit_line_ransac(self, xs, ys, n_iter=50, inlier_thresh=3.0):
        """
        Fit y = a*x + b using simple RANSAC.
        Returns (a, b) or (None, None) on failure.
        """
        n = len(xs)
        if n < 2:
            return None, None

        # If only 2-3 points, just do least squares
        if n <= 3:
            A = np.vstack([xs, np.ones(n)]).T
            result = np.linalg.lstsq(A, ys, rcond=None)
            a, b = result[0]
            return float(a), float(b)

        best_a, best_b = None, None
        best_inliers = 0

        for _ in range(n_iter):
            idx = np.random.choice(n, 2, replace=False)
            x1, y1 = xs[idx[0]], ys[idx[0]]
            x2, y2 = xs[idx[1]], ys[idx[1]]

            if abs(x2 - x1) < 1e-6:
                continue

            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1

            residuals = np.abs(ys - (a * xs + b))
            inliers = np.sum(residuals < inlier_thresh)

            if inliers > best_inliers:
                best_inliers = inliers
                # Refit on inliers
                mask = residuals < inlier_thresh
                A = np.vstack([xs[mask], np.ones(np.sum(mask))]).T
                result = np.linalg.lstsq(A, ys[mask], rcond=None)
                best_a, best_b = float(result[0][0]), float(result[0][1])

        return best_a, best_b

    def _detect_neck_band_in_roi(self, roi_bgr):
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.28, 0.72

        h, w = roi_bgr.shape[:2]
        if h < 20 or w < 40:
            return 0.28, 0.72

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # --- ADAPTIVE preprocessing for lighting robustness ---
        # 1. Apply CLAHE with adaptive clip based on image contrast
        img_std = np.std(gray)
        adaptive_clip = max(2.0, min(5.0, 300.0 / (img_std + 1)))
        clahe_adaptive = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(8, 8))
        gray_eq = clahe_adaptive.apply(gray)

        blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

        # --- Horizontal edge profile (Sobel Y) ---
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

        # --- Intensity-based refinement with ADAPTIVE thresholding ---
        # The fretboard is typically a distinct brightness band (lighter or
        # darker) relative to the background.  Use the row-mean intensity
        # profile to shrink the band inward if the edge detection overshot.
        row_intensity = np.mean(blur[:, x0:x1], axis=1).astype(np.float32)
        row_intensity = cv2.GaussianBlur(
            row_intensity.reshape(-1, 1), (1, 15), 0
        ).flatten()

        # Fretboard interior intensity (mid-region)
        inner_top = top_idx + (bot_idx - top_idx) // 4
        inner_bot = bot_idx - (bot_idx - top_idx) // 4
        fb_mean = float(np.mean(row_intensity[inner_top:inner_bot + 1]))
        fb_std = float(np.std(row_intensity[inner_top:inner_bot + 1]))

        # Walk inward from each edge until intensity is close to fretboard mean
        # ADAPTIVE threshold based on local intensity variation
        bg_top = float(np.mean(row_intensity[max(0, top_idx - margin):top_idx]))
        bg_bot = float(np.mean(row_intensity[bot_idx:min(h, bot_idx + margin)]))

        # Adaptive intensity threshold: scales with local contrast
        contrast_top = abs(fb_mean - bg_top)
        contrast_bot = abs(fb_mean - bg_bot)

        # Use std-based threshold instead of fixed fraction
        # This adapts to both high and low contrast lighting
        adaptive_intensity_frac = 0.35 if img_std > 30 else 0.50  # more lenient for low contrast

        top_trimmed = top_idx
        min_contrast_thresh = max(5.0, fb_std * 0.5)  # adaptive minimum
        if contrast_top > min_contrast_thresh:
            cutoff = contrast_top * adaptive_intensity_frac
            for r in range(top_idx, inner_top):
                if abs(row_intensity[r] - fb_mean) < cutoff:
                    top_trimmed = r
                    break

        bot_trimmed = bot_idx
        if contrast_bot > min_contrast_thresh:
            cutoff = contrast_bot * adaptive_intensity_frac
            for r in range(bot_idx, inner_bot, -1):
                if abs(row_intensity[r] - fb_mean) < cutoff:
                    bot_trimmed = r
                    break

        # Only accept trimming if the result is still a reasonable band
        if bot_trimmed - top_trimmed >= min_gap:
            top_idx = top_trimmed
            bot_idx = bot_trimmed

        # Tighter clamps — fretboard should be roughly centered in ROI
        top_rel = float(np.clip(top_idx / float(h), 0.12, 0.48))
        bottom_rel = float(np.clip(bot_idx / float(h), 0.52, 0.88))

        if bottom_rel - top_rel < 0.18:
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

    # ---------------------------------------------------------
    # Tracking helpers
    # ---------------------------------------------------------
    def _track_optical_flow(self, gray, frame_shape):
        """Track fretboard via sparse Lucas-Kanade optical flow."""
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.current_track_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if new_pts is None:
            return False

        # Forward-backward consistency check
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, new_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        fb_err = np.linalg.norm(
            self.current_track_pts.reshape(-1, 2) - back_pts.reshape(-1, 2),
            axis=1
        )
        good = ((status.ravel() == 1)
                & (back_status.ravel() == 1)
                & (fb_err < 1.5))

        n_good = int(np.sum(good))
        if n_good < 6:
            return False

        good_locked = self.locked_track_pts[good].reshape(-1, 1, 2)
        good_current = new_pts[good].reshape(-1, 1, 2)

        # Similarity transform (4 DOF: rotation, uniform scale, tx, ty)
        M, inliers = cv2.estimateAffinePartial2D(
            good_locked.reshape(-1, 2),
            good_current.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )

        if M is None:
            return False

        inlier_count = int(np.sum(inliers)) if inliers is not None else 0
        inlier_ratio = inlier_count / max(1, n_good)
        self.last_inlier_ratio = inlier_ratio
        self.last_match_count = n_good

        # Convert 2x3 affine to 3x3 for perspectiveTransform compatibility
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = M.astype(np.float32)

        if not self._apply_tracked_transform(H, inlier_ratio, frame_shape):
            return False

        # Update tracked points (keep only inlier subset)
        self.current_track_pts = good_current.copy().astype(np.float32)
        self.locked_track_pts = good_locked.copy().astype(np.float32)

        # Refresh features if the set has shrunk too much
        if len(self.current_track_pts) < 30:
            self._refresh_track_points(gray)

        return True

    def _track_orb_recovery(self, gray, frame_shape):
        """ORB feature matching fallback for recovery when optical flow fails."""
        search_crop, (sx0, sy0, sx1, sy1) = self._crop_bbox_from_quad(
            gray, self.preview_quad
        )

        search_mask = self._make_neck_band_mask(
            search_crop.shape,
            self.locked_top_rel,
            self.locked_bottom_rel
        )

        search_enhanced = self.clahe.apply(search_crop)
        curr_kp, curr_des = self.detector.detectAndCompute(
            search_enhanced, search_mask
        )

        if curr_des is None or curr_kp is None or len(curr_kp) < 12:
            return False

        matches_knn = self.matcher.knnMatch(self.locked_des, curr_des, k=2)

        good_matches = []
        for pair in matches_knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        self.last_match_count = len(good_matches)

        if len(good_matches) < 10:
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

        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        if H is None or inliers is None:
            return False

        inlier_ratio = float(np.sum(inliers)) / float(len(inliers))
        self.last_inlier_ratio = inlier_ratio

        return self._apply_tracked_transform(
            H.astype(np.float32), inlier_ratio, frame_shape
        )

    def _apply_tracked_transform(self, H, inlier_ratio, frame_shape):
        """Apply transform to locked_quad, validate, and update refined_quad."""
        tracked_quad = cv2.perspectiveTransform(
            self.locked_quad.reshape(-1, 1, 2).astype(np.float32),
            H
        ).reshape(-1, 2)

        if inlier_ratio < 0.40:
            return False

        self.last_good_matrix = H.copy()
        self.tracking_ok = True
        self.failed_frames = 0

        tracked_quad = self._clamp_quad_to_frame(tracked_quad, frame_shape)

        if not self._corners_are_reasonable(tracked_quad, frame_shape):
            return False

        new_refined = self._build_refined_quad_from_roi(
            tracked_quad,
            self.locked_top_rel,
            self.locked_bottom_rel
        )
        # Apply same 6% vertical expansion as in lock() for consistent alignment
        new_refined = self._scale_quad_about_center(new_refined, sx=1.0, sy=1.06)

        if self.refined_quad is not None:
            self.refined_quad = (self.smooth_alpha * new_refined
                                 + (1 - self.smooth_alpha) * self.refined_quad)
        else:
            self.refined_quad = new_refined
        return True

    def _init_track_points(self, gray):
        """Detect good features in the current fretboard region for optical flow."""
        if self.refined_quad is None or self.last_good_matrix is None:
            return
        mask = self._make_refined_mask(gray.shape, self.refined_quad)
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=200, qualityLevel=0.01, minDistance=7,
            mask=mask
        )
        if pts is None or len(pts) < 6:
            return

        pts = pts.astype(np.float32)
        self.current_track_pts = pts.copy()

        # Map current-frame points back to locked reference frame
        H_inv = np.linalg.inv(self.last_good_matrix)
        self.locked_track_pts = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2), H_inv.astype(np.float32)
        ).reshape(-1, 1, 2).astype(np.float32)

    def _refresh_track_points(self, gray):
        """Re-detect features in the current fretboard region."""
        self._init_track_points(gray)

    def _make_refined_mask(self, frame_shape, quad):
        """Create a binary mask covering the refined quad region."""
        h, w = frame_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = quad.astype(np.int32)
        # quad order: tl, tr, bl, br -> polygon: tl, tr, br, bl
        polygon = np.array([pts[0], pts[1], pts[3], pts[2]])
        cv2.fillConvexPoly(mask, polygon, 255)
        return mask

    # ---------------------------------------------------------
    # Fret detection helpers
    # ---------------------------------------------------------

    def _smooth_1d(self, signal, sigma=2.0):
        """Gaussian-smooth a 1D signal."""
        k = int(3 * sigma) * 2 + 1
        x = np.arange(k) - k // 2
        kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return np.convolve(signal, kernel, mode='same')

    def _find_peaks_1d(self, signal, min_prominence, min_distance):
        """Find local maxima with minimum prominence and spacing."""
        n = len(signal)
        if n < 3:
            return []

        # Find all local maxima
        candidates = []
        for i in range(1, n - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                candidates.append(i)

        if not candidates:
            return []

        # Compute approximate prominence for each candidate
        window = max(min_distance * 3, 10)
        scored = []
        for idx in candidates:
            left = max(0, idx - window)
            right = min(n, idx + window + 1)
            left_min = float(np.min(signal[left:idx])) if idx > left else float(signal[idx])
            right_min = float(np.min(signal[idx + 1:right])) if idx + 1 < right else float(signal[idx])
            prominence = float(signal[idx]) - max(left_min, right_min)
            if prominence >= min_prominence:
                scored.append((idx, float(signal[idx])))

        if not scored:
            return []

        # Non-maximum suppression: keep highest peaks, enforce min_distance
        scored.sort(key=lambda x: -x[1])
        selected = []
        for idx, height in scored:
            if all(abs(idx - s) >= min_distance for s in selected):
                selected.append(idx)

        selected.sort()
        return selected

    def _validate_fret_spacing(self, peak_xs, ratio_tolerance=0.20):
        """
        Keep only peaks forming a geometrically consistent fret pattern.
        Real guitar frets follow 12-TET spacing where each interval is
        ~0.9439x the previous.  We check that consecutive spacing ratios
        are consistent, allowing for perspective distortion.
        """
        if len(peak_xs) < 3:
            return peak_xs

        peak_xs = sorted(peak_xs)
        spacings = np.diff(peak_xs).astype(float)

        if len(spacings) < 2:
            return peak_xs

        ratios = spacings[1:] / np.maximum(spacings[:-1], 1.0)

        valid_mask = (ratios > 0.5) & (ratios < 2.0)
        if not np.any(valid_mask):
            return peak_xs

        median_ratio = float(np.median(ratios[valid_mask]))

        consistent = np.abs(ratios / median_ratio - 1.0) <= ratio_tolerance

        # Find longest consecutive run of consistent ratios
        best_start = 0
        best_count = 0
        run_start = 0

        for i in range(len(consistent)):
            if consistent[i]:
                count = i - run_start + 1
                if count > best_count:
                    best_count = count
                    best_start = run_start
            else:
                run_start = i + 1

        if best_count == 0:
            return peak_xs

        # A run of k consistent ratios involves k+2 peaks
        first = best_start
        last = best_start + best_count + 1
        return peak_xs[first:last + 1]

    def _extrapolate_frets(self, seed_xs, col_profile, cw):
        """
        Given a validated set of fret positions (seed_xs) and the 1D edge
        profile, predict additional frets in both directions using the
        12-TET geometric model, and confirm them against the profile.
        """
        if len(seed_xs) < 2:
            return seed_xs

        seed_xs = sorted(seed_xs)
        spacings = np.diff(seed_xs).astype(float)

        if len(spacings) < 1:
            return seed_xs

        # Estimate the spacing ratio from the seed frets
        if len(spacings) >= 2:
            ratios = spacings[1:] / np.maximum(spacings[:-1], 1.0)
            valid = ratios[(ratios > 0.7) & (ratios < 1.3)]
            ratio = float(np.median(valid)) if len(valid) > 0 else 0.9439
        else:
            ratio = 0.9439  # 12-TET default: 2^(-1/12)

        # Clamp ratio to a reasonable range
        ratio = float(np.clip(ratio, 0.80, 1.05))

        # Minimum peak height at a predicted position to accept it
        profile_max = float(np.max(col_profile))
        confirm_threshold = 0.06 * profile_max
        margin = max(5, int(cw * 0.02))

        all_frets = list(seed_xs)

        # Extrapolate toward higher frets (narrower spacing, toward body)
        last_spacing = float(spacings[-1])
        last_x = float(seed_xs[-1])
        for _ in range(12):
            next_spacing = last_spacing * ratio
            if next_spacing < 5:
                break
            predicted_x = last_x + next_spacing
            if predicted_x >= cw - margin:
                break
            # Search for a peak near the predicted position
            confirmed_x = self._confirm_peak_near(
                col_profile, predicted_x, next_spacing, confirm_threshold
            )
            if confirmed_x is not None:
                all_frets.append(confirmed_x)
                last_x = float(confirmed_x)
                last_spacing = float(confirmed_x) - float(all_frets[-2])
            else:
                break

        # Extrapolate toward lower frets (wider spacing, toward nut)
        first_spacing = float(spacings[0])
        first_x = float(seed_xs[0])
        for _ in range(6):
            prev_spacing = first_spacing / max(ratio, 0.5)
            if prev_spacing > cw * 0.4:
                break
            predicted_x = first_x - prev_spacing
            if predicted_x <= margin:
                break
            confirmed_x = self._confirm_peak_near(
                col_profile, predicted_x, prev_spacing, confirm_threshold
            )
            if confirmed_x is not None:
                all_frets.insert(0, confirmed_x)
                first_x = float(confirmed_x)
                first_spacing = float(all_frets[1]) - float(confirmed_x)
            else:
                break

        return sorted(set(all_frets))

    def _confirm_peak_near(self, col_profile, predicted_x, spacing, threshold):
        """
        Look for a local peak in col_profile near predicted_x.
        Search window is ±30% of the expected spacing.
        Returns the peak x if found, else None.
        """
        n = len(col_profile)
        window = max(3, int(spacing * 0.30))
        lo = max(0, int(predicted_x - window))
        hi = min(n - 1, int(predicted_x + window))

        if hi <= lo:
            return None

        segment = col_profile[lo:hi + 1]
        local_max_idx = int(np.argmax(segment))
        local_max_val = float(segment[local_max_idx])

        if local_max_val < threshold:
            return None

        # Check it's actually a local peak (not just the edge of the window)
        abs_idx = lo + local_max_idx
        if abs_idx > 0 and abs_idx < n - 1:
            if (col_profile[abs_idx] >= col_profile[abs_idx - 1]
                    and col_profile[abs_idx] >= col_profile[abs_idx + 1]):
                return abs_idx

        return None

    # ---------------------------------------------------------
    # String detection helpers
    # ---------------------------------------------------------

    def _validate_string_spacing(self, peak_ys, tolerance=0.35):
        """
        Keep only peaks forming a uniformly-spaced pattern.
        Guitar strings are approximately equally spaced on the fretboard.
        Returns the longest run of consistently-spaced peaks.
        """
        if len(peak_ys) < 3:
            return peak_ys

        peak_ys = sorted(peak_ys)
        spacings = np.diff(peak_ys).astype(float)

        if len(spacings) < 2:
            return peak_ys

        median_spacing = float(np.median(spacings))
        if median_spacing < 1.0:
            return peak_ys

        consistent = np.abs(spacings / median_spacing - 1.0) <= tolerance

        # Find longest consecutive run of consistent spacings
        best_start = 0
        best_count = 0
        run_start = 0

        for i in range(len(consistent)):
            if consistent[i]:
                count = i - run_start + 1
                if count > best_count:
                    best_count = count
                    best_start = run_start
            else:
                run_start = i + 1

        if best_count == 0:
            return peak_ys

        # A run of k consistent spacings involves k+1 peaks
        first = best_start
        last = best_start + best_count
        return peak_ys[first:last + 1]

    def _detect_strings_in_refined_bbox(self, frame, refined_quad):
        """
        Detect 6 guitar strings within the refined fretboard quad.
        Returns a list of 6 relative Y positions in [0, 1], or [] on failure.

        Primary method: intensity dip detection (strings are thin dark lines).
        Secondary method: Sobel Y horizontal edge detection.
        """
        x0, y0, x1, y1 = self._quad_to_bbox(refined_quad)
        h, w = frame.shape[:2]

        # Store original height for relative position calculation
        original_y0, original_y1 = y0, y1
        original_ch = y1 - y0

        # Expand vertically to capture strings near edges of the refined_quad.
        # Without expansion, outermost strings may be truncated and produce
        # weak/missing peaks. 20% expansion gives enough context for edge peaks.
        expand = int(0.20 * original_ch)
        y0 = y0 - expand
        y1 = y1 + expand

        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return []

        ch, cw = crop.shape[:2]
        if ch < 20 or cw < 40:
            return []

        # Offset from expanded crop top to original refined_quad top
        top_offset = original_y0 - y0

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Light normalization — CLAHE only (no bilateral: it kills thin strings)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # Use central columns to avoid fretboard edge noise
        col_margin = max(5, int(cw * 0.15))
        center = gray_eq[:, col_margin:cw - col_margin]

        # Use original height for string distance calculation
        min_string_dist = max(3, original_ch // 14)
        # Small margin within the original region (not the expanded crop)
        margin = max(2, int(original_ch * 0.02))

        # Define the valid range for peaks: within original region with small tolerance
        peak_min = top_offset + margin
        peak_max = top_offset + original_ch - margin

        # ---- Method 1: Intensity dip detection (primary) ----
        # Strings appear as thin dark lines on the lighter fretboard
        intensity_profile = np.mean(center, axis=1).astype(np.float64)
        intensity_profile = self._smooth_1d(intensity_profile, sigma=1.5)

        # Invert so dips become peaks
        inverted = float(np.max(intensity_profile)) - intensity_profile

        if float(np.max(inverted)) > 1e-6:
            for prominence_factor in [0.12, 0.07, 0.04]:
                min_prom = prominence_factor * float(np.max(inverted))
                peaks = self._find_peaks_1d(inverted, min_prom, min_string_dist)
                # Filter to peaks within original refined_quad region
                peaks = [p for p in peaks if peak_min < p < peak_max]

                print(f"  Strings intensity: prom={prominence_factor}, "
                      f"found {len(peaks)} peaks")

                # Pass original_ch and top_offset for correct relative conversion
                result = self._try_select_6_strings(
                    peaks, inverted, original_ch, top_offset
                )
                if result is not None:
                    print("Strings detected via intensity dips.")
                    return result

        # ---- Method 2: Sobel Y edge detection (secondary) ----
        sobely = cv2.Sobel(gray_eq, cv2.CV_64F, 0, 1, ksize=3)
        sobely = np.abs(sobely)
        sobely_center = sobely[:, col_margin:cw - col_margin]

        row_profile = np.mean(sobely_center, axis=1)
        row_profile = self._smooth_1d(row_profile, sigma=1.5)

        if float(np.max(row_profile)) > 1e-6:
            for prominence_factor in [0.15, 0.08, 0.04]:
                min_prom = prominence_factor * float(np.max(row_profile))
                peaks = self._find_peaks_1d(row_profile, min_prom, min_string_dist)
                # Filter to peaks within original refined_quad region
                peaks = [p for p in peaks if peak_min < p < peak_max]

                print(f"  Strings Sobel: prom={prominence_factor}, "
                      f"found {len(peaks)} peaks")

                result = self._try_select_6_strings(
                    peaks, row_profile, original_ch, top_offset
                )
                if result is not None:
                    print("Strings detected via Sobel Y edges.")
                    return result

        return []

    def _try_select_6_strings(self, peaks, profile, ch, top_offset=0):
        """
        Given a set of candidate peaks, try to extract exactly 6
        uniformly-spaced strings. Handles 5, 6, 7+ peak counts.
        Returns list of 6 relative positions or None.

        Args:
            peaks: candidate peak positions in expanded crop coordinates
            profile: the intensity/edge profile array
            ch: original (non-expanded) crop height for relative conversion
            top_offset: offset from expanded crop top to original crop top
        """
        if len(peaks) < 4:
            return None

        validated = self._validate_string_spacing(peaks)

        if len(validated) == 6:
            return [(y - top_offset) / float(ch) for y in validated]

        if len(validated) == 5:
            result = self._extrapolate_missing_string(
                validated, profile, ch, 0.0  # no threshold — just find nearest peak
            )
            if result is not None and len(result) == 6:
                return [(y - top_offset) / float(ch) for y in result]

        if len(validated) >= 7:
            result = self._drop_weakest_string(validated, profile)
            if result is not None and len(result) == 6:
                return [(y - top_offset) / float(ch) for y in result]

        # If validation trimmed too aggressively but we have enough raw peaks,
        # try picking the best 6 from raw peaks by uniform spacing score
        if len(peaks) >= 6:
            result = self._best_6_from_candidates(peaks, ch)
            if result is not None:
                return [(y - top_offset) / float(ch) for y in result]

        return None

    def _best_6_from_candidates(self, peaks, ch):
        """
        From N >= 6 candidate peaks, pick the 6-subset with most uniform spacing.
        Only tries combinations if N <= 12 to keep it fast.
        """
        from itertools import combinations

        peaks = sorted(peaks)
        if len(peaks) > 12:
            peaks = peaks[:12]  # limit combinatorics

        best = None
        best_score = float('inf')

        for combo in combinations(peaks, 6):
            spacings = np.diff(combo).astype(float)
            median_sp = float(np.median(spacings))
            if median_sp < 1.0:
                continue
            score = float(np.std(spacings / median_sp))
            if score < best_score:
                best_score = score
                best = list(combo)

        # Accept if spacing is reasonably uniform (CV < 25%)
        if best is not None and best_score < 0.25:
            return best

        return None

    def _extrapolate_missing_string(self, peaks, profile, ch, threshold):
        """
        Given 5 validated string peaks, try to find the 6th at top or bottom
        using the median spacing.
        """
        peaks = sorted(peaks)
        spacings = np.diff(peaks).astype(float)
        median_sp = float(np.median(spacings))

        candidates = []

        # Try adding one above (before first peak)
        predicted_top = peaks[0] - median_sp
        if predicted_top > 1:
            confirmed = self._confirm_peak_near(profile, predicted_top, median_sp, threshold)
            if confirmed is not None:
                candidates.append(sorted([confirmed] + peaks))

        # Try adding one below (after last peak)
        predicted_bot = peaks[-1] + median_sp
        if predicted_bot < ch - 1:
            confirmed = self._confirm_peak_near(profile, predicted_bot, median_sp, threshold)
            if confirmed is not None:
                candidates.append(sorted(peaks + [confirmed]))

        # Return the candidate with more uniform spacing
        best = None
        best_score = float('inf')
        for c in candidates:
            sp = np.diff(c).astype(float)
            med = float(np.median(sp))
            if med > 0:
                score = float(np.std(sp / med))
                if score < best_score:
                    best_score = score
                    best = c

        return best

    def _drop_weakest_string(self, peaks, profile):
        """
        Given 7 peaks, drop the one with lowest profile value and re-validate
        to get 6 uniformly-spaced strings.
        """
        peaks = sorted(peaks)
        # Score each peak by its profile value
        scored = [(p, float(profile[p])) for p in peaks]
        scored.sort(key=lambda x: x[1])

        # Try dropping each peak starting from weakest
        for drop_peak, _ in scored:
            remaining = [p for p in peaks if p != drop_peak]
            validated = self._validate_string_spacing(remaining)
            if len(validated) == 6:
                return validated

        return None

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

        # Stage 1: Lighting normalization
        # CLAHE with 8x8 tiles for fine-grained local contrast equalisation
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # Bilateral filter: preserve sharp fret-wire edges, smooth wood grain
        gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

        # Stage 2: Vertical edge extraction (fret wires are near-vertical)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.abs(sobelx)

        # Stage 3: 1D column projection — average edge response across rows.
        # True fret wires span the full crop height and accumulate a strong
        # response; random noise and reflections average out.
        col_profile = np.mean(sobelx, axis=0)
        col_profile = self._smooth_1d(col_profile, sigma=2.0)

        if float(np.max(col_profile)) < 1e-6:
            return []

        # Stage 4: Peak detection — multi-attempt with decreasing strictness
        # Use a small min distance so closely-spaced higher frets aren't filtered
        min_fret_dist = max(6, cw // 40)

        for prominence_factor in [0.20, 0.12, 0.08]:
            min_prom = prominence_factor * float(np.max(col_profile))
            peaks = self._find_peaks_1d(col_profile, min_prom, min_fret_dist)

            # Remove peaks too close to crop edges
            margin = max(5, int(cw * 0.02))
            peaks = [p for p in peaks if margin < p < cw - margin]

            if len(peaks) < 2:
                continue

            # Stage 5: Geometric validation (12-TET fret spacing consistency)
            validated = self._validate_fret_spacing(peaks)

            if len(validated) >= 2:
                # Stage 6: Extrapolate to find frets missed by peak detection
                extended = self._extrapolate_frets(validated, col_profile, cw)
                return [x / float(cw) for x in extended]

        return []

    def _get_search_quad(self, frame_shape):
        if not self.is_tracking or self.locked_quad is None:
            return self._rel_roi_to_abs_quad(frame_shape)

        current_quad = cv2.perspectiveTransform(
            self.locked_quad.reshape(-1, 1, 2).astype(np.float32),
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