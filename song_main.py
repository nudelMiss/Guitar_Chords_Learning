import cv2
import numpy as np
import Chords
import string_and_frets as sf
import time
from songs_data import SONGS


class GuitarSystem:
    def __init__(self):
        # -----------------------------
        # Main state
        # -----------------------------
        self.is_tracking = False
        self.is_strings_calibrated = False

        self.tracking_pts = None
        self.last_gray = None

        # Main tracked fretboard quad: tl, tr, bl, br
        self.refined_quad = None

        self.locked_model = {}
        self.fret_model_rel = []
        self.string_model_rel = []

        # ROI tuned for website camera view
        self.roi_x0_rel = 0.300
        self.roi_x1_rel = 0.880
        self.roi_y0_rel = 0.560
        self.roi_y1_rel = 0.740

        # Tracking params
        self.min_good_points = 6
        self.grid_nx = 10
        self.grid_ny = 4

        # Neck band smoothing only at lock time
        self.prev_neck_top_rel = None
        self.prev_neck_bottom_rel = None
        self.neck_band_smooth_alpha = 0.70

        # Quad smoothing (low smoothing to avoid lag)
        self.prev_refined_quad = None
        self.refined_quad_alpha = 0.10

        # Initialize chord state variables
        self.available_chords = Chords.list_available_chords()
        self.available_chords.sort()
        self.chord_idx = 0
        self.current_chord_name = None
        self.current_chord_data = None

        # Song management
        self.current_lyric = ""
        self.is_playing_song = False
        self.active_song_id = None
        self.song_chords_sequence = []
        self.current_chord_index = 0
        self.chord_start_time = 0.0

        self.tracked_roi_quad = None

    # =========================================================
    # Song logic
    # =========================================================
    def start_playing_song(self, song_id):
        self.is_playing_song = True
        self.active_song_id = song_id
        self.song_chords_sequence = SONGS.get(song_id, [])
        self.current_chord_index = 0
        self.chord_start_time = time.time()
        self._update_chord_from_sequence()
        print(f"Started playing: {song_id}")

    def stop_playing_song(self):
        self.is_playing_song = False
        self.active_song_id = None
        self.song_chords_sequence = []
        self.current_chord_name = None
        self.current_chord_data = None
        print("Stopped playing song. Chord cleared from screen.")

    def _update_chord_from_sequence(self):
        if self.song_chords_sequence and self.current_chord_index < len(self.song_chords_sequence):
            node = self.song_chords_sequence[self.current_chord_index]
            self.current_chord_name = node["chord"]
            self.current_chord_data = Chords.get_chord_data(self.current_chord_name)
            self.current_lyric = node.get("lyric", "")

    # =========================================================
    # ROI / geometry helpers
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

    def _crop_bbox_from_quad(self, frame, quad):
        x0, y0, x1, y1 = self._quad_to_bbox(quad)
        h, w = frame.shape[:2]

        x0 = max(0, min(w - 1, x0))
        x1 = max(x0 + 1, min(w, x1))
        y0 = max(0, min(h - 1, y0))
        y1 = max(y0 + 1, min(h, y1))

        roi = frame[y0:y1, x0:x1].copy()
        return roi, (x0, y0, x1, y1)

    def _clamp_quad_to_frame(self, quad, frame_shape):
        h, w = frame_shape[:2]
        out = quad.copy()
        out[:, 0] = np.clip(out[:, 0], 0, w - 1)
        out[:, 1] = np.clip(out[:, 1], 0, h - 1)
        return out

    def _corners_are_reasonable(self, corners, frame_shape):
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

    def _smooth_scalar(self, prev_val, new_val, alpha):
        if prev_val is None:
            return new_val
        return alpha * prev_val + (1.0 - alpha) * new_val

    def _smooth_quad(self, quad):
        if self.prev_refined_quad is None:
            self.prev_refined_quad = quad.copy()
            return quad
        smoothed = self.refined_quad_alpha * self.prev_refined_quad + (1.0 - self.refined_quad_alpha) * quad
        self.prev_refined_quad = smoothed.copy()
        return smoothed

    def _reset_smoothing(self):
        self.prev_neck_top_rel = None
        self.prev_neck_bottom_rel = None
        self.prev_refined_quad = None

    # =========================================================
    # Neck band detection only at lock time
    # =========================================================
    def _detect_neck_band_in_roi(self, roi_bgr):
        """
        Detect top/bottom of fretboard inside ROI.
        Returns top_rel, bottom_rel in [0,1].
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return 0.25, 0.75

        h, w = roi_bgr.shape[:2]
        if h < 20 or w < 40:
            return 0.25, 0.75

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
            return 0.25, 0.75

        top_idx = int(np.argmax(top_search)) + margin
        bot_idx = int(np.argmax(bot_search)) + mid

        min_gap = max(12, int(0.18 * h))
        if bot_idx - top_idx < min_gap:
            top_rel = 0.28
            bottom_rel = 0.72
        else:
            top_rel = top_idx / float(h)
            bottom_rel = bot_idx / float(h)

        top_rel = float(np.clip(top_rel, 0.05, 0.65))
        bottom_rel = float(np.clip(bottom_rel, 0.35, 0.95))

        if bottom_rel - top_rel < 0.15:
            top_rel, bottom_rel = 0.28, 0.72

        top_rel = self._smooth_scalar(self.prev_neck_top_rel, top_rel, self.neck_band_smooth_alpha)
        bottom_rel = self._smooth_scalar(self.prev_neck_bottom_rel, bottom_rel, self.neck_band_smooth_alpha)

        self.prev_neck_top_rel = top_rel
        self.prev_neck_bottom_rel = bottom_rel

        return top_rel, bottom_rel

    def _build_initial_refined_quad_from_roi(self, roi_quad, top_rel, bottom_rel):
        """
        Build refined fretboard quad once from ROI.
        Uses ROI left/right and detected top/bottom.
        """
        tl, tr, bl, br = roi_quad

        top_left = tl + top_rel * (bl - tl)
        top_right = tr + top_rel * (br - tr)
        bottom_left = tl + bottom_rel * (bl - tl)
        bottom_right = tr + bottom_rel * (br - tr)

        return np.array(
            [top_left, top_right, bottom_left, bottom_right],
            dtype=np.float32
        )

    # =========================================================
    # Fret detection only at lock time
    # =========================================================
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
        if refined_quad is None:
            return []

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
            sobelx,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
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

    # =========================================================
    # Grid tracking (based on rotate_neck_test logic)
    # =========================================================
    def _sample_grid_points_in_quad(self, quad, nx=None, ny=None):
        if nx is None:
            nx = self.grid_nx
        if ny is None:
            ny = self.grid_ny

        tl, tr, bl, br = quad.astype(np.float32)
        pts = []

        # only sample the middle vertical region of the ROI
        v0, v1 = 0.20, 0.80

        for j in range(ny):
            v = v0 + (v1 - v0) * (j / max(1, ny - 1))
            left = tl + v * (bl - tl)
            right = tr + v * (br - tr)

            for i in range(nx):
                u = i / max(1, nx - 1)
                p = left + u * (right - left)
                pts.append([p[0], p[1]])

        return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    def _pick_features_in_quad(self, gray, quad):
        return self._sample_grid_points_in_quad(quad)

    def _reseed_features_if_needed(self, gray, quad):
        if self.tracking_pts is None or len(self.tracking_pts) < 20:
            self.tracking_pts = self._sample_grid_points_in_quad(self.tracked_roi_quad)
            self.last_gray = gray.copy()

    def _estimate_homography_update(self, prev_gray, gray, prev_pts, prev_quad):
        lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, prev_pts, None, **lk_params
        )

        if new_pts is None or status is None:
            return None, None, None, False

        status = status.flatten()
        good_old = prev_pts[status == 1]
        good_new = new_pts[status == 1]

        if len(good_old) < self.min_good_points:
            return None, None, None, False

        M, inliers = cv2.estimateAffinePartial2D(
            good_old.reshape(-1, 2),
            good_new.reshape(-1, 2),
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0
        )

        if M is None:
            return None, None, None, False

        new_quad = cv2.transform(
            prev_quad.reshape(-1, 1, 2), M
        ).reshape(-1, 2)

        updated_pts = cv2.transform(
            prev_pts.reshape(-1, 1, 2), M
        ).reshape(-1, 1, 2).astype(np.float32)

        return new_quad, updated_pts, M, True

    # =========================================================
    # Geometry / overlay
    # =========================================================
    def _current_locked_model_from_tracked_corners(self, tracked_corners):
        xs = tracked_corners[:, 0]
        ys = tracked_corners[:, 1]
        return {
            "x_min": int(np.min(xs)),
            "x_max": int(np.max(xs)),
            "y_t": int(np.min(ys)),
            "y_b": int(np.max(ys)),
        }

    def get_dot_coordinates(self, fret_num, string_num, tracked_corners, mirror_frets=True):
        if not self.fret_model_rel or not self.string_model_rel:
            return -1, -1
        if string_num < 1 or string_num > 6:
            return -1, -1
        if fret_num < 1 or fret_num > len(self.fret_model_rel):
            return -1, -1

        tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

        rel_y = float(self.string_model_rel[6 - string_num])

        num_frets = len(self.fret_model_rel)
        use_fret = (num_frets - fret_num + 1) if mirror_frets else fret_num

        if use_fret == 1:
            rel_x = float(self.fret_model_rel[0]) / 2.0
        else:
            rel_x = (float(self.fret_model_rel[use_fret - 1]) + float(self.fret_model_rel[use_fret - 2])) / 2.0

        top_point = tl + rel_x * (tr - tl)
        bot_point = bl + rel_x * (br - bl)
        pt = top_point + rel_y * (bot_point - top_point)

        return int(pt[0]), int(pt[1])

    # =========================================================
    # Lock initialization
    # =========================================================
    def _lock_from_saved_roi(self, frame, gray):
        roi_quad = self._rel_roi_to_abs_quad(frame.shape)
        roi_quad = self._clamp_quad_to_frame(roi_quad, frame.shape)

        self._reset_smoothing()

        # IMPORTANT: save parent ROI quad BEFORE sampling points from it
        self.tracked_roi_quad = roi_quad.copy()

        roi, _ = self._crop_bbox_from_quad(frame, roi_quad)
        top_rel, bottom_rel = self._detect_neck_band_in_roi(roi)

        initial_refined_quad = self._build_initial_refined_quad_from_roi(
            roi_quad, top_rel, bottom_rel
        )
        initial_refined_quad = self._clamp_quad_to_frame(initial_refined_quad, frame.shape)
        initial_refined_quad = self._smooth_quad(initial_refined_quad)

        self.refined_quad = initial_refined_quad

        # Track the BIG ROI, not the fretboard
        self.tracking_pts = self._sample_grid_points_in_quad(self.tracked_roi_quad)

        if self.tracking_pts is None or len(self.tracking_pts) < self.min_good_points:
            self.is_tracking = False
            self.refined_quad = None
            self.tracked_roi_quad = None
            print("Lock failed: not enough trackable points in ROI.")
            return

        self.last_gray = gray.copy()

        detected_frets_rel = self._detect_frets_in_refined_bbox(frame, self.refined_quad)
        if len(detected_frets_rel) >= 2:
            self.fret_model_rel = detected_frets_rel
            print(f"Neck locked. Detected {len(detected_frets_rel)} fret lines.")
        else:
            self.fret_model_rel = []
            print("Neck locked, but fret detection was weak. Try again while holding still.")

        self.locked_model = self._current_locked_model_from_tracked_corners(self.refined_quad)
        self.is_tracking = True
        self.string_model_rel = []
        self.is_strings_calibrated = False

    # =========================================================
    # Main frame processing
    # =========================================================
    def process_frame(self, frame, key):
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # -----------------------------
        # Commands
        # -----------------------------
        if not self.is_playing_song:
            if key == ord('r'):
                self.__init__()
                print("Reset All.")
                return display_frame

            if key == ord('x'):
                self.string_model_rel = []
                self.is_strings_calibrated = False
                print("Strings cleared. Press [s] to calibrate again.")

            if key == ord('a'):
                self.chord_idx = (self.chord_idx + 1) % len(self.available_chords)
                self.current_chord_name = self.available_chords[self.chord_idx]
                self.current_chord_data = Chords.get_chord_data(self.current_chord_name)
                print("Chord selected:", self.current_chord_name)

            if key == ord('z'):
                self.chord_idx = (self.chord_idx - 1) % len(self.available_chords)
                self.current_chord_name = self.available_chords[self.chord_idx]
                self.current_chord_data = Chords.get_chord_data(self.current_chord_name)
                print("Chord selected:", self.current_chord_name)

            if key == ord('v'):
                self.current_chord_name = None
                self.current_chord_data = None
                print("Chord cleared.")

            if key == ord('c'):
                self._lock_from_saved_roi(frame, gray)

         # -----------------------------
        # Tracking mode
        # -----------------------------
        if (
            self.is_tracking
            and self.tracking_pts is not None
            and self.last_gray is not None
            and self.refined_quad is not None
            and self.tracked_roi_quad is not None
        ):
            new_quad, new_pts, M, ok_track = self._estimate_homography_update(
                self.last_gray,
                gray,
                self.tracking_pts,
                self.tracked_roi_quad
            )

            if ok_track and new_quad is not None and self._corners_are_reasonable(new_quad, frame.shape):
                # new_quad is the NEW PARENT ROI quad
                new_roi_quad = self._clamp_quad_to_frame(new_quad, frame.shape)

                # Re-detect fretboard band INSIDE the newly tracked ROI
                tracked_roi_crop, _ = self._crop_bbox_from_quad(frame, new_roi_quad)
                top_rel, bottom_rel = self._detect_neck_band_in_roi(tracked_roi_crop)

                new_refined_quad = self._build_initial_refined_quad_from_roi(
                    new_roi_quad,
                    top_rel,
                    bottom_rel
                )

                if self._corners_are_reasonable(new_refined_quad, frame.shape):
                    new_refined_quad = self._clamp_quad_to_frame(new_refined_quad, frame.shape)
                    new_refined_quad = self._smooth_quad(new_refined_quad)

                    self.tracked_roi_quad = new_roi_quad
                    self.refined_quad = new_refined_quad
                    self.tracking_pts = new_pts
                    self.last_gray = gray.copy()

                    self._reseed_features_if_needed(gray, self.tracked_roi_quad)
                else:
                    cv2.putText(
                        display_frame,
                        "Tracked fretboard invalid - press [c] to relock",
                        (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
            else:
                cv2.putText(
                    display_frame,
                    "Tracking weak - press [c] to relock",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            tracked_corners = self.refined_quad
            tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]
            current_locked_model = self._current_locked_model_from_tracked_corners(tracked_corners)

            # --- song timing ---
            if self.is_playing_song and self.song_chords_sequence:
                current_time = time.time()
                current_chord_duration = self.song_chords_sequence[self.current_chord_index]["duration"]
                time_elapsed = current_time - self.chord_start_time

                if time_elapsed >= current_chord_duration:
                    self.current_chord_index += 1
                    if self.current_chord_index >= len(self.song_chords_sequence):
                        self.current_chord_index = 0
                    self.chord_start_time = time.time()
                    self._update_chord_from_sequence()

            # --- strings calibration ---
            if key == ord('s') and not self.is_playing_song:
                detected = sf.detect_strings_in_neck(frame, current_locked_model)

                if len(detected) == 6:
                    self.string_model_rel = detected
                    self.is_strings_calibrated = True
                    print("Strings locked (6 detected / modeled).")
                else:
                    self.string_model_rel = []
                    self.is_strings_calibrated = False
                    print("Strings failed.")

            # -----------------------------
            # Draw scaffolding
            # -----------------------------
            if not self.is_playing_song:
                rc = tracked_corners.astype(int)
                cv2.line(display_frame, tuple(rc[0]), tuple(rc[1]), (255, 0, 0), 2)
                cv2.line(display_frame, tuple(rc[1]), tuple(rc[3]), (255, 0, 0), 1)
                cv2.line(display_frame, tuple(rc[3]), tuple(rc[2]), (255, 0, 0), 2)
                cv2.line(display_frame, tuple(rc[2]), tuple(rc[0]), (255, 0, 0), 1)

                for rel_x in self.fret_model_rel:
                    p1 = (tl + rel_x * (tr - tl)).astype(int)
                    p2 = (bl + rel_x * (br - bl)).astype(int)
                    cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 0), 2)

                for rel_y in self.string_model_rel:
                    p1 = (tl + rel_y * (bl - tl)).astype(int)
                    p2 = (tr + rel_y * (br - tr)).astype(int)
                    cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 255), 1)

                # optional: draw tracking grid points for debugging
                for p in self.tracking_pts.reshape(-1, 2):
                    cv2.circle(display_frame, (int(p[0]), int(p[1])), 2, (255, 100, 0), -1)

            # -----------------------------
            # Draw chord dots
            # -----------------------------
            if self.current_chord_data is not None and self.is_strings_calibrated and len(self.fret_model_rel) >= 2:
                for finger_num, string_num, fret_num in self.current_chord_data.get("fingers", []):
                    x, y = self.get_dot_coordinates(
                        fret_num,
                        string_num,
                        tracked_corners,
                        mirror_frets=True
                    )
                    if x >= 0 and y >= 0:
                        cv2.circle(display_frame, (x, y), 12, (70, 180, 120), -1)
                        cv2.circle(display_frame, (x, y), 13, (255, 255, 255), 1)

                        text = str(finger_num)
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        text_x = x - text_size[0] // 2
                        text_y = y + text_size[1] // 2

                        cv2.putText(
                            display_frame,
                            text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2
                        )

            # -----------------------------
            # HUD
            # -----------------------------
            if not self.is_playing_song:
                if self.current_chord_name:
                    cv2.putText(
                        display_frame,
                        f"Chord: {self.current_chord_name}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2
                    )

                if not self.is_strings_calibrated:
                    cv2.putText(
                        display_frame,
                        "Press [s] to calibrate strings",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                cv2.putText(
                    display_frame,
                    "Press [c] to relock fretboard",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

        # -----------------------------
        # Preview mode
        # -----------------------------
        else:
            preview_quad = self._rel_roi_to_abs_quad(frame.shape)
            preview_quad = self._clamp_quad_to_frame(preview_quad, frame.shape)
            pt = preview_quad.astype(int)

            cv2.line(display_frame, tuple(pt[0]), tuple(pt[1]), (0, 255, 255), 2)
            cv2.line(display_frame, tuple(pt[1]), tuple(pt[3]), (0, 255, 255), 2)
            cv2.line(display_frame, tuple(pt[3]), tuple(pt[2]), (0, 255, 255), 2)
            cv2.line(display_frame, tuple(pt[2]), tuple(pt[0]), (0, 255, 255), 2)

            cv2.putText(
                display_frame,
                "Preview: press [c] to lock fretboard from ROI",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        return display_frame