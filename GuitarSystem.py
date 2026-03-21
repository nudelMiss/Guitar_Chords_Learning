import cv2
import chords_data
import string_and_frets as sf
import time

from songs_data import SONGS

class GuitarSystem:
    def __init__(self, use_mediapipe=True):
        # Tracker
        self.tracker = sf.GuitarNeckTracker()

        # Chord state
        self.available_chords = chords_data.list_available_chords()
        self.available_chords.sort()
        self.chord_idx = 0
        self.current_chord_name = None
        self.current_chord_data = None

        # Song management
        self.current_lyric = ""
        self.current_section = ""
        self.is_playing_song = False
        self.active_song_id = None
        self.song_chords_sequence = []
        self.current_chord_index = 0
        self.chord_start_time = 0.0

        # Countdown state
        self.countdown_active = False
        self.countdown_start_time = 0.0
        self.countdown_duration = 3  # seconds

        # Hand detection - toggle between MediaPipe and classical CV
        self.use_mediapipe = use_mediapipe
        self._init_finger_detector()

        self.show_hand_debug = True
        self.show_tracker_lines = True

        self.finger_map = {
            "index": 1,
            "middle": 2,
            "ring": 3,
            "pinky": 4
        }

    def _init_finger_detector(self):
        """Initialize finger detector based on use_mediapipe flag."""
        if self.use_mediapipe:
            from hand_detector import HandDetector
            self.finger_detector = HandDetector(max_hands=2, detection_conf=0.6, tracking_conf=0.6)
            print("[GuitarSystem] Using MediaPipe hand detection")
        else:
            from fingers_detector_cv import FingersDetectorCV
            self.finger_detector = FingersDetectorCV()
            print("[GuitarSystem] Using classical CV finger detection")

    # =========================================================
    # Song logic
    # =========================================================
    def start_playing_song(self, song_id):
        self.active_song_id = song_id
        self.song_chords_sequence = SONGS.get(song_id, [])
        self.current_chord_index = 0
        self._update_chord_from_sequence()
        # Start countdown instead of playing immediately
        self.countdown_active = True
        self.countdown_start_time = time.time()
        self.is_playing_song = False  # Will be set True after countdown
        print(f"Starting countdown for: {song_id}")

    def stop_playing_song(self):
        self.is_playing_song = False
        self.countdown_active = False
        self.active_song_id = None
        self.song_chords_sequence = []
        self.current_chord_name = None
        self.current_chord_data = None
        self.current_lyric = ""
        self.current_section = ""
        print("Stopped playing song. Chord cleared from screen.")

    def _update_chord_from_sequence(self):
        if self.song_chords_sequence and self.current_chord_index < len(self.song_chords_sequence):
            node = self.song_chords_sequence[self.current_chord_index]
            self.current_chord_name = node["chord"]
            self.current_chord_data = chords_data.get_chord_data(self.current_chord_name)
            self.current_lyric = node.get("lyric", "")
            section = node.get("section", "")
            if section:
                self.current_section = section

    def _reset_calibration_only(self):
        """
        Reset only tracker/calibration-related state.
        Keep hand detector alive and do not recreate the whole system.
        """
        self.tracker = sf.GuitarNeckTracker()
        self.current_chord_name = None
        self.current_chord_data = None

        # Reset finger detector
        self._init_finger_detector()

        # Reset tracker lines to visible
        self.show_tracker_lines = True

        # Stop song playback / overlay
        self.current_lyric = ""
        self.current_section = ""
        self.is_playing_song = False
        self.active_song_id = None
        self.song_chords_sequence = []
        self.current_chord_index = 0
        self.chord_start_time = 0.0
        self.countdown_active = False

        print("Calibration reset.")

    def _reset_all(self):
        """
        Full reset.
        """
        self.tracker = sf.GuitarNeckTracker()

        # Reset finger detector
        self._init_finger_detector()

        self.chord_idx = 0
        self.current_chord_name = None
        self.current_chord_data = None

        self.song_chords_sequence = []
        self.current_chord_index = 0
        self.chord_start_time = 0.0

        # Reset tracker lines to visible
        self.show_tracker_lines = True

        self.current_lyric = ""
        self.current_section = ""
        self.is_playing_song = False
        self.active_song_id = None
        self.song_chords_sequence = []
        self.current_chord_index = 0
        self.chord_start_time = 0.0
        self.countdown_active = False

        print("Reset All.")

    # =========================================================
    # Hand + chord validation
    # =========================================================
    def _parse_chord_finger_info(self, finger_info):
        """
        Chords.py Version 1 format:
        (fret_number, string_number, finger_id)
        """
        if len(finger_info) == 2:
            fret_num, string_num = finger_info
            req_finger_id = None
        elif len(finger_info) == 3:
            fret_num, string_num, req_finger_id = finger_info
        else:
            return None, None, None

        return fret_num, string_num, req_finger_id

    def _choose_fretting_hand(self, all_hands):
        """
        Choose the hand whose fingertips are closest to / inside the fretboard area.
        """
        if self.tracker.refined_quad is None or not all_hands:
            return None

        quad = self.tracker.refined_quad
        x_min = int(min(quad[:, 0]))
        x_max = int(max(quad[:, 0]))
        y_min = int(min(quad[:, 1]))
        y_max = int(max(quad[:, 1]))

        # Slight padding around fretboard
        pad_x = 50
        pad_y = 50
        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        best_hand = None
        best_score = -1
        best_avg_dist = float("inf")

        for hand in all_hands:
            tips = hand.get("tips", [])
            if not tips:
                continue

            score = 0
            dist_sum = 0.0

            for _, x, y in tips:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    score += 1

                # distance to expanded fretboard bbox
                dx = 0 if x_min <= x <= x_max else min(abs(x - x_min), abs(x - x_max))
                dy = 0 if y_min <= y <= y_max else min(abs(y - y_min), abs(y - y_max))
                dist_sum += (dx * dx + dy * dy) ** 0.5

            avg_dist = dist_sum / max(1, len(tips))

            if score > best_score or (score == best_score and avg_dist < best_avg_dist):
                best_score = score
                best_avg_dist = avg_dist
                best_hand = hand

        return best_hand

    def _draw_chord_feedback_with_hand(self, raw_frame, display_frame):
        if (
            self.current_chord_data is None
            or not self.tracker.is_strings_calibrated
            or len(self.tracker.fret_model_rel) < 2
            or self.tracker.refined_quad is None
        ):
            return display_frame

        if self.use_mediapipe:
            return self._draw_chord_feedback_mediapipe(raw_frame, display_frame)
        else:
            return self._draw_chord_feedback_cv(raw_frame, display_frame)

    def _draw_chord_feedback_mediapipe(self, raw_frame, display_frame):
        """MediaPipe-based chord feedback (original approach)."""
        all_hands = self.finger_detector.detect_all_hands_fingertips(raw_frame)
        if not all_hands:
            return display_frame

        chosen_hand = self._choose_fretting_hand(all_hands)
        if chosen_hand is None:
            return display_frame

        tips = chosen_hand["tips"]

        if self.show_hand_debug:
            self.finger_detector.last_landmarks_px = chosen_hand["landmarks"]
            self.finger_detector.last_fingertips_px = chosen_hand["tips"]
            display_frame = self.finger_detector.draw_debug(display_frame)

        all_fingers_ok = True

        for finger_info in self.current_chord_data.get("fingers", []):
            fret_num, string_num, req_finger_id = self._parse_chord_finger_info(finger_info)

            if fret_num is None or string_num is None:
                all_fingers_ok = False
                continue

            x, y = self.tracker.get_dot_coordinates(fret_num, string_num, mirror_frets=True)

            if x < 0 or y < 0:
                all_fingers_ok = False
                continue

            is_pressed = False
            for name, fx, fy in tips:
                if req_finger_id is not None and self.finger_map.get(name) != req_finger_id:
                    continue
                dist = ((x - fx) ** 2 + (y - fy) ** 2) ** 0.5
                if dist < 15:
                    is_pressed = True
                    break

            dot_color = (0, 255, 0) if is_pressed else (0, 0, 255)
            if not is_pressed:
                all_fingers_ok = False

            cv2.circle(display_frame, (x, y), 6, dot_color, -1)
            cv2.circle(display_frame, (x, y), 7, (255, 255, 255), 1)

            if req_finger_id is not None:
                cv2.putText(display_frame, str(req_finger_id), (x - 4, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        msg = "CHORD PERFECT!" if all_fingers_ok else "Check positions"
        status_color = (0, 255, 0) if all_fingers_ok else (0, 165, 255)
        cv2.putText(display_frame, msg, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        return display_frame

    def _draw_chord_feedback_cv(self, raw_frame, display_frame):
        """Classical CV position-based chord feedback."""
        # Reset frame state in detector
        self.finger_detector.reset_frame()

        # Tell detector which chord we're checking (resets states on chord change)
        self.finger_detector.set_current_chord(self.current_chord_name)

        all_fingers_ok = True

        for finger_info in self.current_chord_data.get("fingers", []):
            fret_num, string_num, req_finger_id = self._parse_chord_finger_info(finger_info)

            if fret_num is None or string_num is None:
                all_fingers_ok = False
                continue

            x, y = self.tracker.get_dot_coordinates(fret_num, string_num, mirror_frets=True)

            if x < 0 or y < 0:
                all_fingers_ok = False
                continue

            # Check if finger is present at this position (with temporal persistence)
            is_pressed = self.finger_detector.check_position(raw_frame, x, y, fret_num, string_num)

            dot_color = (0, 255, 0) if is_pressed else (0, 0, 255)
            if not is_pressed:
                all_fingers_ok = False

            cv2.circle(display_frame, (x, y), 6, dot_color, -1)
            cv2.circle(display_frame, (x, y), 7, (255, 255, 255), 1)

            if req_finger_id is not None:
                cv2.putText(display_frame, str(req_finger_id), (x - 4, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        # Draw debug visualization if enabled
        if self.show_hand_debug:
            display_frame = self.finger_detector.draw_debug(display_frame)

        msg = "CHORD PERFECT!" if all_fingers_ok else "Check positions"
        status_color = (0, 255, 0) if all_fingers_ok else (0, 165, 255)
        cv2.putText(display_frame, msg, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        return display_frame

    def _draw_chord_targets_only(self, display_frame):
        """
        Draw chord dots only, without hand detection / validation.
        Used during song-playing mode.
        """
        if (
            self.current_chord_data is None
            or not self.tracker.is_strings_calibrated
            or len(self.tracker.fret_model_rel) < 2
            or self.tracker.refined_quad is None
        ):
            return display_frame

        for finger_info in self.current_chord_data.get("fingers", []):
            req_finger_id = None

            if len(finger_info) == 2:
                fret_num, string_num = finger_info
            elif len(finger_info) == 3:
                fret_num, string_num, req_finger_id = finger_info
            else:
                continue

            x, y = self.tracker.get_dot_coordinates(
                fret_num,
                string_num,
                mirror_frets=True
            )

            if x < 0 or y < 0:
                continue

            cv2.circle(display_frame, (x, y), 6, (0, 255, 255), -1)
            cv2.circle(display_frame, (x, y), 7, (255, 255, 255), 1)

            if req_finger_id is not None:
                cv2.putText(
                    display_frame,
                    str(req_finger_id),
                    (x - 4, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (0, 0, 0),
                    1
                )

        return display_frame

    # =========================================================
    # Main frame processing
    # =========================================================
    def process_frame(self, frame, key):
        height, width = frame.shape[:2]

        just_locked = False

        # Show hand debug only in chord-training mode
        self.show_hand_debug = not self.is_playing_song
        

        # -----------------------------
        # Commands
        # -----------------------------
        if not self.is_playing_song:
            if key == ord('r'):
                self._reset_calibration_only()

            if key == ord('x'):
                self.tracker.clear_strings()
                print("Strings cleared. Press [s] to calibrate again.")

            if key == ord('a'):
                self.chord_idx = (self.chord_idx + 1) % len(self.available_chords)
                self.current_chord_name = self.available_chords[self.chord_idx]
                self.current_chord_data = chords_data.get_chord_data(self.current_chord_name)
                print("Chord selected:", self.current_chord_name)

            if key == ord('z'):
                self.chord_idx = (self.chord_idx - 1) % len(self.available_chords)
                self.current_chord_name = self.available_chords[self.chord_idx]
                self.current_chord_data = chords_data.get_chord_data(self.current_chord_name)
                print("Chord selected:", self.current_chord_name)

            if key == ord('v'):
                self.current_chord_name = None
                self.current_chord_data = None
                print("Chord cleared.")

            if key == ord('c'):
                self.tracker.lock(frame)
                just_locked = True

        # -----------------------------
        # Update tracker every frame
        # BUT do not update on the same frame we just locked
        # -----------------------------
        if not just_locked:
            self.tracker.update(frame)

        # Periodically re-detect frets to self-correct a weak initial lock
        # Skip during song playback — fret model is already calibrated and
        # re-detection during motion can introduce spurious peaks.
        if (not self.is_playing_song
                and self.tracker.is_tracking
                and self.tracker.tracking_ok
                and self.tracker.refined_quad is not None):
            self.tracker.maybe_refine_frets(frame)

        # -----------------------------
        # Base display
        # -----------------------------
        if self.is_playing_song or self.countdown_active or not self.show_tracker_lines:
            # Plain camera frame, no fretboard overlay
            display_frame = frame.copy()
        else:
            # Learning / calibration mode: show tracker debug
            display_frame = self.tracker.draw_debug(frame.copy())

        # -----------------------------
        # Countdown before song starts
        # -----------------------------
        if self.countdown_active:
            elapsed = time.time() - self.countdown_start_time
            remaining = self.countdown_duration - elapsed
            if remaining <= 0:
                # Countdown finished, start song
                self.countdown_active = False
                self.is_playing_song = True
                self.chord_start_time = time.time()
                print(f"Started playing: {self.active_song_id}")
            else:
                # Draw countdown number (big, centered)
                countdown_num = int(remaining) + 1
                text = str(countdown_num)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 5.0
                thickness = 8
                (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                h, w = display_frame.shape[:2]
                x = (w - text_w) // 2
                y = (h + text_h) // 2
                cv2.putText(display_frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)

        # -----------------------------
        # Song timing
        # -----------------------------
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

        # -----------------------------
        # String calibration
        # -----------------------------
        if key == ord('s') and not self.is_playing_song and self.tracker.is_tracking:
            self.tracker.calibrate_strings(frame)
            # Learn fretboard appearance for classical CV finger detection
            if not self.use_mediapipe and self.tracker.refined_quad is not None:
                self.finger_detector.learn_fretboard(frame, self.tracker.refined_quad)

        # -----------------------------
        # Draw chord dots + hand feedback
        # -----------------------------
        if (
            self.current_chord_data is not None
            and self.tracker.is_strings_calibrated
            and len(self.tracker.fret_model_rel) >= 2
            and self.tracker.refined_quad is not None
            and self.tracker.tracking_ok
            and not self.countdown_active
        ):
            if self.is_playing_song:
                display_frame = self._draw_chord_targets_only(display_frame)
            else:
                display_frame = self._draw_chord_feedback_with_hand(frame, display_frame)

        # -----------------------------
        # HUD
        # -----------------------------
        if not self.is_playing_song and not self.countdown_active:
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

            if not self.tracker.is_tracking:
                cv2.putText(
                    display_frame,
                    "Preview: press [c] to lock fretboard from ROI",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if self.tracker.is_tracking and not self.tracker.is_strings_calibrated:
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

        return display_frame