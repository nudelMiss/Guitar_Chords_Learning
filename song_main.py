import cv2
import numpy as np
import Chords
import string_and_frets as sf
import time
from songs_data import SONGS

class GuitarSystem:
    def __init__(self):
        # Initialize all system state variables
        self.is_tracking = False
        self.is_strings_calibrated = False

        self.tracking_pts = None
        self.initial_pts = None
        self.last_gray = None

        self.locked_model = {}
        self.fret_model_rel = []
        self.string_model_rel = []

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

    def start_playing_song(self, song_id):
        # Load the song and start the timer
        self.is_playing_song = True
        self.active_song_id = song_id
        
        # Load sequence from the data file
        self.song_chords_sequence = SONGS.get(song_id, [])
        
        self.current_chord_index = 0
        self.chord_start_time = time.time() # Record the exact start time
        self._update_chord_from_sequence()
        print(f"Started playing: {song_id}")

    def stop_playing_song(self):
        # Reset playing state
        self.is_playing_song = False
        self.active_song_id = None
        self.song_chords_sequence = []
        
        # Clear the current chord so it completely disappears from the screen when stopped
        self.current_chord_name = None
        self.current_chord_data = None
        print("Stopped playing song. Chord cleared from screen.")

    def _update_chord_from_sequence(self):
        if self.song_chords_sequence and self.current_chord_index < len(self.song_chords_sequence):
            node = self.song_chords_sequence[self.current_chord_index]
            self.current_chord_name = node["chord"]
            self.current_chord_data = Chords.get_chord_data(self.current_chord_name)
            # Fetch the lyric from the data, default to empty string
            self.current_lyric = node.get("lyric", "")

    def _current_locked_model_from_tracked_corners(self, tracked_corners):
        # Extract bounding box from tracked corners
        xs = tracked_corners[:, 0]
        ys = tracked_corners[:, 1]
        return {
            "x_min": int(np.min(xs)),
            "x_max": int(np.max(xs)),
            "y_t": int(np.min(ys)),
            "y_b": int(np.max(ys)),
        }

    def get_dot_coordinates(self, fret_num, string_num, tracked_corners, mirror_frets=True):
        # Calculate screen coordinates for a specific fret and string intersection
        if not self.fret_model_rel or not self.string_model_rel:
            return -1, -1
        if string_num < 1 or string_num > 6:
            return -1, -1
        if fret_num < 1 or fret_num > len(self.fret_model_rel):
            return -1, -1

        tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

        # Strings direction
        rel_y = float(self.string_model_rel[6 - string_num])

        # Mirror frets left-right if camera is mirrored
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

    def process_frame(self, frame, key):
        # Main function called by Flask for every frame
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # =========================
        # HANDLE COMMANDS (Only active if not playing a song)
        # =========================
        if not self.is_playing_song:
            if key == ord('r'):
                # Reset All
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

        # =========================
        # TRACKING MODE
        # =========================
        if self.is_tracking and self.tracking_pts is not None and self.initial_pts is not None and self.last_gray is not None:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.last_gray, gray, self.tracking_pts, None)

            if new_pts is not None and status is not None:
                good_new = new_pts[status.flatten() == 1]
                good_old = self.initial_pts[status.flatten() == 1]
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
                    self.tracking_pts = cv2.transform(self.initial_pts, matrix)
                    self.last_gray = gray.copy()
                    
                    corners = np.array(
                        [
                            [self.locked_model["x_min"], self.locked_model["y_t"]],
                            [self.locked_model["x_max"], self.locked_model["y_t"]],
                            [self.locked_model["x_min"], self.locked_model["y_b"]],
                            [self.locked_model["x_max"], self.locked_model["y_b"]],
                        ],
                        dtype=np.float32
                    ).reshape(-1, 1, 2)

                    tracked_corners = cv2.transform(corners, matrix).reshape(-1, 2)
                    tl, tr, bl, br = tracked_corners[0], tracked_corners[1], tracked_corners[2], tracked_corners[3]

                    current_locked_model = self._current_locked_model_from_tracked_corners(tracked_corners)

                    # --- SONG TIMING LOGIC ---
                    if self.is_playing_song and self.song_chords_sequence:
                        current_time = time.time()
                        current_chord_duration = self.song_chords_sequence[self.current_chord_index]["duration"]
                        
                        # Check if it's time to move to the next chord
                        time_elapsed = current_time - self.chord_start_time
                        if time_elapsed >= current_chord_duration:
                            self.current_chord_index += 1
                            
                            # Loop the song or stop when finished
                            if self.current_chord_index >= len(self.song_chords_sequence):
                                self.current_chord_index = 0 
                            
                            # Reset the timer for the new chord
                            self.chord_start_time = time.time()
                            self._update_chord_from_sequence()

                    # Calibrate strings anytime during tracking (only if NOT playing a song)
                    if key == ord('s') and not self.is_playing_song:
                        detected = sf.detect_strings_in_neck(frame, current_locked_model)
                        if len(detected) == 6:
                            self.string_model_rel = detected
                            self.is_strings_calibrated = True
                            print("Strings locked (6 detected).")
                        else:
                            self.string_model_rel = []
                            self.is_strings_calibrated = False
                            print("Strings failed: detected", len(detected), "need 6. Try again [s].")

                    # --- DRAWING LOGIC ---
                    
                    # 1. Draw scaffolding ONLY if we are NOT playing a song
                    if not self.is_playing_song:
                        # Draw frets (green)
                        for rel_x in self.fret_model_rel:
                            p1 = (tl + rel_x * (tr - tl)).astype(int)
                            p2 = (bl + rel_x * (br - bl)).astype(int)
                            cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 0), 2)

                        # Draw strings (yellow)
                        for rel_y in self.string_model_rel:
                            p1 = (tl + rel_y * (bl - tl)).astype(int)
                            p2 = (tr + rel_y * (br - tr)).astype(int)
                            cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 255), 1)

                        # Draw neck boundaries (blue)
                        cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                        cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)

                    # 2. Draw chord dots ALWAYS (if strings are calibrated and chord exists)
                    if self.current_chord_data is not None and self.is_strings_calibrated:
                        
                        # Draw ONLY the fretted fingers with numbers (removed open/muted loops)
                        for finger_num, string_num, fret_num in self.current_chord_data.get("fingers", []):
                            x, y = self.get_dot_coordinates(
                                fret_num, string_num,
                                tracked_corners,
                                mirror_frets=True
                            )
                            if x >= 0 and y >= 0:
                                # Draw filled circle and border
                                cv2.circle(display_frame, (x, y), 12, (70, 180, 120), -1)
                                cv2.circle(display_frame, (x, y), 13, (255, 255, 255), 1)
                                
                                # Draw finger number text inside the circle
                                text = str(finger_num)
                                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                                text_x = x - text_size[0] // 2
                                text_y = y + text_size[1] // 2
                                
                                cv2.putText(display_frame, text, (text_x, text_y), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # 3. HUD Display (Only show text when NOT playing a song)
                    if not self.is_playing_song:
                        if self.current_chord_name:
                            cv2.putText(display_frame, "Chord: {}".format(self.current_chord_name),
                                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                        if not self.is_strings_calibrated:
                            cv2.putText(display_frame, "Press [s] to calibrate strings",
                                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # =========================
        # PREVIEW MODE (When not tracking)
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

                    self.initial_pts = np.array(temp_pts, dtype=np.float32).reshape(-1, 1, 2)
                    self.tracking_pts = self.initial_pts.copy()

                    self.locked_model = {"x_min": x_min, "x_max": x_max, "y_t": y_t, "y_b": y_b}
                    self.fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]

                    self.last_gray = gray.copy()
                    self.is_tracking = True

                    # Reset strings after new lock
                    self.string_model_rel = []
                    self.is_strings_calibrated = False
                    print("Neck locked. Press [s] to calibrate strings.")

            if not self.is_tracking:
                cv2.putText(display_frame, "Preview: press [c] to lock neck",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Return the processed frame to the Flask server
        return display_frame