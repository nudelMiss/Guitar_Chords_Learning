"""
Classical CV finger detection for guitar chord learning.
Uses position-targeted verification with finger-specific discrimination.

Key insight: We KNOW where fingers should be (chord positions).
We verify presence AND check that it looks like a fingertip (not an arm).

No MediaPipe, no deep learning.
"""

import cv2
import numpy as np


class PositionState:
    """
    Tracks whether a chord position is pressed with temporal persistence.
    Requires N consecutive frames to change state (prevents flicker).
    """
    def __init__(self, required_frames=4):
        self.is_pressed = False
        self.consecutive_pressed = 0
        self.consecutive_not_pressed = 0
        self.required = required_frames

    def update(self, detected_now: bool) -> bool:
        """Update state and return current stable is_pressed value."""
        if detected_now:
            self.consecutive_pressed += 1
            self.consecutive_not_pressed = 0
            if self.consecutive_pressed >= self.required:
                self.is_pressed = True
        else:
            self.consecutive_not_pressed += 1
            self.consecutive_pressed = 0
            if self.consecutive_not_pressed >= self.required:
                self.is_pressed = False

        return self.is_pressed

    def reset(self):
        """Reset state when chord changes."""
        self.is_pressed = False
        self.consecutive_pressed = 0
        self.consecutive_not_pressed = 0


class FingersDetectorCV:
    """
    Position-based finger verification for guitar chord learning.

    Distinguishes fingers from arms by checking:
    1. Skin color presence
    2. Blob compactness (fingers are small and round, arms are elongated)
    3. Blob is contained within check area (not extending through it)
    """

    # Skin detection in YCrCb (tuned for typical skin tones)
    SKIN_CR_MIN, SKIN_CR_MAX = 135, 175
    SKIN_CB_MIN, SKIN_CB_MAX = 80, 125

    # Patch size around each chord position to check
    PATCH_RADIUS = 15

    # Minimum skin ratio in patch to trigger further analysis
    MIN_SKIN_RATIO = 0.15

    # Maximum skin ratio - if too high, it's probably an arm filling the patch
    MAX_SKIN_RATIO = 0.85

    # Blob compactness threshold (circularity: 4*pi*area/perimeter^2)
    # Fingers are rounder (higher), arms are elongated (lower)
    MIN_COMPACTNESS = 0.25

    # Minimum blob area as fraction of patch
    MIN_BLOB_AREA_RATIO = 0.08

    # Maximum blob area as fraction of patch (if too big, it's an arm)
    MAX_BLOB_AREA_RATIO = 0.70

    # Frames required to change state (prevents flicker)
    PERSISTENCE_FRAMES = 4

    def __init__(self):
        # Fretboard color model (optional, for better discrimination)
        self.fretboard_mean_lab = None
        self.fretboard_std_lab = None
        self.is_learned = False

        # Position states for temporal persistence (keyed by (fret, string))
        self._position_states = {}

        # Debug info
        self.last_checks = []  # List of (x, y, result_info)

        # Current chord tracking (to reset states when chord changes)
        self._current_chord_key = None

    def learn_fretboard(self, frame, quad):
        """
        Learn the fretboard appearance for better skin discrimination.
        Call during string calibration when fretboard is clear.
        """
        if quad is None:
            print("[FingersDetectorCV] Cannot learn: no quad provided")
            return False

        # Warp fretboard to canonical rectangle
        src = np.array(quad, dtype=np.float32)
        dst = np.array([[0, 0], [200, 0], [200, 60], [0, 60]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(frame, M, (200, 60))

        # Learn LAB color distribution
        lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.fretboard_mean_lab = np.mean(lab, axis=(0, 1))
        self.fretboard_std_lab = np.std(lab, axis=(0, 1)) + 1e-6

        self.is_learned = True
        print(f"[FingersDetectorCV] Learned fretboard color")
        return True

    def set_current_chord(self, chord_name):
        """Call when chord changes to reset position states."""
        if chord_name != self._current_chord_key:
            self._current_chord_key = chord_name
            # Reset all position states for new chord
            for state in self._position_states.values():
                state.reset()

    def check_position(self, frame, x, y, fret_num, string_num):
        """
        Check if a FINGER (not arm) is present at position (x, y).

        Returns:
            bool: True if finger is stably detected at this position
        """
        h, w = frame.shape[:2]
        r = self.PATCH_RADIUS

        # Bounds check
        x1 = max(0, x - r)
        x2 = min(w, x + r)
        y1 = max(0, y - r)
        y2 = min(h, y + r)

        if x2 <= x1 or y2 <= y1:
            return False

        # Extract patch
        patch = frame[y1:y2, x1:x2]

        # Analyze the patch for a finger-like blob
        is_finger, info = self._is_finger_present(patch)

        # Get or create position state
        key = (fret_num, string_num)
        if key not in self._position_states:
            self._position_states[key] = PositionState(self.PERSISTENCE_FRAMES)

        # Apply persistence and get stable state
        is_pressed = self._position_states[key].update(is_finger)

        # Store debug info
        self.last_checks.append((x, y, is_pressed, is_finger, info))

        return is_pressed

    def _is_finger_present(self, patch):
        """
        Analyze patch to determine if a finger (not arm) is present.

        Returns:
            (bool, dict): (is_finger, debug_info)
        """
        info = {"skin_ratio": 0, "blob_area_ratio": 0, "compactness": 0, "reason": ""}

        if patch.size == 0:
            info["reason"] = "empty"
            return False, info

        patch_area = patch.shape[0] * patch.shape[1]

        # Step 1: Get skin mask
        skin_mask = self._get_skin_mask(patch)
        skin_pixels = np.count_nonzero(skin_mask)
        skin_ratio = skin_pixels / patch_area

        info["skin_ratio"] = skin_ratio

        # Too little skin = no finger
        if skin_ratio < self.MIN_SKIN_RATIO:
            info["reason"] = "low_skin"
            return False, info

        # Too much skin = probably arm filling the patch
        if skin_ratio > self.MAX_SKIN_RATIO:
            info["reason"] = "high_skin(arm?)"
            return False, info

        # Step 2: Find the largest blob and analyze its shape
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            info["reason"] = "no_contours"
            return False, info

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        blob_area = cv2.contourArea(largest)
        blob_area_ratio = blob_area / patch_area

        info["blob_area_ratio"] = blob_area_ratio

        # Blob too small = noise
        if blob_area_ratio < self.MIN_BLOB_AREA_RATIO:
            info["reason"] = "small_blob"
            return False, info

        # Blob too big = arm
        if blob_area_ratio > self.MAX_BLOB_AREA_RATIO:
            info["reason"] = "big_blob(arm?)"
            return False, info

        # Step 3: Check blob compactness (circularity)
        perimeter = cv2.arcLength(largest, True)
        if perimeter > 0:
            compactness = 4 * np.pi * blob_area / (perimeter * perimeter)
        else:
            compactness = 0

        info["compactness"] = compactness

        # Fingers are compact/round, arms are elongated
        if compactness < self.MIN_COMPACTNESS:
            info["reason"] = "elongated(arm?)"
            return False, info

        # Step 4: Check if blob touches edges (arm passing through)
        blob_mask = np.zeros(skin_mask.shape, dtype=np.uint8)
        cv2.drawContours(blob_mask, [largest], -1, 255, -1)

        # Check if blob touches patch boundary significantly
        edge_pixels = (
            np.count_nonzero(blob_mask[0, :]) +      # top edge
            np.count_nonzero(blob_mask[-1, :]) +     # bottom edge
            np.count_nonzero(blob_mask[:, 0]) +      # left edge
            np.count_nonzero(blob_mask[:, -1])       # right edge
        )
        edge_ratio = edge_pixels / (2 * (patch.shape[0] + patch.shape[1]))

        # If blob extends through patch (touches multiple edges heavily), it's likely an arm
        if edge_ratio > 0.5:
            info["reason"] = "extends_through(arm?)"
            return False, info

        # Passed all checks - likely a finger!
        info["reason"] = "finger"
        return True, info

    def _get_skin_mask(self, patch):
        """Get skin color mask, optionally filtering out fretboard color."""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(patch, cv2.COLOR_BGR2YCrCb)

        # Skin mask
        skin_mask = cv2.inRange(
            ycrcb,
            (0, self.SKIN_CR_MIN, self.SKIN_CB_MIN),
            (255, self.SKIN_CR_MAX, self.SKIN_CB_MAX)
        )

        # If we have fretboard model, also require "different from fretboard"
        if self.is_learned:
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
            diff = (lab - self.fretboard_mean_lab) / self.fretboard_std_lab
            dist = np.linalg.norm(diff, axis=2)
            fretboard_diff_mask = (dist > 1.5).astype(np.uint8) * 255

            # Combine: must be skin AND different from fretboard
            skin_mask = cv2.bitwise_and(skin_mask, fretboard_diff_mask)

        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        return skin_mask

    def reset_frame(self):
        """Call at start of each frame to clear debug info."""
        self.last_checks = []

    def draw_debug(self, frame):
        """Draw minimal debug visualization - only show results, not distracting boxes."""
        out = frame.copy()

        for (x, y, is_pressed, raw_detected, info) in self.last_checks:
            # Only draw a small indicator, not full rectangles
            if is_pressed:
                # Small green dot for confirmed press
                cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
            elif raw_detected:
                # Small yellow dot for raw detection (waiting for persistence)
                cv2.circle(out, (x, y), 2, (0, 255, 255), -1)
            # Don't draw anything for non-detections to keep display clean

        return out

    def get_detection_info(self):
        """Return info string for HUD."""
        n_pressed = sum(1 for _, _, pressed, _, _ in self.last_checks if pressed)
        n_total = len(self.last_checks)
        learned = "Yes" if self.is_learned else "No"
        return f"CV: {n_pressed}/{n_total}, FB: {learned}"

    # =========================================================================
    # Legacy interface (for compatibility)
    # =========================================================================

    def detect_fingertip_candidates(self, frame, quad, string_ys_rel=None):
        """Legacy interface - not used in new approach."""
        self.reset_frame()
        return []
