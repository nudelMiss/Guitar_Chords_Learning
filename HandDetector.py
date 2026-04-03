"""
Provides hand landmark detection using MediaPipe.
Identifies the location of fingertips and applies EMA filtering 
to reduce jitter in coordinates for accurate chord validation.
Input: Video frames.
Output: Pixel coordinates of fingertips (Index, Middle, Ring, Pinky). """

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

FINGERTIP_IDS = {
    # "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

# Hand connection pairs for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (17, 13), (13, 9), (9, 5),
]

class _EMA:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.state = None

    def __call__(self, xy):
        xy = np.array(xy, dtype=np.float32)
        if self.state is None:
            self.state = xy
        else:
            self.state = self.alpha * xy + (1 - self.alpha) * self.state
        return self.state


class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.6, tracking_conf=0.6):
        # path to mediapipe model
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, "models", "hand_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=tracking_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self._ema = {name: _EMA(alpha=0.6) for name in FINGERTIP_IDS}

        self.last_landmarks_px = None
        self.last_fingertips_px = None

    def detect_fingers_and_frets(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Detect hand landmarks
        results = self.hand_landmarker.detect(mp_image)

        self.last_landmarks_px = None
        self.last_fingertips_px = None

        if not results.hand_landmarks:
            return 0, []

        hand_lms = results.hand_landmarks[0]

        pts = []
        for lm in hand_lms:
            pts.append((int(lm.x * w), int(lm.y * h)))
        self.last_landmarks_px = pts

        tips = []
        for name, idx in FINGERTIP_IDS.items():
            x, y = pts[idx]
            x_s, y_s = self._ema[name]([x, y])
            tips.append((name, int(x_s), int(y_s)))

        self.last_fingertips_px = tips
        return len(tips), tips

    def draw_debug(self, frame_bgr):
        if self.last_landmarks_px is None:
            return frame_bgr

        out = frame_bgr.copy()

        # draw connections
        for a, b in HAND_CONNECTIONS:
            xa, ya = self.last_landmarks_px[a]
            xb, yb = self.last_landmarks_px[b]
            cv2.line(out, (xa, ya), (xb, yb), (255, 255, 255), 1)

        # draw points
        for (x, y) in self.last_landmarks_px:
            cv2.circle(out, (x, y), 2, (255, 255, 255), -1)

        # draw fingertips
        if self.last_fingertips_px:
            for (name, x, y) in self.last_fingertips_px:
                cv2.circle(out, (x, y), 4, (255, 0, 0), -1)

        return out

    def detect_all_hands_fingertips(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_landmarker.detect(mp_image)

        all_hands = []

        if not results.hand_landmarks:
            self.last_landmarks_px = None
            self.last_fingertips_px = None
            return []

        for hand_lms in results.hand_landmarks:
            pts = []
            for lm in hand_lms:
                pts.append((int(lm.x * w), int(lm.y * h)))

            tips = []
            for name, idx in FINGERTIP_IDS.items():
                x, y = pts[idx]
                # no EMA per-hand here for now
                tips.append((name, int(x), int(y)))

            all_hands.append({
                "landmarks": pts,
                "tips": tips
            })

        return all_hands