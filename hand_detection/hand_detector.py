import cv2
import mediapipe as mp
import numpy as np

FINGERTIP_IDS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._ema = {name: _EMA(alpha=0.6) for name in FINGERTIP_IDS}

        self.last_landmarks_px = None
        self.last_fingertips_px = None

    def detect_fingers_and_frets(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        self.last_landmarks_px = None
        self.last_fingertips_px = None

        if not res.multi_hand_landmarks:
            return 0, []

        hand_lms = res.multi_hand_landmarks[0]

        pts = []
        for lm in hand_lms.landmark:
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
        for a, b in self.mp_hands.HAND_CONNECTIONS:
            xa, ya = self.last_landmarks_px[a]
            xb, yb = self.last_landmarks_px[b]
            cv2.line(out, (xa, ya), (xb, yb), (255, 255, 255), 2)

        # draw points
        for (x, y) in self.last_landmarks_px:
            cv2.circle(out, (x, y), 2, (255, 255, 255), -1)

        # draw fingertips
        if self.last_fingertips_px:
            for (name, x, y) in self.last_fingertips_px:
                cv2.circle(out, (x, y), 8, (255, 0, 0), -1)
                cv2.putText(out, name, (x + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return out
