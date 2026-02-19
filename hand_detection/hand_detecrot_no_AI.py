import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass(frozen=True)
class FingerObservation:
    finger: str
    px: int
    py: int
    grid_x: float
    grid_y: float
    fret_index: Optional[int]
    string_index: Optional[int]


class HandDetectorCV:
    def __init__(self):
        # ROI: tune this to your scene
        self.roi_rel = (0.55, 0.15, 0.42, 0.70)

        self._frets_x = None
        self._strings_y = None

        # Calibration flags
        self._hand_calibrated = False
        self._bg_calibrated = False

        # Background model (ROI grayscale)
        self._bg_gray = None

        # Thresholds / tuning
        self._min_area = 2000.0
        self.max_fingers = 5
        self._min_tip_dist_px = 35
        self._tip_above_center_margin = 15

        # Smoothing / anti-jitter
        self._ema_alpha = 0.18          # smaller = more smoothing
        self._deadband_px = 6           # if movement < deadband -> keep old
        self._ema_px: Dict[str, Tuple[float, float]] = {}

        # Debug
        self.last_mask = None
        self.last_area = 0.0

    # ---------------- GRID ----------------
    def calibrate_frets(self, frets_x):
        self._frets_x = list(map(int, frets_x)) if frets_x else None

    def calibrate_strings(self, strings_y):
        self._strings_y = list(map(int, strings_y)) if strings_y else None

    # ---------------- ROI ----------------
    def _extract_roi(self, frame):
        H, W = frame.shape[:2]
        rx, ry, rw, rh = self.roi_rel
        x0 = int(rx * W); y0 = int(ry * H)
        w = int(rw * W);  h = int(rh * H)
        roi = frame[y0:y0 + h, x0:x0 + w]
        return roi, (x0, y0, w, h)

    # ---------------- Background calibration ----------------
    def calibrate_background(self, frame) -> bool:
        """
        Learn background in ROI (no hand inside ROI!).
        """
        roi, _ = self._extract_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        self._bg_gray = gray
        self._bg_calibrated = True
        print("Background calibration: SUCCESS (keep ROI empty when calibrating)")
        return True

    # ---------------- Foreground mask (background diff) ----------------
    def _fg_mask_from_bg(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if self._bg_gray is None:
            return None

        diff = cv2.absdiff(self._bg_gray, gray)
        # threshold diff to get moving/changed pixels (hand)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morph cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        return mask

    # ---------------- Contour helpers ----------------
    @staticmethod
    def _largest_contour(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    @staticmethod
    def _contour_centroid(cnt):
        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-6:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy

    # ---------------- Fingertip extraction ----------------
    def _dedupe_points(self, points: List[Tuple[int, int]], min_dist: int) -> List[Tuple[int, int]]:
        out = []
        for p in points:
            if all((p[0]-q[0])**2 + (p[1]-q[1])**2 >= min_dist*min_dist for q in out):
                out.append(p)
        return out

    def _extract_fingertips(self, cnt, center_xy) -> List[Tuple[int, int]]:
        cx, cy = center_xy
        hull_pts = cv2.convexHull(cnt, returnPoints=True).reshape(-1, 2)

        # Candidates: above centroid and far enough from centroid
        candidates = []
        for (x, y) in hull_pts:
            if y < cy - self._tip_above_center_margin:
                dist = np.hypot(x - cx, y - cy)
                candidates.append((int(x), int(y), dist))

        if not candidates:
            return []

        # Sort by "height" (y small first), then by distance
        candidates.sort(key=lambda t: (t[1], -t[2]))
        pts = [(x, y) for (x, y, _) in candidates]
        pts = self._dedupe_points(pts, self._min_tip_dist_px)

        # Keep top N (highest)
        pts = sorted(pts, key=lambda p: p[1])[: self.max_fingers]
        return pts

    def _label_fingers(self, tips_full: List[Tuple[int, int]]) -> List[Tuple[str, Tuple[int, int]]]:
        """
        Heuristic: sort by x and label INDEX..PINKY, thumb is unreliable.
        If 5 tips: label them as FINGER_0.. to avoid wrong naming.
        """
        tips_sorted = sorted(tips_full, key=lambda p: p[0])

        if len(tips_sorted) <= 4:
            names = ["INDEX", "MIDDLE", "RING", "PINKY"]
            return [(names[i], tips_sorted[i]) for i in range(len(tips_sorted))]

        # If 5: don't pretend we know which is thumb
        return [(f"FINGER_{i}", tips_sorted[i]) for i in range(len(tips_sorted))]

    # ---------------- Smoothing with deadband ----------------
    def _smooth(self, key: str, x: float, y: float) -> Tuple[float, float]:
        if key not in self._ema_px:
            self._ema_px[key] = (x, y)
            return x, y

        ox, oy = self._ema_px[key]

        # deadband: if tiny movement, keep old
        if (x - ox) ** 2 + (y - oy) ** 2 < self._deadband_px ** 2:
            return ox, oy

        nx = self._ema_alpha * x + (1 - self._ema_alpha) * ox
        ny = self._ema_alpha * y + (1 - self._ema_alpha) * oy
        self._ema_px[key] = (nx, ny)
        return nx, ny

    # ---------------- Calibration ----------------
    def calibrate_open_hand(self, frame) -> bool:
        """
        This just sets min_area based on detected hand area.
        Requires background calibration for best results.
        """
        roi, _ = self._extract_roi(frame)

        if not self._bg_calibrated:
            print("Calibration failed: background not calibrated yet (press 'b' first).")
            return False

        mask = self._fg_mask_from_bg(roi)
        if mask is None:
            print("Calibration failed: no background model.")
            return False

        self.last_mask = mask
        cnt = self._largest_contour(mask)
        if cnt is None:
            self.last_area = 0.0
            print("Calibration: FAILED (no contour)")
            return False

        area = float(cv2.contourArea(cnt))
        self.last_area = area
        if area < 1500:
            print(f"Calibration: FAILED (area too small: {area:.0f})")
            return False

        self._hand_calibrated = True
        self._min_area = max(1200.0, 0.22 * area)  # dynamic
        self._ema_px.clear()
        print(f"Hand calibration: SUCCESS (area={area:.0f}, min_area={self._min_area:.0f})")
        return True

    # ---------------- Detection ----------------
    def detect_fingers(self, frame) -> List[FingerObservation]:
        if not (self._hand_calibrated and self._bg_calibrated):
            return []

        roi, (x0, y0, _, _) = self._extract_roi(frame)
        mask = self._fg_mask_from_bg(roi)
        if mask is None:
            return []

        self.last_mask = mask
        cnt = self._largest_contour(mask)
        if cnt is None:
            self.last_area = 0.0
            return []

        area = float(cv2.contourArea(cnt))
        self.last_area = area
        if area < self._min_area:
            return []

        centroid = self._contour_centroid(cnt)
        if centroid is None:
            return []
        cx, cy = centroid

        tips_roi = self._extract_fingertips(cnt, (cx, cy))
        tips_full = [(x0 + px, y0 + py) for (px, py) in tips_roi]
        labeled = self._label_fingers(tips_full)

        out = []
        for name, (px, py) in labeled:
            sx, sy = self._smooth(name, float(px), float(py))
            px_i, py_i = int(round(sx)), int(round(sy))

            grid_x, fret = self._map_x_to_frets(sx)
            grid_y, string = self._map_y_to_strings(sy)

            out.append(FingerObservation(
                finger=name,
                px=px_i, py=py_i,
                grid_x=grid_x, grid_y=grid_y,
                fret_index=fret, string_index=string
            ))
        return out

    # ---------------- Grid mapping ----------------
    def _map_x_to_frets(self, x):
        if not self._frets_x or len(self._frets_x) == 0:
            return float("nan"), None
        idx = int(np.argmin([abs(x - fx) for fx in self._frets_x]))
        return float(idx), idx

    def _map_y_to_strings(self, y):
        if not self._strings_y or len(self._strings_y) == 0:
            return float("nan"), None
        idx = int(np.argmin([abs(y - sy) for sy in self._strings_y]))
        return float(idx), idx


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetectorCV()

    # Fake grid (testing only)
    detector.calibrate_frets([300, 350, 400, 450, 500, 550, 600])
    detector.calibrate_strings([200, 230, 260, 290, 320, 350])

    print("INSTRUCTIONS:")
    print("1) Press 'b' to calibrate BACKGROUND (ROI must be empty).")
    print("2) Put open hand in ROI and press 'c' to calibrate HAND.")
    print("3) Then it will detect fingers.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI
        H, W = frame.shape[:2]
        rx, ry, rw, rh = detector.roi_rel
        x0 = int(rx * W); y0 = int(ry * H)
        x1 = int((rx + rw) * W); y1 = int((ry + rh) * H)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Status
        status = f"BG={'ON' if detector._bg_calibrated else 'OFF'} | HAND={'ON' if detector._hand_calibrated else 'OFF'} | area={detector.last_area:.0f}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2)

        if detector._hand_calibrated and detector._bg_calibrated:
            fingers = detector.detect_fingers(frame)
            for f in fingers:
                cv2.circle(frame, (f.px, f.py), 7, (0, 0, 255), -1)
                cv2.putText(frame, f.finger, (f.px + 10, f.py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("VIDEO", frame)
        if detector.last_mask is not None:
            cv2.imshow("MASK", detector.last_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            detector.calibrate_background(frame)
        elif key == ord("c"):
            detector.calibrate_open_hand(frame)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
