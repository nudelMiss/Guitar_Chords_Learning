import cv2
import numpy as np


class ForegroundMaskCV:
    def __init__(self):
        # ROI: tune this to your scene (rx, ry, rw, rh) in relative coords
        self.roi_rel = (0.55, 0.15, 0.42, 0.70)

        # Background model (ROI grayscale)
        self._bg_gray = None
        self._bg_calibrated = False

        # Debug / output
        self.last_mask = None

    # ---------------- ROI ----------------
    def _extract_roi(self, frame):
        H, W = frame.shape[:2]
        rx, ry, rw, rh = self.roi_rel
        x0 = int(rx * W)
        y0 = int(ry * H)
        w = int(rw * W)
        h = int(rh * H)
        roi = frame[y0:y0 + h, x0:x0 + w]
        return roi, (x0, y0, w, h)

    # ---------------- Background calibration ----------------
    def calibrate_background(self, frame) -> bool:
        """
        Learn background in ROI (ROI must be empty).
        """
        roi, _ = self._extract_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        self._bg_gray = gray
        self._bg_calibrated = True

        print("Background calibration: SUCCESS (keep ROI empty when calibrating)")
        return True

    # ---------------- Foreground mask (background diff) ----------------
    def make_mask(self, frame):
        """
        Returns the foreground mask in ROI using EXACT same logic:
        gray+blur -> absdiff vs bg -> threshold(25) -> MORPH_OPEN/CLOSE with ellipse(7,7), iter=2
        """
        if not self._bg_calibrated or self._bg_gray is None:
            self.last_mask = None
            return None

        roi, _ = self._extract_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        diff = cv2.absdiff(self._bg_gray, gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        self.last_mask = mask
        return mask


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = ForegroundMaskCV()

    print("INSTRUCTIONS:")
    print("1) Press 'b' to calibrate BACKGROUND (ROI must be empty).")
    print("2) Move your hand/object inside ROI to see the MASK.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI rectangle (same as original style)
        H, W = frame.shape[:2]
        rx, ry, rw, rh = detector.roi_rel
        x0 = int(rx * W)
        y0 = int(ry * H)
        x1 = int((rx + rw) * W)
        y1 = int((ry + rh) * H)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

        # Status
        status = f"BG={'ON' if detector._bg_calibrated else 'OFF'}"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Make/show mask
        detector.make_mask(frame)

        cv2.imshow("VIDEO", frame)
        if detector.last_mask is not None:
            cv2.imshow("MASK", detector.last_mask)
        else:
            # Show a black window until background is calibrated
            blank = np.zeros((max(1, y1 - y0), max(1, x1 - x0)), dtype=np.uint8)
            cv2.imshow("MASK", blank)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("b"):
            detector.calibrate_background(frame)
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()