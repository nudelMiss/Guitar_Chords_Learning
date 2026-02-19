import cv2
import numpy as np

class FretTracker:
    """
    Tracks vertical frets after manual calibration.
    Draws one representative line per fret, stable across frames.
    """
    def __init__(self):
        self.tracked_lines = []  # list of {"x": x-coordinate, "y1": top, "y2": bottom}

    def reset(self, lines):
        """
        Reset tracker with newly detected lines from calibration.
        Uses merged lines as representative.
        """
        self.tracked_lines = self.merge_vertical_lines(lines)

    @staticmethod
    def merge_vertical_lines(lines, x_threshold=5):
        """
        Merge vertical lines that are close in X (overlapping or nearby)
        Returns a list of representative lines [x, min_y, max_y]
        """
        if not lines:
            return []

        # Sort lines by X
        lines.sort(key=lambda l: l[0])

        merged = []
        current_group = [lines[0]]

        for line in lines[1:]:
            if abs(line[0] - current_group[-1][0]) <= x_threshold:
                current_group.append(line)
            else:
                # merge current group
                min_y = min(l[1] for l in current_group)
                max_y = max(l[3] for l in current_group)
                avg_x = int(np.mean([l[0] for l in current_group]))
                merged.append({"x": avg_x, "y1": min_y, "y2": max_y})
                current_group = [line]

        # merge last group
        min_y = min(l[1] for l in current_group)
        max_y = max(l[3] for l in current_group)
        avg_x = int(np.mean([l[0] for l in current_group]))
        merged.append({"x": avg_x, "y1": min_y, "y2": max_y})

        return merged

    def draw(self, frame):
        """
        Draw each tracked fret as a single vertical line
        """
        for line in self.tracked_lines:
            cv2.line(frame, (line["x"], line["y1"]), (line["x"], line["y2"]), (0,0,255), 2)

def detect_frets_bottom(frame, brightness_thresh=120, bottom_fraction=0.5):
    """
    Detect strong vertical frets only in the bottom part of the frame.
    Returns a list of vertical lines [x1, y1, x2, y2]
    """
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    _, thresh = cv2.threshold(sobelx, brightness_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=10,
        maxLineGap=5
    )

    vertical_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 5 and dy > 5:
                vertical_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    vertical_lines.sort(key=lambda l: l[0])
    return vertical_lines

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    tracker = FretTracker()
    brightness_thresh = 120
    bottom_fraction = 0.5
    calibrated = False  # flag to know if calibration was done

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw tracked frets if calibration was done
        if calibrated:
            tracker.draw(frame)

        cv2.imshow("Vertical Frets Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Calibration: detect frets once and store in tracker
            ret, calib_frame = cap.read()
            if ret:
                lines = detect_frets_bottom(calib_frame, brightness_thresh, bottom_fraction)
                tracker.reset(lines)
                calibrated = True  # now tracking is active

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
