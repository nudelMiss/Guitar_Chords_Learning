import cv2
import numpy as np

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
            if abs(dx) < 5 and dy > 5:  # mostly vertical
                vertical_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    # Sort by X position
    vertical_lines.sort(key=lambda l: l[0])
    return vertical_lines

def merge_vertical_lines(lines, x_threshold=5):
    """
    Merge vertical lines that are close in X (overlapping or nearby).
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
            merged.append([avg_x, min_y, avg_x, max_y])
            current_group = [line]

    # merge last group
    min_y = min(l[1] for l in current_group)
    max_y = max(l[3] for l in current_group)
    avg_x = int(np.mean([l[0] for l in current_group]))
    merged.append([avg_x, min_y, avg_x, max_y])

    return merged

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    brightness_thresh = 120
    bottom_fraction = 0.5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lines = detect_frets_bottom(frame, brightness_thresh, bottom_fraction)
        merged_lines = merge_vertical_lines(lines, x_threshold=5)

        # Draw representative vertical lines
        for line in merged_lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imshow("Vertical Frets (Merged)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
