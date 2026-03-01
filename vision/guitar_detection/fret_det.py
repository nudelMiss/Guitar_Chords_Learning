import cv2
import numpy as np

def detect_frets_bottom(frame, brightness_thresh=120, bottom_fraction=0.5):
    """
    Detect strong vertical frets only in the bottom part of the frame.
    brightness_thresh: minimum gradient to detect a line
    bottom_fraction: fraction of the image height to search from the bottom
    Returns a list of vertical lines [x1, y1, x2, y2]
    """

    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))  # start of the bottom part

    # Crop bottom part of the frame
    bottom_frame = frame[start_row:, :]

    # --- Grayscale and blur ---
    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (3,3), 0)l X) ---
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.convertScaleAbs(sobelx)

    # --- Threshold to detect strong vertical transitions ---
    _, thresh = cv2.threshold(sobelx, brightness_thresh, 255, cv2.THRESH_BINARY)

    # Morphological closing to connect broken vertical segments
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- Hough transform for vertical lines ---
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=15,
        maxLineGap=5
    )

    # --- Vertical lines filtering ---
    vertical_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            # Mostly vertical
            if abs(dx) < 5 and dy > 10:
                # Adjust y coordinates to original frame
                vertical_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    vertical_lines.sort(key=lambda l: l[0])
    return vertical_lines


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    brightness_thresh = 120
    bottom_fraction = 0.4  # search only in bottom 50% of the frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frets = detect_frets_bottom(frame, brightness_thresh, bottom_fraction)

        # Draw detected vertical frets
        for line in frets:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)

        cv2.imshow("Vertical Frets (Bottom)", frame)

        # --- Key handling ---
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to perform detection loop
        if key == ord('c'):
            stored_lines = []
            i = 0
            while i < 10:
                ret, frame = cap.read()
                if not ret:
                    break
                frets = detect_frets_bottom(frame, brightness_thresh, 0.3)
                stored_lines += frets
                print(f"Detected {len(stored_lines)} frets")
                i += 1
            stored_lines = merge_vertical_lines(stored_lines)

        # Press 'q' to quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def merge_vertical_lines(lines, x_threshold=5, y_threshold=10):
    """
    Merge vertical lines that are close to each other in X and Y.
    lines: list of [x1, y1, x2, y2]
    x_threshold: max horizontal distance to consider lines part of the same group
    y_threshold: max vertical overlap to consider lines part of the same group
    Returns: list of merged lines [{"x": avg_x, "y1": min_y, "y2": max_y}]
    """

    if not lines:
        return []

    # Sort lines by X coordinate
    lines.sort(key=lambda l: l[0])

    merged = []
    current_group = [lines[0]]

    for l in lines[1:]:
        # Compare X distance with last line in group
        if abs(l[0] - current_group[-1][0]) <= x_threshold:
            # Check vertical overlap
            last_y1 = min(l[1] for l in current_group)
            last_y2 = max(l[3] for l in current_group)
            # If any overlap vertically (or close)
            if l[3] + y_threshold >= last_y1 and l[1] - y_threshold <= last_y2:
                current_group.append(l)
            else:
                # Merge current group into one line
                min_y = min(line[1] for line in current_group)
                max_y = max(line[3] for line in current_group)
                avg_x = int(np.mean([line[0] for line in current_group]))
                merged.append({"x": avg_x, "y1": min_y, "y2": max_y})
                current_group = [l]
        else:
            # Merge current group into one line
            min_y = min(line[1] for line in current_group)
            max_y = max(line[3] for line in current_group)
            avg_x = int(np.mean([line[0] for line in current_group]))
            merged.append({"x": avg_x, "y1": min_y, "y2": max_y})
            current_group = [l]

    # Merge last group
    min_y = min(line[1] for line in current_group)
    max_y = max(line[3] for line in current_group)
    avg_x = int(np.mean([line[0] for line in current_group]))
    merged.append({"x": avg_x, "y1": min_y, "y2": max_y})

    return merged


if __name__ == "__main__":
    main()