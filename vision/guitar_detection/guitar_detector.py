import cv2
import numpy as np

def detect_strings_bottom(frame, brightness_thresh=80, bottom_fraction=0.5):
    """
    Detect strong horizontal strings only in the bottom part of the frame.
    brightness_thresh: minimum gradient to detect a line
    bottom_fraction: fraction of the image height to search from the bottom
    Returns a list of horizontal lines [x1, y1, x2, y2]
    """

    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))  # start of the bottom part

    # Crop bottom part of the frame
    bottom_frame = frame[start_row:, :]

    # --- Grayscale and blur ---
    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # --- Horizontal gradient (Sobel Y) ---
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)

    # --- Threshold to detect strong horizontal transitions ---
    _, thresh = cv2.threshold(sobely, brightness_thresh, 255, cv2.THRESH_BINARY)

    # Morphological closing to connect broken horizontal segments
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- Hough transform for horizontal lines ---
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi/180,
        threshold=30,
        minLineLength=50,
        maxLineGap=5
    )

    horizontal_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            # Mostly horizontal
            if abs(dy) < 5 and dx > 20:
                # Adjust y coordinates to original frame
                horizontal_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    # Sort by Y (top to bottom)
    horizontal_lines.sort(key=lambda l: l[1])
    return horizontal_lines

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    brightness_thresh = 80
    bottom_fraction = 0.5  # search only in bottom 50% of the frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect horizontal strings
        strings = detect_strings_bottom(frame, brightness_thresh, bottom_fraction)

        # Draw detected strings in green
        for line in strings:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # Show the frame
        cv2.imshow("Detected Strings (Bottom)", frame)

        # Exit on 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()