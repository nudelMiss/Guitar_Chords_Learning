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
    #blurred = cv2.GaussianBlur(gray, (3,3), 0)

    # --- Vertical gradient (Sobel X) ---
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
    bottom_fraction = 0.5  # search only in bottom 50% of the frame

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

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()