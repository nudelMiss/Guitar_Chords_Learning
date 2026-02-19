import cv2
import numpy as np

def detect_frets_bottom(frame, brightness_thresh=120, bottom_fraction=0.5):
    """
    Detect strong vertical frets only in the bottom part of the frame.
    Returns a list of vertical lines [x1, y1, x2, y2]
    """

    height, width = frame.shape[:2]
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
        maxLineGap=5
    )

    vertical_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
                vertical_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    vertical_lines.sort(key=lambda l: l[0])
    return vertical_lines

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    brightness_thresh = 120

    while True:
        ret, frame = cap.read()
        if not ret:
            break


            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0,0,255), 2)


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
