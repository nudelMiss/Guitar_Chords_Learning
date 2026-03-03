import cv2
import numpy as np

def detect_strings_bottom(frame, brightness_thresh=50, bottom_fraction=0.5):
    """
    Improved string detection using horizontal morphological operations.
    """
    height, width = frame.shape[:2]
    start_row = int(height * (1 - bottom_fraction))
    bottom_frame = frame[start_row:, :]

    # --- Grayscale and Contrast enhancement ---
    gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast to make strings pop
    gray = cv2.equalizeHist(gray)

    # --- Horizontal gradient (Sobel Y) ---
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)

    # --- Threshold ---
    _, thresh = cv2.threshold(sobely, brightness_thresh, 255, cv2.THRESH_BINARY)

    # --- CRITICAL CHANGE: Horizontal Kernel ---
    # This connects the thin string segments horizontally
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # --- Hough transform ---
    lines = cv2.HoughLinesP(
        thresh,
        rho=1,
        theta=np.pi/180,
        threshold=40,       # Increased threshold for more stability
        minLineLength=100,    # Strings are long, so we want long lines
        maxLineGap=10
    )

    horizontal_lines = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            
            # Calculate angle to filter out non-horizontal lines
            angle = np.abs(np.degrees(np.arctan2(dy, dx)))
            
            # Allow only lines that are close to horizontal (e.g., < 10 degrees)
            if angle < 10:
                horizontal_lines.append([x1, y1 + start_row, x2, y2 + start_row])

    # Sort by Y position
    horizontal_lines.sort(key=lambda l: l[1])
    
    # Optional: Merge lines that are too close to each other (similar to your fret logic)
    return horizontal_lines

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Try a lower threshold if strings are hard to see
        strings = detect_strings_bottom(frame, brightness_thresh=40)

        for line in strings:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Improved Strings Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()