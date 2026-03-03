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

    # --- Setup for Stability ---
    frame_buffer = []      # List to store detections from the last N frames
    MAX_BUFFER_SIZE = 6    # How many frames to remember (Temporal Memory)
    brightness_thresh = 120
    bottom_fraction = 0.4

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detect raw lines in the current frame
        raw_lines = detect_frets_bottom(frame, brightness_thresh, bottom_fraction)

        # 2. Update the temporal buffer
        frame_buffer.append(raw_lines)
        if len(frame_buffer) > MAX_BUFFER_SIZE:
            frame_buffer.pop(0) # Remove oldest frame detections

        # 3. Flatten buffer: combine all lines from all frames in the buffer
        all_recent_detections = [line for f_lines in frame_buffer for line in f_lines]

        # 4. Merge them using the improved logic
        # Increasing x_threshold slightly helps group jittery lines together
        stable_frets = merge_vertical_lines(all_recent_detections, x_threshold=20)

        # 5. Detect the neck boundaries
        y_top, y_bottom = detect_guitar_neck_bounds(frame)

        # 6. Draw the neck boundaries for reference (Blue lines)
        cv2.line(frame, (0, y_top), (frame.shape[1], y_top), (255, 0, 0), 2)
        cv2.line(frame, (0, y_bottom), (frame.shape[1], y_bottom), (255, 0, 0), 2)

        # 7. Use these boundaries when drawing your STABLE frets
        for line in stable_frets:
            # We ignore the original y1, y2 from detection and use the neck bounds
            x_pos = line[0]
            cv2.line(frame, (x_pos, y_top), (x_pos, y_bottom), (0, 255, 0), 2)

        # 8. Drawing - Visualizing the stable result
        for line in stable_frets:
            # Green lines for stable, merged frets
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)

        # Optional: draw raw detections in thin red to see the difference
        # for line in raw_lines:
        #    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)

        cv2.imshow("Stable Fret Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def merge_vertical_lines(lines, x_threshold=50):
    """
    Groups vertical lines based on horizontal proximity and averages them.
    """
    if not lines:
        return []

    # Sort lines by X coordinate to ensure they are processed in order
    lines.sort(key=lambda l: l[0])

    groups = []
    if len(lines) > 0:
        # Start the first group with the first line
        current_group = [lines[0]]

        for i in range(1, len(lines)):
            # If the X distance between current and previous line is small, group them
            if abs(lines[i][0] - lines[i - 1][0]) <= x_threshold:
                current_group.append(lines[i])
            else:
                # Close current group and start a new one
                groups.append(current_group)
                current_group = [lines[i]]

        # Add the last group to the list
        groups.append(current_group)

    # Calculate a single representative line for each group
    final_lines = []
    for group in groups:
        # Average the X coordinates
        avg_x = int(np.mean([l[0] for l in group]))
        # Take the extreme Y values to cover the full length of the group
        min_y = min([l[1] for l in group])
        max_y = max([l[3] for l in group])

        # Format as [x1, y1, x2, y2] to keep consistency with detection output
        final_lines.append([avg_x, min_y, avg_x, max_y])

    return final_lines


def detect_guitar_neck_bounds(frame):
    """
    Detects the top and bottom horizontal boundaries of the guitar neck.
    Returns (y_top, y_bottom)
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detect horizontal edges using Sobel Y ---
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)

    # Threshold to find strong horizontal lines (strings/neck edges)
    _, thresh = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY)

    # --- Hough Transform for horizontal lines ---
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=100,
        minLineLength=width // 3, maxLineGap=20
    )

    y_coords = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Filter for mostly horizontal lines
            if abs(y2 - y1) < 10:
                y_coords.append((y1 + y2) // 2)

    if len(y_coords) >= 2:
        y_top = min(y_coords)
        y_bottom = max(y_coords)
        return y_top, y_bottom

    # Default values if no neck is detected (preventing errors)
    return int(height * 0.4), int(height * 0.9)


if __name__ == "__main__":
    main()