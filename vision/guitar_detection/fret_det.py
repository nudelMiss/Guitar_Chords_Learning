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

    # --- Stability/Hysteresis Memory ---
    last_stable_frets = []
    last_y_top, last_y_bottom = 0, 0
    STABILITY_THRESHOLD_X = 5  # Min pixels to move before updating frets
    STABILITY_THRESHOLD_Y = 5  # Min pixels to move before updating neck

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

        # 5. Detect current potential neck boundaries
        y_top_new, y_bottom_new = detect_guitar_neck_bounds(frame, bottom_fraction)

        # 6. Hysteresis: Update Neck Bounds only if change is significant
        if abs(y_top_new - last_y_top) > STABILITY_THRESHOLD_Y or \
                abs(y_bottom_new - last_y_bottom) > STABILITY_THRESHOLD_Y:
            last_y_top, last_y_bottom = y_top_new, y_bottom_new

        # 7. Hysteresis: Update Frets only if count changed or movement is significant
        if len(stable_frets) != len(last_stable_frets):
            last_stable_frets = stable_frets
        elif len(stable_frets) > 0:
            # Check average movement of all frets to see if it's just jitter
            avg_diff = np.mean([abs(stable_frets[i][0] - last_stable_frets[i][0]) for i in range(len(stable_frets))])
            if avg_diff > STABILITY_THRESHOLD_X:
                last_stable_frets = stable_frets

        # 8. Drawing - Use the LAST STABLE values to prevent flickering
        # Draw Neck Boundaries (Blue)
        cv2.line(frame, (0, last_y_top), (frame.shape[1], last_y_top), (255, 0, 0), 2)
        cv2.line(frame, (0, last_y_bottom), (frame.shape[1], last_y_bottom), (255, 0, 0), 2)

        # Draw Frets (Green) - Constrained between the blue neck lines
        for line in last_stable_frets:
            x_pos = line[0]
            cv2.line(frame, (x_pos, last_y_top), (x_pos, last_y_bottom), (0, 255, 0), 2)

        cv2.imshow("Stable Fret Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def merge_vertical_lines(lines, x_threshold=20):
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


def detect_guitar_neck_bounds(frame, bottom_fraction=0.4):
    """
    Detects the top and bottom horizontal boundaries of the guitar neck,
    searching only within the bottom part of the frame.
    """
    height, width = frame.shape[:2]
    # Calculate starting row for the ROI
    start_row = int(height * (1 - bottom_fraction))

    # Crop the frame to focus only on the bottom part
    bottom_roi = frame[start_row:, :]
    gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)

    # --- Detect horizontal edges in the ROI ---
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.convertScaleAbs(sobely)

    # Threshold for horizontal lines
    _, thresh = cv2.threshold(sobely, 50, 255, cv2.THRESH_BINARY)

    # --- Hough Transform for horizontal segments ---
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=50,
        minLineLength=width // 4, maxLineGap=50
    )

    y_coords = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            # Verify the line is mostly horizontal
            if abs(y2 - y1) < 15:
                # Add the detected Y and offset it by start_row to match full frame
                y_coords.append(((y1 + y2) // 2) + start_row)

    if len(y_coords) >= 2:
        # Sort or take min/max to find the boundary strings
        y_coords.sort()
        y_top = y_coords[0]
        y_bottom = y_coords[-1]
        return y_top, y_bottom

    # Fallback values relative to the bottom fraction if detection fails
    return int(height * 0.6), int(height * 0.9)


if __name__ == "__main__":
    main()