#!/usr/bin/env python3
import cv2
from hand_detector import HandDetector


def main():
    hd = HandDetector(max_hands=1, detection_conf=0.6, tracking_conf=0.6)

    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # macOS friendly
    if not cap.isOpened():
        print("No camera available")
        return

    # optional: set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("failed to read frame")
            break

        num, tips = hd.detect_fingers_and_frets(frame)

        vis = hd.draw_debug(frame)
        cv2.putText(vis, f"tips: {num}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Print tips (you will map these to fretboard later)
        # tips = [(name, x, y), ...]
        # comment this out if spammy:
        # print(tips)

        cv2.imshow("camera (MediaPipe fingertips)", vis)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
