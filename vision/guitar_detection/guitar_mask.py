import cv2
import numpy as np

# ===== FINAL INSTRUMENT MASK VALUES (from your tuner) =====
LOWER_INST = np.array([0, 57, 60], dtype=np.uint8)
UPPER_INST = np.array([166, 241, 245], dtype=np.uint8)
K_OPEN = 1
K_CLOSE = 21


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask from calibrated HSV
        mask = cv2.inRange(hsv, LOWER_INST, UPPER_INST)

        # Morphological cleaning
        k_open = max(1, K_OPEN | 1)
        k_close = max(1, K_CLOSE | 1)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Keep only the best connected component (prefer big + rightmost)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num > 1:
            best_label = 1
            best_score = -1.0

            # (optional) ignore tiny blobs
            MIN_AREA = 2000

            for lbl in range(1, num):
                area = stats[lbl, cv2.CC_STAT_AREA]
                if area < MIN_AREA:
                    continue

                cx = centroids[lbl][0]  # centroid x

                # Score: prefer large components and those more to the right
                score = area + 0.0008 * cx * area  # tweak 0.0008 if needed

                if score > best_score:
                    best_score = score
                    best_label = lbl

            mask = (labels == best_label).astype(np.uint8) * 255

            # Crop to the selected component's bounding box (removes stray blobs)
            x = stats[best_label, cv2.CC_STAT_LEFT]
            y = stats[best_label, cv2.CC_STAT_TOP]
            w = stats[best_label, cv2.CC_STAT_WIDTH]
            h = stats[best_label, cv2.CC_STAT_HEIGHT]

            clean = np.zeros_like(mask)
            clean[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
            mask = clean

        # Overlay mask for visualization
        overlay = frame.copy()
        overlay[mask > 0] = (0, 255, 0)
        vis = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

        cv2.imshow("Instrument Mask Overlay", vis)
        cv2.imshow("Instrument Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()