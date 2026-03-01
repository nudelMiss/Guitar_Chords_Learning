import cv2
import numpy as np

def nothing(x):
    # Dummy callback function required by OpenCV trackbars
    pass

def main():
    # Create the windows first before attaching trackbars
    cv2.namedWindow('Original Video')
    cv2.namedWindow('Mask')
    
    # Create trackbars directly on the 'Mask' window so they won't disappear
    cv2.createTrackbar('H-Min', 'Mask', 0, 179, nothing)
    cv2.createTrackbar('S-Min', 'Mask', 20, 255, nothing)
    cv2.createTrackbar('V-Min', 'Mask', 70, 255, nothing)
    
    cv2.createTrackbar('H-Max', 'Mask', 20, 179, nothing)
    cv2.createTrackbar('S-Max', 'Mask', 255, 255, nothing)
    cv2.createTrackbar('V-Max', 'Mask', 255, 255, nothing)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    print("Adjust the sliders on the 'Mask' window.")
    print("Press 'q' when you are done to print the values.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Read current positions of all trackbars from the 'Mask' window
        h_min = cv2.getTrackbarPos('H-Min', 'Mask')
        s_min = cv2.getTrackbarPos('S-Min', 'Mask')
        v_min = cv2.getTrackbarPos('V-Min', 'Mask')
        
        h_max = cv2.getTrackbarPos('H-Max', 'Mask')
        s_max = cv2.getTrackbarPos('S-Max', 'Mask')
        v_max = cv2.getTrackbarPos('V-Max', 'Mask')

        # Create numpy arrays for the color boundaries based on sliders
        lower_skin = np.array([h_min, s_min, v_min])
        upper_skin = np.array([h_max, s_max, v_max])

        # Create the binary mask (white for skin, black for background)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Display the frames
        cv2.imshow('Original Video', frame)
        cv2.imshow('Mask', mask)

        # Wait for the 'q' key to quit and print the selected values
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n=== YOUR PERFECT CALIBRATION VALUES ===")
            print(f"self.lower_skin = np.array([{h_min}, {s_min}, {v_min}], dtype=np.uint8)")
            print(f"self.upper_skin = np.array([{h_max}, {s_max}, {v_max}], dtype=np.uint8)")
            print("=======================================\n")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()