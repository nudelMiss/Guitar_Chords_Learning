# camera.py
import cv2
import numpy as np

class VideoCamera(object):
    def __init__(self):
        # Open the webcam (0 is usually the default camera)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release the camera when the object is destroyed
        self.video.release()

    def process_frame(self, frame):
        """
        THIS IS WHERE YOUR CODE GOES!
        Receive a raw frame, apply CV logic, and return the processed frame.
        """
        
        # --- START OF YOUR ALGORITHM ---
        
        # Example 1: Convert to Grayscale just to show it works
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Example 2: Drawing a static rectangle (Simulating a fretboard zone)
        cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
        
        # Example 3: Add text
        cv2.putText(frame, "Guitar Hero Logic Here", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # --- END OF YOUR ALGORITHM ---
        
        return frame

    def get_frame(self):
        # 1. Read frame from camera
        success, frame = self.video.read()
        if not success:
            return None
        
        # 2. Flip the frame (mirror effect)
        frame = cv2.flip(frame, 1)

        # 3. Process the frame using your logic
        processed_frame = self.process_frame(frame)

        # 4. Encode the frame to JPEG so the browser can understand it
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        return jpeg.tobytes()