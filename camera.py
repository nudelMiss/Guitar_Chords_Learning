# camera.py

import cv2

class VideoCamera(object):
    def __init__(self):
        # Open the webcam (0 is usually the default camera)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release the camera when the object is destroyed
        self.video.release()

    def process_frame(self, frame):
        
        
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
    
    

#-------------------------------- --------------------------------
#               temporary camera.py for UI development
#-------------------------------- --------------------------------

# camera.py (Temporary version for UI development)
# import cv2

# class VideoCamera(object):
#     def __init__(self):
#         # Open the default camera (index 0)
#         self.video = cv2.VideoCapture(0)

#     def __del__(self):
#         # Release the camera resource
#         self.video.release()

#     def get_frame(self):
#         success, frame = self.video.read()
#         if not success:
#             return None
            
#         # Flip the frame horizontally (mirror effect) for better UX
#         frame = cv2.flip(frame, 1)

#         # --- ALGORITHMS BYPASSED FOR UI TESTING ---
#         # We are skipping the heavy logic (Guitar/Hand detection) 
#         # to allow the web server to run without errors.
        
#         # Draw static text to confirm video feed is working
#         cv2.putText(frame, "UI Mode - Logic Disabled", (50, 50), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
#         # Encode frame to JPEG format for the browser
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         return jpeg.tobytes()