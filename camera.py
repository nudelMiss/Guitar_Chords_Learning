import cv2

# --- 1. Imports from guitar files ---
from vision.guitar_detection.fret_det import detect_frets_bottom
from vision.guitar_detection.guitar_detector import detect_strings_bottom

# --- 2. Import from Hand Detector ---
from vision.hand_detection.hand_detector_cv import HandDetector

# --- 3. Import the Chord Logic ---
from chords_identify.chord_logic import ChordIdentifier

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        # Initialize the classes
        self.hand_detector = HandDetector()
        self.chord_logic = ChordIdentifier()

    def __del__(self):
        self.video.release()

    def process_frame(self, frame):
        # 1) Get frets/strings from Ayala's functions
        raw_frets = detect_frets_bottom(frame)
        raw_strings = detect_strings_bottom(frame)

        # Ayala's code returns lines [x1, y1, x2, y2]. We need just the 1D arrays of X and Y.
        frets_x = [line[0] for line in raw_frets] if raw_frets else []
        strings_y = [line[1] for line in raw_strings] if raw_strings else []

        # 2) Calibrate your hand detector using the extracted grid
        if frets_x and len(frets_x) >= 2:
            self.hand_detector.calibrate_frets(frets_x)
        if strings_y and len(strings_y) >= 6:
            self.hand_detector.calibrate_strings(strings_y)

        # 3) Your detection: returns (num_fingers, list of (x, y, fret, string))
        num_fingers, finger_data = self.hand_detector.detect_fingers_and_frets(frame)

        # 4) Data Translation for Chord Logic
        # Your detector sorts points from left to right (Pinky to Thumb on left hand facing camera).
        # ChordLogic expects IDs: 1=Index, 2=Middle, 3=Ring, 4=Pinky
        logic_finger_mapping = [4, 3, 2, 1, 0] # Left to right mapping
        
        triples = []
        for i, data in enumerate(finger_data):
            x, y, fret, string = data
            
            # Map the visual finger to the ID number
            finger_id = logic_finger_mapping[i] if i < len(logic_finger_mapping) else 0
            
            # Draw the point on the finger
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            
            if finger_id != 0: # Usually the thumb (0) doesn't press strings in basic chords
                triples.append((fret, string, finger_id))

        # 5) Chord logic consumes the translated triples
        chord_result = self.chord_logic.identify(triples)
        
        # 6) Visualization
        # Draw frets (Red)
        for x in frets_x:
            cv2.line(frame, (int(x), 0), (int(x), frame.shape[0]), (0, 0, 255), 1)
        # Draw strings (Green)
        for y in strings_y:
            cv2.line(frame, (0, int(y)), (frame.shape[1], int(y)), (0, 255, 0), 1)

        # Show Output texts
        # Display the message directly from the logic engine
        message = chord_result.get("message", "No chord detected.")
        cv2.putText(frame, f'Status: {message}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        
        cv2.putText(frame, f'Active Frets: {triples}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return frame

    def get_frame(self):
        ok, frame = self.video.read()
        if not ok:
            return None

        # Mirror the frame so it acts like a natural mirror for the user
        frame = cv2.flip(frame, 1)
        frame = self.process_frame(frame)

        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()