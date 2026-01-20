# camera.py
import cv2
from guitar_detector import GuitarDetector      # Ayala
from hand_detector import HandDetector          # Michal
from chord_logic import ChordLogic              # Yuval
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.guitar_detector = GuitarDetector()
        self.hand_detector = HandDetector()
        self.chord_logic = ChordLogic()

    def __del__(self):
        self.video.release()

    def process_frame(self, frame):
        # 1) Get frets/strings from Girl 1 (DO NOT hardcode here)
        # ---- CHANGE THIS LINE ONLY if Girl 1 API is different ----
        frets_x, strings_y = self.guitar_detector.detect(frame)
        # ---------------------------------------------------------

        # 2) Calibrate your hand detector using Girl 1 output
        if frets_x and len(frets_x) >= 2:
            self.hand_detector.calibrate_frets(frets_x)
        if strings_y and len(strings_y) >= 6:
            self.hand_detector.calibrate_strings(strings_y)

        # 3) Your detection: return triples (fret,string,finger)
        num_fingers, triples = self.hand_detector.detect_fingers_and_frets(frame)

        # 4) Chord logic consumes YOUR triples
        chord_result = self.chord_logic.predict(triples)  # or whatever function name

        # 5) Visualization (optional but super useful)
        # draw frets/strings received from Girl 1
        for x in frets_x:
            cv2.line(frame, (int(x), 0), (int(x), frame.shape[0]), (255, 255, 0), 1)
        for y in strings_y:
            cv2.line(frame, (0, int(y)), (frame.shape[1], int(y)), (0, 255, 255), 1)

        # show outputs
        cv2.putText(frame, f'Fingers: {num_fingers}', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.putText(frame, f'Triples: {triples}', (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f'Chord: {chord_result}', (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        return frame

    def get_frame(self):
        ok, frame = self.video.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        frame = self.process_frame(frame)

        ret, jpeg = cv2.imencode(".jpg", frame)
        return jpeg.tobytes()
