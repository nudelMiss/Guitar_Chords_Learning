from flask import Flask, render_template, Response
import cv2
import numpy as np
import string_and_frets as sf
import central_coordinating as cc

app = Flask(__name__)

# Global state to sync with the camera loop
last_web_key = -1

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Initialize all your variables from the central logic
    is_tracking = False
    locked_model = {}
    fret_model_rel = []
    string_model_rel = []
    is_strings_calibrated = False
    tracking_pts = None
    initial_pts = None
    last_gray = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        global last_web_key
        key = last_web_key
        last_web_key = -1 # Reset after capturing

        # Prepare frames
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = frame.copy()

        # --- REPLICATING CENTRAL_COORDINATING LOGIC ---
        if is_tracking and tracking_pts is not None:
            # Tracking logic: Optical Flow
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(last_gray, gray, tracking_pts, None)
            if new_pts is not None and status is not None:
                good_new = new_pts[status.flatten() == 1]
                good_old = initial_pts[status.flatten() == 1]
                
                if len(good_new) >= 4:
                    matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                    if matrix is not None:
                        tracking_pts = cv2.transform(initial_pts, matrix)
                        last_gray = gray.copy()
                        
                        # Calculate corners
                        corners = np.array([[locked_model["x_min"], locked_model["y_t"]],
                                          [locked_model["x_max"], locked_model["y_t"]],
                                          [locked_model["x_min"], locked_model["y_b"]],
                                          [locked_model["x_max"], locked_model["y_b"]]], dtype=np.float32).reshape(-1,1,2)
                        
                        tc = cv2.transform(corners, matrix).reshape(-1, 2)
                        tl, tr, bl, br = tc[0], tc[1], tc[2], tc[3]

                        # DRAW RED/GREEN LINES (The feedback you were missing)
                        # Draw Frets
                        for rel_x in fret_model_rel:
                            p1 = (tl + rel_x * (tr - tl)).astype(int)
                            p2 = (bl + rel_x * (br - bl)).astype(int)
                            cv2.line(display_frame, tuple(p1), tuple(p2), (0, 255, 0), 2)
                        # Draw Neck Bounds
                        cv2.line(display_frame, tuple(tl.astype(int)), tuple(tr.astype(int)), (255, 0, 0), 2)
                        cv2.line(display_frame, tuple(bl.astype(int)), tuple(br.astype(int)), (255, 0, 0), 2)

                        # Handle String Calibration Command from Web
                        if key == ord('s'):
                            curr_model = {"x_min": int(np.min(tc[:,0])), "x_max": int(np.max(tc[:,0])),
                                          "y_t": int(np.min(tc[:,1])), "y_b": int(np.max(tc[:,1]))}
                            detected = sf.detect_strings_in_neck(frame, curr_model)
                            if len(detected) == 6:
                                string_model_rel = detected
                                is_strings_calibrated = True

        else:
            # PREVIEW MODE: Looking for the neck (Red lines before locking)
            raw_frets = sf.detect_frets_bottom(frame)
            y_t, y_b = sf.detect_guitar_neck_bounds(frame)

            if y_t is not None and y_b is not None:
                # THESE ARE THE RED LINES YOU WANT TO SEE
                cv2.line(display_frame, (0, y_t), (width, y_t), (0, 0, 255), 2)
                cv2.line(display_frame, (0, y_b), (width, y_b), (0, 0, 255), 2)

            # Handle Lock Command (C) from Web
            if key == ord('c') and y_t is not None and len(raw_frets) > 2:
                stable_f = sf.merge_vertical_lines(raw_frets)
                if len(stable_f) >= 2:
                    x_min, x_max = stable_f[0][0], stable_f[-1][0]
                    # Setup points for tracking
                    grid_x = np.linspace(x_min, x_max, 10)
                    grid_y = np.linspace(y_t, y_b, 4)
                    temp_pts = [[gx, gy] for gx in grid_x for gy in grid_y]
                    initial_pts = np.array(temp_pts, dtype=np.float32).reshape(-1, 1, 2)
                    tracking_pts = initial_pts.copy()
                    locked_model = {"x_min": x_min, "x_max": x_max, "y_t": y_t, "y_b": y_b}
                    fret_model_rel = [(f[0] - x_min) / (x_max - x_min) for f in stable_f]
                    last_gray = gray.copy()
                    is_tracking = True

        # Encode the frame with drawings to be sent to the browser
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/')
def index():
    # The main landing page we designed earlier
    return render_template('index.html')

@app.route('/chords-training')
def chords_training():
    return render_template('chords-training.html')

@app.route('/song-practice')
def song_practice():
    return render_template('song-practice.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_command/<key>')
def send_command(key):
    global last_web_key
    last_web_key = ord(key)
    return f"Command {key} received", 200

if __name__ == '__main__':
    app.run(debug=True)