from flask import Flask, render_template, Response, jsonify
import cv2
import threading

from GuitarSystem import GuitarSystem

app = Flask(__name__)


class CameraCapture:
    """Reads frames in a background thread so the processing loop never blocks."""

    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while self._running:
            ok, frame = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._running = False
        self._thread.join(timeout=2)
        self.cap.release()


# Core system
guitar = GuitarSystem()
guitar_lock = threading.Lock()

# Thread-safe key handling
key_lock = threading.Lock()
last_web_key = -1

# Camera with background capture thread
camera = CameraCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")

@app.route('/')
def index():
    with guitar_lock:
        guitar._reset_all()
    return render_template('index.html')

@app.route('/chords-training')
def chords_training():
    with guitar_lock:
        guitar._reset_all()
    return render_template('chords-training.html')

@app.route('/set_chord/<chord_name>')
def set_chord(chord_name):
    import chords_data
    with guitar_lock:
        guitar.current_chord_name = chord_name.upper()
        guitar.current_chord_data = chords_data.CHORD_LIBRARY.get(guitar.current_chord_name)
    return "OK", 200

@app.route('/song-practice')
def song_practice():
    with guitar_lock:
        guitar._reset_all()
    return render_template('song-practice.html')

@app.route('/send_command/<key>')
def send_command(key):
    global last_web_key
    with key_lock:
        last_web_key = ord(key.lower())
    return "OK", 200
    
@app.route('/set_tracker_lines/<state>')
def set_tracker_lines(state):
    with guitar_lock:
        guitar.show_tracker_lines = (state == "1")
    return "OK", 200

@app.route('/start_song/<song_id>')
def start_song(song_id):
    with guitar_lock:
        guitar.start_playing_song(song_id)
    return "OK", 200

@app.route('/play-song/<song_name>')
def play_song(song_name):
    return render_template('song-practice.html', song_name=song_name)

@app.route('/stop_song')
def stop_song():
    with guitar_lock:
        guitar.stop_playing_song()
        guitar.show_tracker_lines = True
        guitar._reset_calibration_only()
    return "OK", 200

@app.route('/current_chord')
def current_chord():
    with guitar_lock:
        chord = guitar.current_chord_name if guitar.current_chord_name else "Ready"
        lyric = guitar.current_lyric if guitar.is_playing_song else ""
    return jsonify({"chord": chord, "lyric": lyric})

def generate_frames():
    global last_web_key

    while True:
        success, frame = camera.read()

        if not success or frame is None:
            continue

        with key_lock:
            current_key = last_web_key
            last_web_key = -1

        try:
            with guitar_lock:
                display_frame = guitar.process_frame(frame, current_key)
        except Exception as e:
            print("ERROR in process_frame:", e)
            display_frame = frame

        ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, threaded=True, use_reloader=False, port=5000)