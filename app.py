from flask import Flask, render_template, Response, jsonify
import cv2
import threading

from GuitarSystem import GuitarSystem

app = Flask(__name__)

# Core system
guitar = GuitarSystem()

# Thread-safe key handling
key_lock = threading.Lock()
last_web_key = -1

# Thread-safe camera handling
camera_lock = threading.Lock()
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")

@app.route('/')
def index():
    guitar._reset_all()
    return render_template('index.html')

@app.route('/chords-training')
def chords_training():
    guitar._reset_all()
    return render_template('chords-training.html')

@app.route('/set_chord/<chord_name>')
def set_chord(chord_name):
    # This directly sets the chord in the guitar system
    guitar.current_chord_name = chord_name.upper()
    import chords_data
    guitar.current_chord_data = chords_data.CHORD_LIBRARY.get(guitar.current_chord_name)
    return "OK", 200

@app.route('/song-practice')
def song_practice():
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
    guitar.show_tracker_lines = (state == "1")
    return "OK", 200

@app.route('/start_song/<song_id>')
def start_song(song_id):
    guitar.start_playing_song(song_id)
    return "OK", 200

@app.route('/play-song/<song_name>')
def play_song(song_name):
    return render_template('play-song.html', song_name=song_name)

@app.route('/stop_song')
def stop_song():
    guitar.stop_playing_song()
    guitar.show_tracker_lines = True
    guitar._reset_calibration_only()
    return "OK", 200

@app.route('/current_chord')
def current_chord():
    return jsonify({
        "chord": guitar.current_chord_name if guitar.current_chord_name else "Ready",
        "lyric": guitar.current_lyric if guitar.is_playing_song else ""
    })

def generate_frames():
    global last_web_key, camera

    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success or frame is None:
            continue

        with key_lock:
            current_key = last_web_key
            last_web_key = -1

        display_frame = guitar.process_frame(frame, current_key)

        ret, buffer = cv2.imencode('.jpg', display_frame)
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
    app.run(debug=True, port=5000)