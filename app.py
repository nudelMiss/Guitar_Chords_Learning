from flask import Flask, render_template, Response, jsonify
import cv2
import threading

# Import the main guitar tracking logic
from song_main import GuitarSystem 

app = Flask(__name__)

# Create the core system instance once
guitar = GuitarSystem() 

# Use a lock to ensure thread-safe operations when reading/writing the key command
key_lock = threading.Lock()
last_web_key = -1

@app.route('/')
def index():
    # Render the main landing page
    return render_template('index.html')

@app.route('/chords-training')
def chords_training(): 
    # Render the specific page for training chords
    return render_template('chords-training.html')

@app.route('/song-practice')
def song_practice():
    # Render the specific page for practicing songs
    return render_template('song-practice.html')

@app.route('/send_command/<key>')
def send_command(key):
    global last_web_key
    
    # Acquire the lock before modifying the global state
    with key_lock:
        # Convert the received character to its ASCII value for the cv2 logic
        last_web_key = ord(key.lower())
        
    return "OK", 200

@app.route('/start_song/<song_id>')
def start_song(song_id):
    # Tell the guitar system to start the timer and load chords for this song
    # This is crucial for updating the current_chord_name!
    guitar.start_playing_song(song_id)
    return "OK", 200

@app.route('/play-song/<song_name>')
def play_song(song_name):
    # Pass the song ID to the new template so it knows which song to display
    # You can later use this ID to load the specific chords for this song
    return render_template('play-song.html', song_name=song_name)

@app.route('/current_chord')
def current_chord():
    # Return both chord and lyric as a JSON object
    return jsonify({
        "chord": guitar.current_chord_name if guitar.current_chord_name else "Ready",
        "lyric": guitar.current_lyric if guitar.is_playing_song else ""
    })

def generate_frames():
    global last_web_key
    
    # Open the default webcam
    cap = cv2.VideoCapture(0)
    
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Safely read and immediately reset the current key command
            with key_lock:
                current_key = last_web_key
                # Reset the key so the command executes only once per press
                last_web_key = -1 

            # Process the frame using the GuitarSystem logic
            # This handles both the computer vision and drawing the UI overlays
            display_frame = guitar.process_frame(frame, current_key)
            
            # Encode the processed frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', display_frame)
            
            # Yield the byte stream for the browser to consume
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    finally:
        # Crucial for resource management: release the camera if the stream stops 
        # (e.g., when the user refreshes or closes the browser tab)
        cap.release()

@app.route('/video_feed')
def video_feed():
    # Serve the continuous video stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True, port=5000)