# app.py
from flask import Flask, render_template, Response
from camera import VideoCamera

app = Flask(__name__)

def gen(camera):
    """
    Generator function that yields frames continuously.
    Creates a 'Motion JPEG' stream.
    """
    while True:
        frame = camera.get_frame()
        if frame is not None:
            # MJPEG format structure
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    # Renders the HTML page
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Returns the streaming response
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Run the server on localhost:5000
    # debug=True allows the server to auto-reload when you save code changes
    app.run(host='0.0.0.0', port=5000, debug=True)