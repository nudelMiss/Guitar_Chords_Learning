# Guitar Chord Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/MediaPipe-FF3614?style=for-the-badge&logo=mediapipe&logoColor=white" alt="MediaPipe" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E" alt="JavaScript" />
  <img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git" />
</p>


https://github.com/user-attachments/assets/5faa1886-a52b-4051-bd88-b93b27bcef09


## Overview
A robust Computer Vision and Web-based application designed for tracking and recognizing guitar chords in real-time.

This interactive learning tool utilizes **MediaPipe** for precise hand and finger tracking, alongside **OpenCV** for optical flow and guitar neck analysis. The backend is powered by **Flask**, serving a user-friendly web interface. Users can seamlessly calibrate their instrument, practice specific chord shapes in training mode, and play along with songs using the built-in tracking and timing logic.

## Project Structure
* `app.py`: The main Flask server managing web routes, endpoints, and video streaming.
* `song_main.py` / `GuitarSystem`: The central coordinating logic for video processing, tracking, and song timing.
* `Chords.py`: Database containing chord fingerings, open strings, and muted strings.
* `string_and_frets.py`: Custom logic and algorithms for guitar neck and string calibration.
* `templates/`: HTML files for the web user interface.

## Hand Tracking Modes: Deep Learning vs. Computer Vision
The system allows switching between two primary methods for hand tracking to optimize performance and accuracy:

* Deep Learning Mode (MediaPipe): Utilizes pre-trained neural networks for high-precision landmark detection of fingers and joints.

* Computer Vision Mode (Optical Flow): Uses traditional CV algorithms to track movement and positioning relative to the guitar neck.

**How to switch:**
To switch modes, modify the instantiation in GuitarSystem.py:
(line 15): system = GuitarSystem(use_mediapipe=True)

## Getting Started

### Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

### Installation & Running the Application

1. **Clone the repository:**
   ```bash
   # Clone the project to your local machine
   git clone https://github.com/nudelMiss/Guitar_Chords_Learning.git

   # Navigate into the project directory
   cd guitar-chord-recognition
2. **Create a virtual environment (Recommended):**
   ```bash
   # Create a virtual environment to keep dependencies organized
    python -m venv venv

    # Activate it:
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
3. **Install dependencies:**
   ```bash
   # Install all required libraries
    pip install -r requirements.txt
4. **Run the application:**
   ```bash
    # Start the Flask server
    python app.py
    Open your browser and navigate to http://127.0.0.1:5000.
