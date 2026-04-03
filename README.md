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

## Overview
A robust Computer Vision and Web-based application designed for tracking and recognizing guitar chords in real-time.

This interactive learning tool utilizes **MediaPipe** for precise hand and finger tracking, alongside **OpenCV** for optical flow and guitar neck analysis. The backend is powered by **Flask**, serving a user-friendly web interface. Users can seamlessly calibrate their instrument, practice specific chord shapes in training mode, and play along with songs using the built-in tracking and timing logic.

## Screenshots

![Uploading Screenshot 2026-03-21 at 20.52.20.png…]()


## Technical Architecture

### 1. Image Enhancement and Pre-processing
* **CLAHE (Contrast Limited Adaptive Histogram Equalization):** An image processing technique applied to normalize lighting variations and enhance local contrast across the fretboard. It divides the image into small regions and performs local equalization, enhancing details in low-contrast areas without excessively amplifying noise.
* **Bilateral Filter:** A non-linear filter used to reduce background noise and wood textures while keeping the sharp metallic edges of the fret wires intact. It adds a second Gaussian component based on pixel intensity difference, ensuring pixels with significantly different intensities (like a bright fret against dark wood) are preserved.
* **1D Gaussian Kernel Convolution:** A discrete Gaussian kernel convolved with the 1D signal to suppress high-frequency noise caused by wood grain or minor reflections, specifically targeting the signal vector rather than smoothing the entire image.

### 2. Object Tracking 
* **Feature Detection (ORB):** Oriented FAST and Rotated BRIEF is used to identify distinctive keypoints and generate binary descriptors resistant to noise and rotation, providing a robust, real-time baseline for structural representation.
* **Motion Tracking (Sparse Optical Flow):** Utilizing the Lucas-Kanade method to track the movement of specific feature points between consecutive video frames. By analyzing local image intensities, it provides fast and continuous real-time tracking of moving objects.

### 3. Feature Extraction and Geometric Modeling
* **Edge Detection (Sobel):** Computes image gradients to identify strong intensity transitions. The Sobel operator approximates the derivative of the image intensity, highlighting structural elements.
* **Feature Matching and RANSAC:** Finds correspondences between keypoints in different images. The Random Sample Consensus (RANSAC) algorithm iteratively estimates geometric transformations, discarding outlier matches to ensure accuracy.
* **Geometric Modeling:** Uses mathematical modeling to identify structured elements like string spacing patterns when direct detection is difficult.

### 4. Hand Pose Estimation and Validation
* **Intensity Dip Detection:** Locates strings by calculating horizontal intensity profiles. Strings, appearing as dark lines, are identified as prominent local minima (dips) in the averaged pixel values.
* **Hand Landmark Detection (MediaPipe):** A machine learning framework for real-time hand tracking. It detects predefined landmarks representing key points on the hand (fingertips, joints) for precise analysis of finger placement.
* **Skin Segmentation:** A chrominance-based detection method separating luminance from color to provide illumination-invariant skin isolation.
* **Shape Compactness:** A geometric metric (`C = 4π * A / P^2`) used to differentiate circular fingertips from elongated forearm regions.
* **Temporal Persistence:** A finite state machine requiring consistent detection over consecutive frames to "debounce" finger status and prevent flickering.

## Project Structure
* `app.py`: The main Flask server managing web routes, endpoints, and video streaming.
* `song_main.py` / `GuitarSystem`: The central coordinating logic for video processing, tracking, and song timing.
* `Chords.py`: Database containing chord fingerings, open strings, and muted strings.
* `string_and_frets.py`: Custom logic and algorithms for guitar neck and string calibration.
* `templates/`: HTML files for the web user interface.

## Getting Started

### Prerequisites
Ensure you have Python 3.9 or higher installed on your system.

### Installation & Running the Application

1. **Clone the repository:**
   ```bash
   # Clone the project to your local machine
   git clone <repository-url>

   # Navigate into the project directory
   cd guitar-chord-recognition
