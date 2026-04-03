---
name: opencv-debugging
description: Guide for debugging OpenCV-based computer vision pipelines in this project, including detection failures, tracking instability, and performance issues. Use this when debugging any vision-related problem.
---

# OpenCV Debugging

This skill helps debug issues in the computer vision pipeline.

## Use this skill when

Use this skill when:
- detection is unstable or incorrect
- tracking is jittery or drifting
- frame rate is slow
- overlays don’t match detected structures
- results vary a lot between frames
- something "looks wrong" but code seems correct

## Core debugging principle

Never debug OpenCV blindly.

Always **visualize intermediate steps**.

## Step-by-step debugging strategy

### 1. Visualize every stage

Add debug outputs for:
- raw frame
- cropped ROI
- masks / thresholds
- edges (Canny / Sobel)
- detected lines (Hough)
- tracked model (strings, frets, corners)

If you can’t see it, you can’t debug it.

---

### 2. Isolate the failing stage

Ask:
- Is detection wrong?
- Or is tracking wrong?
- Or is mapping wrong?

Fix one layer at a time.

---

### 3. Check assumptions

Common hidden assumptions:
- lighting is stable
- guitar color is consistent
- background is simple
- camera exposure doesn’t change

If any of these break → your pipeline may fail.

---

## Common OpenCV issues in this project

### 1. Bad edge detection

Symptoms:
- too many edges
- or missing important lines

Fix:
- tune thresholds
- blur before edge detection
- limit to ROI

---

### 2. Hough lines detect noise

Symptoms:
- random lines
- missing strings

Fix:
- filter by angle (horizontal lines for strings)
- filter by length
- cluster similar lines

---

### 3. Color thresholding unstable

Symptoms:
- mask changes dramatically between frames

Fix:
- avoid relying only on color
- use adaptive thresholds or combine cues
- restrict region

---

### 4. Flickering detections

Symptoms:
- objects appear/disappear every frame

Fix:
- use temporal smoothing
- require persistence over frames
- store recent history

---

### 5. Slow performance

Symptoms:
- low FPS
- laggy UI

Fix:
- reduce resolution
- avoid heavy operations every frame
- reuse previous results when possible
- move calibration-heavy work out of main loop

---

## Performance tips

- Crop early (ROI first!)
- Avoid recomputing expensive transforms
- Use integer operations where possible
- Cache models (but NOT pixel positions tied to frames)
- Use tracking instead of full detection every frame

---

## Debug tools to add

Highly recommended:

- toggleable debug mode
- overlay:
  - strings
  - frets
  - detected lines
  - tracked corners
- print:
  - number of detected lines
  - tracking confidence
  - FPS

---

## Golden rules

1. If something is unstable → add smoothing or constraints
2. If something is slow → reduce computation or frequency
3. If something is wrong → visualize before changing code
4. If something works sometimes → find what changes between frames

---

## Validation checklist

After fixing an issue:

- test with slight motion
- test with different lighting
- test calibration + tracking together
- test switching modes
- verify FPS is acceptable

A good CV pipeline is not just correct — it is stable and predictable.