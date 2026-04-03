---
name: guitar-neck-tracking
description: Instructions for working on guitar neck calibration, fret and string modeling, fretboard quad tracking, temporal smoothing, and stability issues in this project. Use this when modifying or debugging the OpenCV-based guitar tracking pipeline.
---

# Guitar Neck Tracking

This skill is for tasks related to detecting, modeling, and tracking the guitar fretboard across frames.

## Use this skill when

Use this skill when the task involves:
- locking the fretboard from the preview ROI
- detecting fretboard boundaries
- detecting or refining fret positions
- calibrating strings
- tracking the fretboard between frames
- stabilizing chord overlays on the guitar
- improving refresh rate without breaking tracking
- debugging drift, jitter, lag, or bad overlay alignment

## Project context

This project is a classical computer vision system for helping beginners learn guitar chords.

Relevant files usually include:
- `string_and_frets.py` — main fretboard lock, fret/string detection, and tracking logic
- `GuitarSystem.py` — main processing flow, mode logic, chord drawing, and tracker integration
- `app.py` — Flask routes, streaming, reset behavior, and frontend/backend interaction

The project uses classical CV methods rather than full scene deep-learning detection. Preserve that design unless the task explicitly asks otherwise.

## Current tracking pipeline

The tracker currently works roughly like this:

1. Start from a fixed preview ROI
2. Lock the neck/fretboard region from that ROI
3. Detect the neck band within the ROI
4. Build a refined fretboard quadrilateral (`refined_quad`)
5. Detect frets inside the refined region
6. Optionally tighten the fretboard boundary using detected frets
7. Track motion primarily with sparse Lucas-Kanade optical flow
8. Use ORB matching as fallback recovery if optical flow fails
9. Smooth the updated `refined_quad`
10. Periodically re-detect frets to improve a weak initial lock

Strings are calibrated separately after lock. They are not automatically calibrated during `lock()`.

## Core principles

1. Prefer stable tracked geometry over frame-by-frame re-detection.
2. Keep lock/calibration logic separate from tracking logic.
3. Use temporal smoothing carefully: more smoothing improves stability but increases lag.
4. Avoid unnecessary recalibration during normal tracking.
5. Preserve the current coordinate conventions for frets and strings.
6. Distinguish tracking issues from app-state or overlay-state issues.
7. Test both static and moving-guitar cases after meaningful tracking changes.

## Expected workflow

When changing tracking logic, follow this order:

1. Verify lock quality
   - preview ROI covers the fretboard reasonably
   - neck band detection is reasonable
   - `refined_quad` matches the visible fretboard
   - fret detection is plausible

2. Verify tracking update logic
   - `refined_quad` moves smoothly
   - optical-flow updates do not jump to background features
   - ORB recovery only activates when needed
   - weak updates are rejected

3. Verify fret and string model consistency
   - fret model remains plausible after motion
   - strings are calibrated separately and remain ordered correctly
   - overlay coordinates still match the tracked fretboard

4. Verify mode/reset behavior
   - reset invalidates stale tracking-dependent overlays
   - switching between chord mode and song mode does not reuse stale geometry incorrectly
   - stop-song and calibration reset behave cleanly

5. Evaluate smoothing vs responsiveness
   - if jitter is high, increase smoothing or tighten acceptance checks
   - if lag is high, reduce smoothing or avoid redundant heavy computation

## Recommended debugging checklist

When tracking is bad, inspect these first:
- Is the preview ROI appropriate for the current camera framing?
- Did `lock()` produce a valid `refined_quad`?
- Is the neck band detection reasonable?
- Are frets detected plausibly inside the refined region?
- Are string and fret relative models still valid after motion?
- Is smoothing applied only where useful?
- Is optical flow failing because track points were lost?
- Is ORB recovery searching in a reasonable area?
- Is old state being reused after reset or mode changes?
- Are overlays being drawn from stale model data?

## Common failure modes

### 1. Overlay jitter
Possible causes:
- noisy fret detection
- unstable tracked quad update
- weak optical-flow correspondences
- insufficient temporal smoothing

Possible fixes:
- smooth `refined_quad`
- reject weak updates
- refresh track points when needed
- reduce sensitivity to single-frame noise

### 2. Overlay lag
Possible causes:
- too much smoothing
- repeated heavy recomputation
- unnecessary re-detection during stable tracking

Possible fixes:
- reduce smoothing coefficient
- keep tracking lightweight
- recalibrate or refine only when needed

### 3. Drift off the fretboard
Possible causes:
- tracker locks onto wrong texture
- search region is too loose
- transform accepted despite weak evidence

Possible fixes:
- tighten the search region
- strengthen sanity checks on tracked geometry
- reject implausible transforms or low-inlier updates

### 4. Weak initial lock
Possible causes:
- poor ROI placement
- insufficient ORB features
- neck band detection not matching the visible fretboard
- weak fret detection during lock

Possible fixes:
- improve ROI assumptions
- improve neck-band preprocessing
- refine the boundary using fret evidence
- retry lock while holding the guitar still

### 5. Chords still visible after reset or mode change
Possible causes:
- song state not reset
- overlay state not cleared
- tracking state recreated but app state still holds chord data

Possible fixes:
- clear tracking and overlay-dependent state together
- clear song/chord mode state on reset
- verify route and stop-song behavior in `app.py` and `GuitarSystem.py`

## Implementation guidelines

- Prefer small, local changes over rewriting the whole tracker.
- Keep data flow explicit: preview ROI → locked quad → refined quad → fret/string models → overlay coordinates.
- Reuse existing helper functions and conventions when possible.
- Do not silently change fret or string numbering conventions.
- Keep drawing/debug logic separate from tracking logic where possible.
- When adding constants, document the tradeoff they control.

## When proposing code changes

Prefer solutions that:
- improve stability without adding heavy complexity
- preserve the optical-flow-first / ORB-recovery design
- keep UI behavior predictable
- preserve current reset and mode-switch behavior
- remain explainable in a student project presentation/report

## Validation after changes

After any change, test at least:
- preview mode before lock
- lock from ROI
- string calibration after lock
- calibration / learning mode
- single chord practice mode
- song practice mode
- switching from song mode back to calibration
- slight guitar movement
- temporary occlusion by the fretting hand
- stop-song and reset flows from the Flask app

A change is not complete unless tracking stability, overlay alignment, and reset/mode-switch behavior are all checked.