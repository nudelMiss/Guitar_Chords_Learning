---
name: hand-finger-detection
description: Instructions for working on fretting-hand and fingertip detection for chord verification in this project. Use this when modifying fingertip detection, MediaPipe hand landmarks, fretting-hand selection, or logic that checks whether fingers press the correct chord positions.
---

# Hand and Finger Detection

This skill is for tasks related to detecting the fretting hand and determining whether fingers are placed correctly for a target chord.

## Use this skill when

Use this skill when the task involves:
- detecting the player's fretting hand
- fingertip localization
- selecting the correct hand near the fretboard
- checking whether a chord is pressed correctly
- reducing false positives from the wrong hand or background
- improving robustness of fingertip-based chord verification
- modifying MediaPipe-based hand landmark logic

## Project context

The goal of hand detection in this project is not general hand pose estimation. The goal is narrower:

- find the relevant fretting-hand fingertips
- choose the hand closest to the fretboard
- determine whether fingertips are close to expected chord target points
- provide useful feedback for chord learning

This means solutions should be optimized for the guitar-learning use case, not for generic full-hand understanding.

Relevant files usually include:
- `hand_detector.py`
- `GuitarSystem.py`
- `string_and_frets.py`

## Current implementation

The current project uses a MediaPipe-based hand detector.

### Hand detection
`hand_detector.py` currently:
- uses MediaPipe Tasks Hand Landmarker
- detects hand landmarks in the full frame
- extracts fingertips for:
  - index
  - middle
  - ring
  - pinky
- does not use the thumb in the main fingertip set

### Fretting-hand selection
`GuitarSystem.py` currently:
- detects all visible hands
- chooses the hand whose fingertips are most inside / near the fretboard area
- uses an expanded fretboard bounding box to score each hand

### Chord verification
The current chord check is point-based:
- expected chord finger positions are converted into image coordinates using `get_dot_coordinates(...)`
- each expected target point is compared to detected fingertips
- a finger counts as correct if a matching fingertip is close enough to the target point
- if a chord definition includes a required finger id, the correct finger must match that target

This is currently simpler than full fret-zone or string-zone reasoning.

## Core principles

1. Favor reliable fingertip localization over full hand complexity.
2. Use the tracked fretboard geometry to simplify hand reasoning.
3. Select the correct fretting hand before evaluating chord correctness.
4. Judge correctness relative to expected chord target points, not only raw hand presence.
5. Reduce false detections from the wrong hand, guitar body, and background.

## Current reasoning pipeline

A good description of the current pipeline is:

1. Track the fretboard
2. Detect all hands in the frame with MediaPipe
3. Extract fingertip landmarks for each hand
4. Choose the hand closest to the fretboard
5. Compute expected chord target points on the fretboard
6. Compare fingertips to expected target points
7. Draw feedback based on correct / incorrect fingertip placement

## MediaPipe guidance

When modifying the current system:
- treat MediaPipe as the landmark provider
- keep chord-checking logic separate from hand detection
- use fretboard geometry to choose the relevant hand
- reject detections from hands far from the fretboard
- keep the project explanation clear: MediaPipe landmarks + project-specific chord verification

## Current limitations to keep in mind

- The main chord-verification flow currently uses full-frame hand detection, not fretboard-only ROI detection.
- The active chord-checking flow currently uses `detect_all_hands_fingertips()`, which does not apply per-hand EMA smoothing.
- Correctness is currently based on distance to expected target points, not explicit string-band / fret-zone membership.
- The fingertip distance threshold is currently fixed in `GuitarSystem.py`.

If you improve the logic, preserve compatibility with the current project structure unless the task explicitly asks for a redesign.

## Classical CV guidance

Classical CV approaches are not the current main implementation. Only use them if the task explicitly asks to replace or augment the MediaPipe-based detector.

If exploring a classical CV alternative:
- use ROI restriction aggressively
- exploit contrast between hand and fretboard when stable
- validate detections against fretboard geometry
- prefer simple, interpretable methods over brittle pipelines

Do not assume thresholding alone will be robust across lighting changes.

## Common failure modes

### 1. Detecting the wrong hand
Possible causes:
- full-frame detection returns multiple hands
- fretting-hand selection is too weak
- fretboard bounding area is too loose

Fix ideas:
- improve hand scoring relative to the fretboard
- tighten or better shape the fretboard proximity logic
- consider hand-side or wrist-position heuristics if needed

### 2. Correct chord marked as wrong
Possible causes:
- fingertip threshold too strict
- wrong chord target coordinates
- wrong finger-id matching
- mismatch between tracked fretboard and overlay targets

Fix ideas:
- visualize fingertips and target points together
- tune the distance threshold
- verify finger numbering conventions
- verify `get_dot_coordinates(...)` output carefully

### 3. Fingertips flicker or jump
Possible causes:
- unstable landmarks
- no smoothing in multi-hand detection path
- temporary hand occlusion

Fix ideas:
- add per-hand temporal smoothing
- require consistency across frames
- use debug overlays to inspect jitter directly

### 4. Background or non-fretting hand influences results
Possible causes:
- wrong hand selected
- all hands treated equally
- no additional filtering after landmark detection

Fix ideas:
- improve fretting-hand selection
- reject hands far from the fretboard
- compare only the chosen hand against the chord

## Implementation guidelines

- Always visualize detections during debugging.
- Draw detected fingertips and expected chord target points together.
- Keep hand detection logic separate from chord scoring logic.
- Document assumptions such as finger naming and string numbering.
- If adding thresholds, make them named constants.
- Optimize for stable and believable feedback in a live demo.

## Validation after changes

After any change, test:
- several different chords
- adjacent-string fingerings
- one easy open chord and one denser fingering
- slight hand motion
- multiple hands visible in frame
- partial occlusion
- lighting variation if possible

The best solution is not the fanciest detector. It is the one that gives stable and believable chord feedback during live use.