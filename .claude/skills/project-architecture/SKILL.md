---
name: project-architecture
description: Overview and working conventions for the Guitar Chords Learning project. Use this when making changes that affect multiple files, app flow, state management, calibration flow, or integration between tracking, chord logic, hand detection, and the Flask frontend.
---

# Project Architecture

This skill explains how to work safely across the main parts of the Guitar Chords Learning project.

## Project purpose

This project helps beginners learn guitar chords using computer vision.

The system:
- locks and calibrates the guitar fretboard
- models strings and frets
- tracks the fretboard geometry over time
- overlays chord positions visually
- supports both single-chord practice and song practice
- can verify finger placement using hand landmarks

The project is designed around classical computer vision for guitar/fretboard tracking, plus lightweight application logic and MediaPipe-based hand detection.

## Main components

### 1. Tracking / geometry
Responsible for:
- preview ROI and fretboard lock
- fretboard boundaries
- string layout
- fret layout
- temporal fretboard tracking
- stable coordinate system for overlays

Typical file:
- `string_and_frets.py`

### 2. Main system logic
Responsible for:
- connecting tracking with chord display
- switching between learning mode and song mode
- maintaining chord and song state
- integrating hand detection with chord feedback
- per-frame processing
- reset behavior at the system level

Typical file:
- `GuitarSystem.py`

### 3. Web application / UI backend
Responsible for:
- Flask routes
- streaming frames
- frontend commands
- route-level resets
- communication between page state and backend system state

Typical file:
- `app.py`

### 4. Chord knowledge / song data
Responsible for:
- chord definitions
- finger placements
- open/muted string data
- song chord sequences
- chord naming and lookup

Typical files:
- `chords_data.py`
- `songs_data.py`

### 5. Hand/finger detection
Responsible for:
- hand landmark detection
- fingertip extraction
- choosing the fretting hand
- checking whether a chord is pressed correctly

Typical files:
- `hand_detector.py`
- `GuitarSystem.py`

## Working assumptions

When making changes, assume:
- geometry comes first
- overlays depend on a valid tracked and calibrated fretboard model
- UI state must stay synchronized with backend state
- resetting calibration should invalidate stale tracking and overlay state
- song mode and chord mode share infrastructure but must reset cleanly
- some bugs that look like CV bugs are actually state-management or streaming bugs

## Use this skill when

Use this skill when:
- a bug spans multiple files
- reset behavior is broken
- calibration mode and song mode interfere with each other
- chords continue drawing after mode switches
- backend/frontend actions do not match visible behavior
- route transitions produce stale visual state
- you need to modify logic without breaking the whole flow

## Recommended mental model

Think in layers:

1. **Geometry layer**
   - Where is the fretboard?
   - Where are the strings and frets?

2. **Tracking layer**
   - How does that geometry persist and update across frames?

3. **Application state layer**
   - What mode is active?
   - What chord or song step is active?
   - Has fretboard lock happened?
   - Have strings been calibrated?
   - Should overlays be visible right now?

4. **Rendering layer**
   - What is drawn on the current frame?

5. **Web/control layer**
   - Which route is active?
   - Which command was just sent?
   - Did the backend reset state consistently before the next streamed frame?

Many bugs happen when one layer resets and another does not.

## Common integration bugs

### 1. Chords still visible after leaving song mode
Likely cause:
- current song/chord state not cleared
- overlay conditions still satisfied
- stale tracked/calibrated geometry reused after mode change

### 2. Calibration runs but old model still affects output
Likely cause:
- old tracked quad or relative fret/string models not invalidated
- partial reset only
- app route changed without fully resetting dependent state

### 3. UI button appears to work but backend behavior does not change
Likely cause:
- frontend command sent but backend state not updated correctly
- backend state changed but streamed frame still uses stale values
- route reset and per-frame logic disagree

### 4. Tracking seems correct but chord markers are misplaced
Likely cause:
- stale fret/string model
- mismatch in coordinate convention
- wrong source of geometry used for drawing
- chord data lookup inconsistency

### 5. Song playback works but chord lookup fails
Likely cause:
- chord name exists in `songs_data.py` but not in `chords_data.py`
- inconsistent chord normalization between direct library access and `get_chord_data()`

## Safe change strategy

When modifying project flow:

1. Identify which layer the bug belongs to
   - geometry
   - tracking
   - state
   - rendering
   - web/control flow

2. Change as little as possible
   - avoid rewriting unrelated logic
   - preserve current conventions

3. Reset all dependent state together
   - if calibration resets, clear tracking-dependent overlay state too
   - if song mode stops, clear song progression and displayed chord state too
   - if a route changes mode, make sure the next streamed frame reflects that reset

4. Add temporary debug output if needed
   - current mode
   - current chord
   - song-playing state
   - tracking validity
   - string calibration state
   - overlay enabled/disabled

## Coding conventions for this project

- Keep state names explicit.
- Avoid hidden coupling between UI actions and drawing logic.
- Prefer helper methods for reset operations.
- Group related resets into dedicated functions rather than scattering assignments.
- When changing shared logic, test both main user flows:
  - single chord practice
  - song practice

## Good reset design

A reset operation should clearly answer:
- Is fretboard lock still valid?
- Is tracking still valid?
- Are strings still calibrated?
- Is a song running?
- Is a chord overlay active?
- Is hand feedback active?
- Should tracker lines be shown?
- Which mode should the next streamed frame use?

If these are not explicit, stale behavior is likely.

## Validation after changes

After any architecture-affecting change, test:
- app launch
- entering chord training
- entering song practice
- locking the fretboard
- calibrating strings
- entering single-chord mode
- starting song mode
- stopping song mode
- switching from song mode back to calibration/training
- resetting while overlays are visible
- verifying that the next streamed frame reflects the new state

A correct change should improve behavior without creating hidden state inconsistencies.