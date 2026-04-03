---
name: chord-visualization
description: Instructions for rendering chord overlays on the guitar, including mapping chord definitions to fret/string positions and drawing them correctly on the tracked neck. Use this when modifying how chords are displayed or aligned on the guitar.
---

# Chord Visualization

This skill is for tasks related to displaying chord positions on the guitar in a visually correct and stable way.

## Use this skill when

Use this skill when:
- drawing chord dots on the fretboard
- mapping chord definitions to screen coordinates
- debugging misaligned chord overlays
- working on single-chord practice mode
- working on song playback visualization
- improving visual clarity of chord markers

## Project context

Chord data typically comes from:
- `chords_data.py` (`CHORD_LIBRARY`, including finger positions and open/muted strings)
- `songs_data.py` (sequence of song chords and durations)

Rendering depends on:
- fret model (relative fret positions)
- string model (relative string positions)
- tracked neck bounding box or tracked corners

Chord lookup is normalized through `get_chord_data(chord_name)`, which currently uses `chord_name.upper()`. Keep chord naming consistent between `songs_data.py` and `CHORD_LIBRARY`.

## Core idea

A chord is defined in guitar coordinates:
- `(fret_number, string_number, finger_id)`

You must convert it to image coordinates on the current frame.

This mapping depends entirely on the current tracked guitar model.

## Mapping logic

Typical mapping pipeline:

1. Get current tracked neck bounds or corners
2. Use the relative fret model to compute the horizontal position
3. Use the relative string model to compute the vertical position
4. Combine them to get `(x, y)` for each finger marker

Important:
- This mapping must be recomputed from the current tracked model
- Do not cache absolute pixel positions across frames

## Rendering rules

When drawing chords:
- draw finger positions clearly at the target fret/string locations
- optionally differentiate:
  - pressed strings
  - open strings
  - muted strings
- keep visual encoding consistent across modes

## Stability principles

1. Visualization must follow tracking, not raw re-detection
2. Never draw based on stale geometry
3. Small tracking noise should not cause large visual jumps
4. Prefer stable tracked input over ad-hoc drawing fixes

## Common bugs

### 1. Chord markers drift off strings
Cause:
- wrong mapping from relative model
- stale tracked model

Fix:
- recompute positions from current tracked neck data
- verify string ordering

### 2. Chord appears shifted or mirrored
Cause:
- incorrect coordinate system
- string indexing mismatch

Fix:
- verify string numbering direction:
  - `1 = top string on screen`
  - `6 = bottom string on screen`
- verify neck orientation and left/right mapping

### 3. Chord dots lag behind guitar movement
Cause:
- cached positions
- drawing from old frame state

Fix:
- always compute positions from the current tracked model

### 4. Wrong fret placement
Cause:
- incorrect fret spacing model
- off-by-one error in fret indexing

Fix:
- visualize fret boundaries
- confirm that fret 1 means the region between the nut and the first fret

### 5. Chord name exists in song data but not in chord library
Cause:
- inconsistent naming or missing library entry

Fix:
- ensure every song chord exists in `CHORD_LIBRARY`
- keep chord keys consistent with the normalization used by `get_chord_data`

## Implementation guidelines

- Keep mapping logic separate from drawing logic
- Use a helper function for coordinate conversion
- Avoid mixing UI decisions with geometry calculations
- Use debug overlays to draw:
  - strings
  - frets
  - computed dot positions

## Validation checklist

After changes, verify:
- multiple chords from the library
- edge strings (1 and 6)
- different frets
- slight guitar motion
- switching between modes
- song chords that use lookup normalization correctly

A correct implementation should make chord markers feel attached to the guitar and consistent with the tracked neck geometry.