# --- Chord Library Data Structure Updated ---
# (fret_number, string_number, finger_id)
# finger_id: 1=Index, 2=Middle, 3=Ring, 4=Pinky

CHORD_LIBRARY = {
    "C": {
        "fingers": [(1, 5, 1), (2, 3, 2), (3, 2, 3)],
        "muted": [1],
        "open": [4, 6]
    },
    "G": {
        "fingers": [(3, 1, 2), (2, 2, 1), (3, 6, 3)],
        "muted": [],
        "open": [3, 4, 5]
    },
    "D": {
        "fingers": [(2, 4, 1), (3, 5, 3), (2, 6, 2)],
        "muted": [1, 2],
        "open": [3]
    },
    "Am": {
        "fingers": [(1, 5, 1), (2, 3, 2), (2, 4, 3)],
        "muted": [1],
        "open": [2, 6]
    },
    "E": {
        "fingers": [(1, 4, 1), (2, 2, 2), (2, 3, 3)],
        "muted": [],
        "open": [1, 5, 6]
    }
}

def get_chord_data(chord_name):
    return CHORD_LIBRARY.get(chord_name.upper(), None)

def list_available_chords():
    return list(CHORD_LIBRARY.keys())