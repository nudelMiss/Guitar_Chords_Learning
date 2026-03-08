# Format: (fret_number, string_number, finger_id)
# String numbering:
# 6 = top string on screen
# 1 = bottom string on screen

CHORD_LIBRARY = {
    "C": {
        "fingers": [(1, 2, 1), (2, 4, 2), (3, 5, 3)],
        "muted": [6],
        "open": [1, 3]
    },
    "G": {
        "fingers": [(2, 5, 1), (3, 6, 2), (3, 1, 3)],
        "muted": [],
        "open": [2, 3, 4]
    },
    "D": {
        "fingers": [(2, 3, 1), (2, 1, 2), (3, 2, 3)],
        "muted": [5, 6],
        "open": [4]
    },
    "Am": {
        "fingers": [(1, 2, 1), (2, 4, 2), (2, 3, 3)],
        "muted": [6],
        "open": [1, 5]
    },
    "E": {
        "fingers": [(1, 3, 1), (2, 5, 2), (2, 4, 3)],
        "muted": [],
        "open": [1, 2, 6]
    },
    "Em": {
        "fingers": [(2, 5, 2), (2, 4, 3)],
        "muted": [],
        "open": [1, 2, 3, 6]
    },
    "Dm": {
        "fingers": [(1, 1, 1), (2, 3, 2), (3, 2, 3)],
        "muted": [5, 6],
        "open": [4]
    },
    "A": {
        "fingers": [(2, 4, 1), (2, 3, 2), (2, 2, 3)],
        "muted": [6],
        "open": [1, 5]
    }
}

def get_chord_data(chord_name):
    if chord_name is None:
        return None
    return CHORD_LIBRARY.get(chord_name.upper(), None)

def list_available_chords():
    return list(CHORD_LIBRARY.keys())