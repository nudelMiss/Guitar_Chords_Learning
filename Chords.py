# --- Chord Library Data Structure ---
# Each chord contains a list of fingers: (fret_number, string_number)
# String 1: Top string in camera view
# String 6: Bottom string in camera view

CHORD_LIBRARY = {
    "C": {
        "fingers": [(1, 5), (2, 3), (3, 2)],  # flipped strings
        "muted": [1],
        "open": [4, 6]
    },
    "G": {
        "fingers": [(3, 1), (2, 2), (3, 6)],
        "muted": [],
        "open": [3, 4, 5]
    },
    "D": {
        "fingers": [(2, 4), (3, 5), (2, 6)],
        "muted": [1, 2],
        "open": [3]
    },
    "Am": {
        "fingers": [(1, 5), (2, 3), (2, 4)],
        "muted": [1],
        "open": [2, 6]
    },
    "E": {
        "fingers": [(1, 4), (2, 2), (2, 3)],
        "muted": [],
        "open": [1, 5, 6]
    }
}


def get_chord_data(chord_name):
    """
    Returns the finger positions for a given chord name.
    If the chord is not found, returns None.
    """
    return CHORD_LIBRARY.get(chord_name.upper(), None)


def list_available_chords():
    """
    Returns a list of all chords currently in the library.
    """
    return list(CHORD_LIBRARY.keys())