# Dictionary containing all chord structures
# Format for fingers: (finger_number, string_number, fret_number)
# Open and muted strings are lists of string numbers (1-6)
CHORDS_DICT = {
    "Am": {
        "fingers": [(1, 2, 1), (2, 4, 2), (3, 3, 2)],
        "open": [1, 5],
        "muted": [6]
    },
    "Em": {
        "fingers": [(2, 5, 2), (3, 4, 2)],
        "open": [1, 2, 3, 6],
        "muted": []
    },
    "G": {
        "fingers": [(1, 5, 2), (2, 6, 3), (3, 1, 3)],
        "open": [2, 3, 4],
        "muted": []
    },
    "C": {
        "fingers": [(1, 2, 1), (2, 4, 2), (3, 5, 3)],
        "open": [1, 3],
        "muted": [6]
    },
    "Dm": {
        "fingers": [(1, 1, 1), (2, 3, 2), (3, 2, 3)],
        "open": [4],
        "muted": [5, 6]
    },
    "D": {
        "fingers": [(1, 3, 2), (2, 1, 2), (3, 2, 3)],
        "open": [4],
        "muted": [5, 6]
    },
    "F": {
        # Using a mini-F shape to avoid full barre complexity for now
        "fingers": [(1, 2, 1), (2, 3, 2), (3, 5, 3), (4, 4, 3)], 
        "open": [],
        "muted": [1, 6]
    },
    "A7sus4": {
        "fingers": [(2, 4, 2), (3, 2, 3)], 
        "open": [1, 3, 5],
        "muted": [6]
    }
}

def get_chord_data(chord_name):
    # Safely return chord data or None if it does not exist to prevent crashes
    return CHORDS_DICT.get(chord_name, None)

def list_available_chords():
    # Return a list of all available chord names for the UI
    return list(CHORDS_DICT.keys())