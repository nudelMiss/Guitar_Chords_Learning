# --- Chord Library Data Structure ---
# Each chord contains a list of fingers: (fret_number, string_number)
# Standard Tuning Convention:
# String 1: High E (Thinnest, bottom in standard diagrams)
# String 6: Low E (Thickest, top in standard diagrams)

CHORD_LIBRARY = {
    "C": {
        "fingers": [(3, 5), (2, 4), (1, 2)], # 3rd fret on 5th string, 2nd on 4th, 1st on 2nd
        "muted": [6],
        "open": [1, 3]
    },
    "G": {
        "fingers": [(3, 6), (2, 5), (3, 1)], # 3rd fret on 6th string, 2nd on 5th, 3rd on 1st
        "muted": [],
        "open": [2, 3, 4]
    },
    "D": {
        "fingers": [(2, 3), (3, 2), (2, 1)], # 2nd fret on 3rd string, 3rd on 2nd, 2nd on 1st
        "muted": [5, 6],
        "open": [4]
    },
    "AM": {
        "fingers": [(2, 4), (2, 3), (1, 2)], # 2nd fret on 4th string, 2nd on 3rd, 1st on 2nd
        "muted": [6],
        "open": [1, 5]
    },
    "E": {
        "fingers": [(2, 5), (2, 4), (1, 3)], # 2nd fret on 5th string, 2nd on 4th, 1st on 3rd
        "muted": [],
        "open": [1, 2, 6]
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