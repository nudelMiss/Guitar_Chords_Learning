from chord_logic import ChordIdentifier

def print_result(test_name, result):
    """Helper function to print results cleanly in the terminal"""
    print(f"\n=== {test_name} ===")
    print(f"Chord:    {result['chord']}")
    print(f"Status:   {result['status']}")
    print(f"Message:  {result['message']}")
    
    if result['corrections']:
        print("Corrections needed:")
        for note in result['corrections']:
            print(f" [!] {note}")
    print("-" * 50)

def run_tests():
    identifier = ChordIdentifier()

    # ---------------------------------------------------------
    # TEST 1: D Major (Perfect)
    # ---------------------------------------------------------
    # Scenario: User plays perfectly.
    # D Major: Fret 2/Str 3 (Index), Fret 3/Str 2 (Ring), Fret 2/Str 1 (Middle)
    input_d_major = [
        (2, 3, 1), 
        (3, 2, 3), 
        (2, 1, 2)
    ]
    result = identifier.identify(input_d_major)
    print_result("Test 1: D Major (Perfect Execution)", result)


    # ---------------------------------------------------------
    # TEST 2: A Minor (Wrong Finger)
    # ---------------------------------------------------------
    # Scenario: User presses the right frets/strings but uses wrong fingers.
    # A Minor requires: Middle(2), Ring(3), Index(1).
    # User uses: Index(1) for everything (common beginner mistake).
    input_am_bad_fingers = [
        (2, 4, 1), # Wrong: Used Index, needed Middle
        (2, 3, 1), # Wrong: Used Index, needed Ring
        (1, 2, 1)  # Correct
    ]
    result = identifier.identify(input_am_bad_fingers)
    print_result("Test 2: A Minor (Right Notes, Wrong Fingers)", result)


    # ---------------------------------------------------------
    # TEST 3: E Major Shape (Wrong Strings)
    # ---------------------------------------------------------
    # Scenario: User makes the E Major shape (Frets 2,2,1)
    # but places it on strings 3,2,1 (instead of 5,4,3).
    input_em_wrong_strings = [
        (2, 3, 2), 
        (2, 2, 3), 
        (1, 1, 1)
    ]
    result = identifier.identify(input_em_wrong_strings)
    print_result("Test 3: E Major Shape on Wrong Strings", result)


    # ---------------------------------------------------------
    # TEST 4: F Major (Mini-Barre)
    # ---------------------------------------------------------
    # Scenario: Testing the logic for chords where one finger holds two strings.
    # F Major: Index finger holds both String 1 and String 2 at Fret 1.
    input_f_major = [
        (3, 4, 3), # Ring
        (2, 3, 2), # Middle
        (1, 2, 1), # Index (Barre part 1)
        (1, 1, 1)  # Index (Barre part 2)
    ]
    result = identifier.identify(input_f_major)
    print_result("Test 4: F Major (Complex / Mini-Barre)", result)


    # ---------------------------------------------------------
    # TEST 5: Random Noise
    # ---------------------------------------------------------
    # Scenario: User is just moving fingers randomly.
    input_noise = [(1, 6, 1), (4, 4, 2)]
    result = identifier.identify(input_noise)
    print_result("Test 5: Random Noise", result)

if __name__ == "__main__":
    run_tests()