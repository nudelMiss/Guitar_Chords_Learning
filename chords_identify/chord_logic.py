class ChordIdentifier:
    def __init__(self):
        """
        Database format: 
        Key = Chord Name
        Value = Set of tuples: (Fret, String, Finger)
        
        Order changed to match hierarchy logic:
        1. Fret (Most important for basic note)
        2. String (Specific position)
        3. Finger (Technique)
        """
        self.chord_db = {
            "C": {
                (3, 5, 3), # Fret 3, String 5, Ring Finger
                (2, 4, 2), # Fret 2, String 4, Middle Finger
                (1, 2, 1)  # Fret 1, String 2, Index Finger 
            },
            "G": {
                (3, 6, 2), # Fret 3, String 6, Middle
                (2, 5, 1), # Fret 2, String 5, Index
                (3, 1, 3)  # Fret 3, String 1, Ring
            },
            "Am": {
                (2, 4, 2), # Fret 2, String 4, Middle
                (2, 3, 3), # Fret 2, String 3, Ring
                (1, 2, 1)  # Fret 1, String 2, Index
            },
             "E": {
                (2, 5, 2), # Fret 2, String 5, Middle
                (2, 4, 3), # Fret 2, String 4, Ring
                (1, 3, 1)  # Fret 1, String 3, Index
            },
            
            "D": {
                (2, 3, 1), # Fret 2, String 3, Index
                (3, 2, 3), # Fret 3, String 2, Ring
                (2, 1, 2)  # Fret 2, String 1, Middle
            },
            "A": {
                (2, 4, 1), # Fret 2, String 4, Index
                (2, 3, 2), # Fret 2, String 3, Middle
                (2, 2, 3)  # Fret 2, String 2, Ring
            },
            "Em": {
                (2, 5, 2), # Fret 2, String 5, Middle
                (2, 4, 3)  # Fret 2, String 4, Ring
            },
            "Dm": {
                (2, 3, 2), # Fret 2, String 3, Middle
                (3, 2, 3), # Fret 3, String 2, Ring
                (1, 1, 1)  # Fret 1, String 1, Index
            },
            "F": {
                (3, 4, 3), # Fret 3, String 4, Ring
                (2, 3, 2), # Fret 2, String 3, Middle
                (1, 2, 1), # Fret 1, String 2, Index (Barre part 1)
                (1, 1, 1)  # Fret 1, String 1, Index (Barre part 2)
            }
        }
        
        self.finger_names = {
            1: "Index",
            2: "Middle",
            3: "Ring",
            4: "Pinky",
            0: "Unknown"
        }

    def identify(self, detected_input):
            """
            Input: detected_input = List/Set of (Fret, String, Finger)
            Output: Dictionary with keys {'chord', 'status', 'message', 'corrections'}
            """
            
            # Parse Input Data
            input_full = set(detected_input)
            
            # NAME CHANGE: input_notes -> input_string_fret
            # Holds only the positions the user played (without finger ID)
            input_string_fret = {(f, s) for f, s, fing in detected_input} 
            
            input_frets = {f for f, s, fing in detected_input}
            
            best_result = {
                "chord": None, 
                "status": "unknown", 
                "message": "No chord detected.",
                "corrections": []
            }

            # NAME CHANGE: target_data -> correct_chord
            for chord_name, correct_chord in self.chord_db.items():
                
                # Prepare Target Data layers
                # NAME CHANGE: target_notes -> correct_string_fret
                correct_string_fret = {(f, s) for f, s, fing in correct_chord}
                
                target_frets = {f for f, s, fing in correct_chord}

                # --- LEVEL 3: PERFECT MATCH ---
                # Using 'correct_chord' instead of 'target_data'
                if correct_chord.issubset(input_full):
                    return {
                        "chord": chord_name,
                        "status": "perfect",
                        "message": f"Perfect! {chord_name} is played correctly.",
                        "corrections": []
                    }

                # --- LEVEL 2: WRONG FINGER ---
                # Using 'correct_string_fret' and 'input_string_fret'
                if correct_string_fret.issubset(input_string_fret):
                    
                    correction_list = []
                    
                    # Map target: (Fret, String) -> Finger
                    target_map = {(f, s): fing for f, s, fing in correct_chord}
                    
                    for f, s, user_finger in detected_input:
                        
                        if (f, s) in target_map:
                            needed_finger = target_map[(f, s)]
                            
                            if user_finger != needed_finger:
                                user_finger_name = self.finger_names.get(user_finger, str(user_finger))
                                needed_finger_name = self.finger_names.get(needed_finger, str(needed_finger))
                                
                                feedback = (
                                    f"On String {s} (Fret {f}): "
                                    f"You used {user_finger_name}, "
                                    f"but you need {needed_finger_name}."
                                )
                                correction_list.append(feedback)

                    return {
                        "chord": chord_name,
                        "status": "wrong_finger",
                        "message": f"{chord_name} detected, but check your finger positions.",
                        "corrections": correction_list
                    }

                # --- LEVEL 1: WRONG STRINGS ---
                if target_frets.issubset(input_frets):
                    best_result = {
                        "chord": chord_name,
                        "status": "wrong_string",
                        "message": f"Frets match {chord_name}, but strings do not. Check string positions.",
                        "corrections": []
                    }

            return best_result