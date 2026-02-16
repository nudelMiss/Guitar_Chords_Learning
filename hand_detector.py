import typing

class HandDetector:
    """Stub HandDetector for hand-detection responsibilities.

    Implement these methods to provide real detection. The current
    implementations are placeholders so the rest of the pipeline runs.
    """
    def __init__(self):
        self.frets = []
        self.strings = []

    def calibrate_frets(self, frets_x: typing.List[float]):
        self.frets = list(frets_x)

    def calibrate_strings(self, strings_y: typing.List[float]):
        self.strings = list(strings_y)

    def detect_fingers_and_frets(self, frame):
        """Detect fingers and map them to (fret, string, finger_id).

        Returns: (num_fingers, list_of_triples)
        - num_fingers: int
        - list_of_triples: [(fret:int, string:int, finger:int), ...]

        Current behavior: returns empty detection. Replace with real logic.
        """
        return 0, []
