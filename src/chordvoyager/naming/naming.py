"""
A file containing the logic to compute proper names and user-faced information.
This is a complex matter as composers themselves sometimes struggle with it.
For now, we have simplified functions that do not depend on context.
"""

from ..constants import Letter

SHARP_NAMES = {
    0: "C",
    1: "C#",
    2: "D",
    3: "D#",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "G#",
    9: "A",
    10: "A#",
    11: "B",
}

FLAT_NAMES = {
    0: "C",
    1: "Db",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "Gb",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


def get_pitch_name(semitones: int, prefer_sharp: bool = True) -> str:
    if prefer_sharp:
        return SHARP_NAMES[semitones]
    return FLAT_NAMES[semitones]


class PitchName:
    """
    Class to better handle traditional pitch notation, with letters and accidentals.
    Not used for now, depends on the tonality and accidentals.
    """

    degree: int
    accidental: int
    # tonality: Tonality
    # TODO: Refactor depending on scale.
    # TODO: Adjust depending on the considered tonality.

    def __init__(self):
        self.letter: Letter = Letter(self.degree)

    def __str__(self):
        a = self.accidental
        acc_string = "#" * a if a > 0 else "b" * abs(a)
        return f"{self.letter.name}{acc_string}"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__str__()}>"
