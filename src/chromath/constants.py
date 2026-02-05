from enum import Enum
from dataclasses import dataclass
import numpy as np

IONIAN_SEMITONES: list[int] = [0, 2, 4, 5, 7, 9, 11]


@dataclass(frozen=True)
class DefaultMusicSystem:
    tones: int = 12
    tones_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
    degrees: int = 7
    degrees_set = {0, 1, 2, 3, 4, 5, 6, 7}
    func_degrees: int = 13  # degrees meaningful for intervals
    func_degrees_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    # bitwise representation of tones
    powers: tuple[int] = tuple(2**i for i in range(12))
    # max bitwise representation
    max_int_repr: int = 2**12


MS = DefaultMusicSystem


class Letter(Enum):
    C = 1
    D = 2
    E = 3
    F = 4
    G = 5
    A = 6
    B = 7


intervals = np.array(
    [
        [1, 0],  # P0
        [2, 1],  # m2
        [2, 2],  # M2
        [2, 3],  # aug2
        [3, 3],  # m3
        [3, 4],  # M3
        [4, 5],  # P4
        [4, 6],  # aug4
        [5, 6],  # b5
        [5, 7],  # P5
        [5, 8],  # aug5
        [6, 8],  # m6
        [6, 9],  # M6
        [6, 10],  # aug6
        [7, 9],  # b7
        [7, 10],  # m7
        [7, 11],  # M7
        [8, 0],  # P8
        [9, 1],  # m9
        [9, 2],  # M9
        [9, 3],  # aug9
        [10, 3],  # m10
        [10, 4],  # M10
        [11, 5],  # P11
        [11, 6],  # aug11
        [13, 8],  # m13
        [13, 9],  # M13
    ]
)
# fmt: off
interval_names = np.array([
    "P0","m2","M2","aug2","m3","M3","P4","aug4","b5","P5","aug5",
    "m6","M6","aug6","b7","m7","M7","P8","m9","M9","aug9",
    "m10","M10","P11","aug11","m13","M13"
])
# fmt: on


# class ChordShape(Enum):
#     """
#     Class to store all the classical chord shapes
#     """

#     # Triads (most common)
#     M = {Interval.P0, Interval.M3, Interval.P5}  # Major
#     m = {Interval.P0, Interval.m3, Interval.P5}  # Minor
#     aug = {Interval.P0, Interval.M3, Interval.aug5}  # Augmented
#     dim = {Interval.P0, Interval.m3, Interval.b5}  # Diminished

#     # Seventh chords (very common)
#     M7 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7}  # Major 7
#     m7 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7}  # Minor 7
#     dom7 = {Interval.P0, Interval.M3, Interval.P5, Interval.m7}  # Dominant 7
#     mM7 = {Interval.P0, Interval.m3, Interval.P5, Interval.M7}  # Minor Major 7
#     aug7 = {Interval.P0, Interval.M3, Interval.aug5, Interval.m7}  # Augmented 7
#     dim7 = {Interval.P0, Interval.m3, Interval.b5, Interval.b7}  # Diminished 7
#     halfdim7 = {Interval.P0, Interval.m3, Interval.b5, Interval.m7}  # Half-diminished 7

#     # Sixth chords (common)
#     M6 = {Interval.P0, Interval.M3, Interval.P5, Interval.M6}  # Major 6
#     m6 = {Interval.P0, Interval.m3, Interval.P5, Interval.M6}  # Minor 6
#     Mb6 = {Interval.P0, Interval.M3, Interval.P5, Interval.m6}  # Major with minor 6
#     mb6 = {Interval.P0, Interval.m3, Interval.P5, Interval.m6}  # Minor with minor 6

#     # Suspended chords (common)
#     sus2 = {Interval.P0, Interval.M2, Interval.P5}  # Suspended 2
#     sus4 = {Interval.P0, Interval.P4, Interval.P5}  # Suspended 4
#     sus47 = {Interval.P0, Interval.P4, Interval.P5, Interval.m7}  # Suspended 4 with 7

#     # Add9 and extended chords
#     add9 = {Interval.P0, Interval.M3, Interval.P5, Interval.M9}  # Major add 9
#     madd9 = {Interval.P0, Interval.m3, Interval.P5, Interval.M9}  # Minor add 9
#     M9 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.M9}  # Major 9
#     m9 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7, Interval.M9}  # Minor 9
#     dom9 = {
#         Interval.P0,
#         Interval.M3,
#         Interval.P5,
#         Interval.m7,
#         Interval.M9,
#     }  # Dominant 9

#     # 11th chords
#     M11 = {
#         Interval.P0,
#         Interval.M3,
#         Interval.P5,
#         Interval.M7,
#         Interval.M9,
#         Interval.P11,
#     }  # Major 11
#     m11 = {
#         Interval.P0,
#         Interval.m3,
#         Interval.P5,
#         Interval.m7,
#         Interval.M9,
#         Interval.P11,
#     }  # Minor 11
#     dom11 = {
#         Interval.P0,
#         Interval.M3,
#         Interval.P5,
#         Interval.m7,
#         Interval.M9,
#         Interval.P11,
#     }  # Dominant 11

#     # 13th chords
#     M13 = {
#         Interval.P0,
#         Interval.M3,
#         Interval.P5,
#         Interval.M7,
#         Interval.M9,
#         Interval.M13,
#     }  # Major 13
#     m13 = {
#         Interval.P0,
#         Interval.m3,
#         Interval.P5,
#         Interval.m7,
#         Interval.M9,
#         Interval.m13,
#     }  # Minor 13
#     dom13 = {
#         Interval.P0,
#         Interval.M3,
#         Interval.P5,
#         Interval.m7,
#         Interval.M9,
#         Interval.M13,
#     }  # Dominant 13

#     # Power chords (common in rock)
#     P5 = {Interval.P0, Interval.P5}  # Power chord (no third)

#     # Flat 5 / sharp 5 variations
#     augM7 = {Interval.P0, Interval.M3, Interval.aug5, Interval.M7}  # Augmented Major 7
#     dimM7 = {Interval.P0, Interval.m3, Interval.b5, Interval.M7}  # Diminished Major 7

#     # Omitted chord variations
#     M_no5 = {Interval.P0, Interval.M3}  # Major with no 5
#     m_no5 = {Interval.P0, Interval.m3}  # Minor with no 5

#     aug6 = {Interval.P0, Interval.M3, Interval.P5, Interval.aug6}

#     # Jazz and advanced
#     aug9 = {Interval.P0, Interval.M3, Interval.aug5, Interval.M9}  # Augmented 9
#     M7sharp11 = {
#         Interval.P0,
#         Interval.M3,
#         Interval.P5,
#         Interval.M7,
#         Interval.aug11,
#     }  # Lydian
#     # Half-diminished (alternate spelling)
#     m7b5 = {Interval.P0, Interval.m3, Interval.b5, Interval.m7}

#     # Cluster / polychords (less common, added tones)
#     M7add13 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.M13}
#     m7add13 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7, Interval.m13}

#     # Extended suspensions
#     sus2sus4 = {Interval.P0, Interval.M2, Interval.P4, Interval.P5}  # Sus2 Sus4

#     # Empty/Unknown
#     UNKNOWN = set()

#     @classmethod
#     def from_intervals(cls, intervals: list[Interval] | set[Interval]):
#         target_semitones = {i.functional_degree % 12 for i in intervals}
#         for shape in cls:
#             shape_semitones = {i.functional_degree % 12 for i in shape.intervals}
#             if target_semitones == shape_semitones:
#                 return shape
#         return cls.UNKNOWN

#     def __str__(self):
#         return self.name

#     def __repr__(self):
#         return self.name

#     @property
#     def intervals(self) -> set[Interval]:
#         return self.value
