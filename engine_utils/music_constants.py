from __future__ import annotations
from functools import cached_property
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from music_objects import Note

import engine_utils.computation_utils as utils
import numpy as np

A4_MIDI_VALUE = 69
A4_FREQ = 440.0
TUNING = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)  # Chromatic scale semitones
IONIAN_SEMITONES = [0, 2, 4, 5, 7, 9, 11]

def pitch_distance(p1: int | Pitch, p2: int | Pitch) -> int:
    """ Distance between two notes (defined by semitones)
    within the 12-tones equal temperament. Values in [0-6]"""
    if isinstance(p1, Pitch): p1 = p1.sounding_pitch
    if isinstance(p2, Pitch): p2 = p2.sounding_pitch
    if not 0 <= p1 <= 11 or not 0 <= p2 <= 11: raise ValueError(f'Pitches must be between 0 and 11 - got {p1}, {p2}')
    diff = abs(p2 - p1) % 12
    return diff if diff <= 6 else 12 - diff


# NOTE: l'utilité de cette classe est peut-être limitée. Peut être géré au niveau de Note.
# D'un autre côté, c'est la classe qui représente la similitude par octave des notes.
@dataclass
class Pitch:
    absolute_pitch: int
    accidentals: int = 0

    def __post_init__(self):
        if not 0 <= self.absolute_pitch <= 11:
            self.absolute_pitch = self.absolute_pitch % 12

    @cached_property
    def sounding_pitch(self): return (self.absolute_pitch + self.accidentals) % 12

    @cached_property
    def is_altered(self): return self.accidentals != 0

    def __sub__(self, other: Pitch | int):
        if isinstance(other, Pitch):
            return Pitch(self.absolute_pitch - other.absolute_pitch, self.accidentals - other.accidentals)
        return Pitch((self.absolute_pitch - other) % 12, self.accidentals)

    def __add__(self, other: Pitch | int):
        if isinstance(other, Pitch):
            return Pitch((self.absolute_pitch + other.absolute_pitch) % 12, self.accidentals + other.accidentals)
        return Pitch((self.absolute_pitch + other) % 12, self.accidentals)

    def shift(self, semitones): return Pitch(self.absolute_pitch + semitones, self.accidentals)
    def alter(self, accidentals): return Pitch(self.absolute_pitch, self.accidentals + accidentals)

# NOTE: Notes will handle octaves -> 9th, 13th etc. Extended Intervals
# Ici : attention à la confusion. Semitones = pitch ? J'enregistre 2 semitones dans l'intervalle ou 14 ?
# Pour moi, ... 14 sera déduit au niveau de l'interprétation de l'intervalle
# (degré > 7 => +12 dans les semitones - ou pas selon le mode de génération de l'accord)
@dataclass(frozen=True, order=True)
class Interval:
    functional_degree: int
    pitch: int

    def __post_init__(self):
        if not 1 <= self.functional_degree <= 13:
            raise ValueError(f'Functional degree must be between 1 and 13, got {self.functional_degree}')
        if not 0 <= self.pitch <= 11:
            raise ValueError(f'Pitch must be between 0 and 11, got {self.pitch}')

    @cached_property
    def degree(self): return self.functional_degree % 7

    @cached_property
    def dissonance(self): return utils.dissonance([0, self.functional_degree])


class Intervals:
    P0    = (0, 1)
    m2    = (1, 2)
    M2    = (2, 2)
    aug2  = (3, 2)
    m3    = (3, 3)
    M3    = (4, 3)
    P4    = (5, 4)
    aug4  = (6, 4)
    b5    = (6, 5)
    P5    = (7, 5)
    aug5  = (8, 5)
    m6    = (8, 6)
    M6    = (9, 6)
    b7    = (9, 7)
    aug6  = (10, 6)
    m7    = (10, 7)
    M7    = (11, 7)
    P8    = (0, 8)
    m9    = (1, 9)
    M9    = (2, 9)
    aug9  = (3, 9)
    m10   = (3, 10)
    M10   = (4, 10)
    P11   = (5, 11)
    aug11 = (6, 11)
    m13   = (7, 13)
    M13   = (8, 13)

    def __str__(self): return self.__class__.__name__
    def __repr__(self): return self.__str__()
    def is_known_interval(self, iv: Interval): return iv in vars(Interval).values()


@dataclass(frozen=True)
class Cycle:
    step: int
    start_pitch: int = 0

    def __post_init__(self):
        if not 1 <= self.step <= 12:
            raise ValueError('Step must be between 1 and 12:', self.step)

    def __getitem__(self, index: int): return (self.step * index + self.start_pitch) % 12

    @cached_property
    def periodicity(self): return 12 // self.step if 12 % self.step == 0 else 12

    @cached_property
    def positions(self) -> list[None | int]:
        """ Gives for each pitch in the cycle, its position in the cycle.
        Negative or Positive. None if the pitch is not in the cycle. """
        output: list[None | int] = [None] * 12
        # self[i] nous donne le prochain élément du cycle en semitones, donc l'index.
        # Ensuite, il s'agit de vérifier s'il est plus court de le trouver dans le côté négatif ou positif du cycle.
        # par périodicité, self[i] = self[i - self.periodicity]
        for i in range(self.periodicity):
            positive_is_shortest = i <= -(i - self.periodicity)
            output[self[i]] = i if positive_is_shortest else i - \
                self.periodicity
        return output

    @cached_property
    def mask(self): return [0 if pos is None else 1 for pos in self.positions]

    def from_pitch(self, pitch: int): return replace(self, start_pitch=pitch)


class Cycles:
    FIFTH = Cycle(7)
    MAJOR_THIRD = Cycle(4)
    MINOR_THIRD = Cycle(3)
    MAJOR_SECOND = Cycle(2)
    MINOR_SECOND = Cycle(1)


@dataclass(frozen=True)
class Scale:
    degree_semitones: list[int]

    def __post_init__(self):
        if len(self.degree_semitones) != 7:
            raise ValueError(f'Need 7 notes, got {len(self.degree_semitones)}')
        for st in self.degree_semitones:
            if not 0 <= st <= 11:
                raise ValueError(f'Semitones must be between 0 and 11, got {self.degree_semitones}')

    def __str__(self): return f"Scale({self.degree_semitones})"
    def __repr__(self): return f"<{self.__class__.__name__}: {self.__str__()}>"
    def __getitem__(self, deg: int): return self.degree_semitones[(deg - 1) % 7]
    def __len__(self): return len(self.degree_semitones)

    @classmethod
    def from_mask(cls, note_mask: list[int]): return Scale(np.flatnonzero(note_mask).tolist())

    @classmethod
    def from_semitones_diff(cls, st_diff: list[int]): return Scale(np.cumsum(st_diff).tolist())

    @cached_property
    def semitones_diff(self) -> list[int]:
        return np.diff(self.degree_semitones, prepend=self.degree_semitones[-1] - 12).tolist()

    @cached_property
    def mask(self) -> list[int]:
        mask = np.zeros(12, dtype=int)
        mask[self.degree_semitones] = 1
        return mask.tolist()

    def shift(self, start_degree: int) -> Scale:
        new_semitones = np.roll(self.degree_semitones, start_degree % 7)
        new_semitones = (new_semitones - new_semitones[0]) % 12
        return Scale(new_semitones.tolist())

    def transpose(self, start_pitch: int):
        return Scale([(st + start_pitch) % 12 for st in self.degree_semitones])


class Scales:
    IONIAN = Scale(IONIAN_SEMITONES)
    DORIAN = Scale(IONIAN_SEMITONES).shift(2)
    PHRYGIAN = Scale(IONIAN_SEMITONES).shift(4)
    LYDIAN = Scale(IONIAN_SEMITONES).shift(5)
    MIXOLYDIAN = Scale(IONIAN_SEMITONES).shift(7)
    AEOLIAN = Scale(IONIAN_SEMITONES).shift(9)
    LOCRIAN = Scale(IONIAN_SEMITONES).shift(11)
    MINOR_HARM = Scale([0, 2, 3, 5, 7, 8, 11])
    MINOR_ASC = Scale([0, 2, 3, 5, 7, 9, 11])



class ChordShape(Enum):
    # Triads (most common)
    M = {Intervals.P0, Intervals.M3, Intervals.P5}      # Major
    m = {Intervals.P0, Intervals.m3, Intervals.P5}      # Minor
    aug = {Intervals.P0, Intervals.M3, Intervals.aug5}  # Augmented
    dim = {Intervals.P0, Intervals.m3, Intervals.b5}    # Diminished

    # Seventh chords (very common)
    M7 = {Intervals.P0, Intervals.M3,
          Intervals.P5, Intervals.M7}      # Major 7
    m7 = {Intervals.P0, Intervals.m3,
          Intervals.P5, Intervals.m7}      # Minor 7
    dom7 = {Intervals.P0, Intervals.M3,
            Intervals.P5, Intervals.m7}    # Dominant 7
    mM7 = {Intervals.P0, Intervals.m3, Intervals.P5,
           Intervals.M7}     # Minor Major 7
    aug7 = {Intervals.P0, Intervals.M3,
            Intervals.aug5, Intervals.m7}  # Augmented 7
    dim7 = {Intervals.P0, Intervals.m3, Intervals.b5,
            Intervals.b7}    # Diminished 7
    halfdim7 = {Intervals.P0, Intervals.m3,
                Intervals.b5, Intervals.m7}  # Half-diminished 7

    # Sixth chords (common)
    M6 = {Intervals.P0, Intervals.M3,
          Intervals.P5, Intervals.M6}      # Major 6
    m6 = {Intervals.P0, Intervals.m3,
          Intervals.P5, Intervals.M6}      # Minor 6
    Mb6 = {Intervals.P0, Intervals.M3, Intervals.P5,
           Intervals.m6}      # Major with minor 6
    mb6 = {Intervals.P0, Intervals.m3, Intervals.P5,
           Intervals.m6}      # Minor with minor 6

    # Suspended chords (common)
    sus2 = {Intervals.P0, Intervals.M2,
            Intervals.P5}     # Suspended 2
    sus4 = {Intervals.P0, Intervals.P4,
            Intervals.P5}     # Suspended 4
    sus47 = {Intervals.P0, Intervals.P4, Intervals.P5,
             Intervals.m7}  # Suspended 4 with 7

    # Add9 and extended chords
    add9 = {Intervals.P0, Intervals.M3,
            Intervals.P5, Intervals.M9}    # Major add 9
    madd9 = {Intervals.P0, Intervals.m3,
             Intervals.P5, Intervals.M9}   # Minor add 9
    M9 = {Intervals.P0, Intervals.M3, Intervals.P5,
          Intervals.M7, Intervals.M9}     # Major 9
    m9 = {Intervals.P0, Intervals.m3, Intervals.P5,
          Intervals.m7, Intervals.M9}     # Minor 9
    dom9 = {Intervals.P0, Intervals.M3, Intervals.P5,
            Intervals.m7, Intervals.M9}   # Dominant 9

    # 11th chords
    M11 = {Intervals.P0, Intervals.M3, Intervals.P5,
           Intervals.M7, Intervals.M9, Intervals.P11}    # Major 11
    m11 = {Intervals.P0, Intervals.m3, Intervals.P5,
           Intervals.m7, Intervals.M9, Intervals.P11}    # Minor 11
    dom11 = {Intervals.P0, Intervals.M3, Intervals.P5,
             Intervals.m7, Intervals.M9, Intervals.P11}  # Dominant 11

    # 13th chords
    M13 = {Intervals.P0, Intervals.M3, Intervals.P5,
           Intervals.M7, Intervals.M9, Intervals.M13}    # Major 13
    m13 = {Intervals.P0, Intervals.m3, Intervals.P5,
           Intervals.m7, Intervals.M9, Intervals.m13}    # Minor 13
    dom13 = {Intervals.P0, Intervals.M3, Intervals.P5,
             Intervals.m7, Intervals.M9, Intervals.M13}  # Dominant 13

    # Power chords (common in rock)
    P5 = {Intervals.P0, Intervals.P5}  # Power chord (no third)

    # Flat 5 / sharp 5 variations
    augM7 = {Intervals.P0, Intervals.M3, Intervals.aug5,
             Intervals.M7}  # Augmented Major 7
    dimM7 = {Intervals.P0, Intervals.m3, Intervals.b5,
             Intervals.M7}    # Diminished Major 7

    # Omitted chord variations
    M_no5 = {Intervals.P0, Intervals.M3}            # Major with no 5
    m_no5 = {Intervals.P0, Intervals.m3}            # Minor with no 5

    aug6 = {Intervals.P0, Intervals.M3, Intervals.P5, Intervals.aug6}

    # Jazz and advanced
    aug9 = {Intervals.P0, Intervals.M3,
            Intervals.aug5, Intervals.M9}   # Augmented 9
    M7sharp11 = {Intervals.P0, Intervals.M3, Intervals.P5,
                 Intervals.M7, Intervals.aug11}  # Lydian
    # Half-diminished (alternate spelling)
    m7b5 = {Intervals.P0, Intervals.m3, Intervals.b5, Intervals.m7}

    # Cluster / polychords (less common, added tones)
    M7add13 = {Intervals.P0, Intervals.M3,
               Intervals.P5, Intervals.M7, Intervals.M13}
    m7add13 = {Intervals.P0, Intervals.m3,
               Intervals.P5, Intervals.m7, Intervals.m13}

    # Extended suspensions
    sus2sus4 = {Intervals.P0, Intervals.M2,
                Intervals.P4, Intervals.P5}  # Sus2 Sus4

    # Empty/Unknown
    UNKNOWN = set()

    @classmethod
    def from_intervals(cls, intervals: list[Interval] | set[Interval]) -> ChordShape:
        target_semitones = {i.functional_degree % 12 for i in intervals}
        for shape in cls:
            shape_semitones = {i.functional_degree % 12 for i in shape.intervals}
            if target_semitones == shape_semitones:
                return shape
        return cls.UNKNOWN

    @classmethod
    def from_notes(cls, notes: list[Note]) -> ChordShape:
        if not notes:
            return cls.UNKNOWN

        sorted_midi = sorted([n.midi for n in notes])
        root_midi = sorted_midi[0]
        intervals = [Intervals.from_semitones(
            m - root_midi) for m in sorted_midi]

        return cls.from_intervals(intervals)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def intervals(self) -> set[Interval]:
        return self.value

    @property
    def similar(self):
        similar_shapes: set[ChordShape] = set()
        similar_intervals: set[Intervals] = set()
        for interval in self.intervals:
            similar_intervals = similar_intervals.union({Intervals.P0})  # TODO

        for shape in ChordShape:
            if shape.intervals <= similar_intervals and len(shape.intervals) == len(self.intervals):
                similar_shapes.add(shape)

        return similar_shapes
