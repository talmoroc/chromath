from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from music_objects import Note, Chord

A4_MIDI_VALUE = 69
A4_FREQ = 440.0

def pitch_distance(p1: int | Pitch, p2: int | Pitch) -> int:
    """ Distance between two notes (defined by semitones)
    within the 12-tones equal temperament. Values in [0-6]"""
    if isinstance(p1, Pitch): p1 = p1.altered_pitch
    if isinstance(p2, Pitch): p2 = p2.altered_pitch
    if not 0 <= p1 <= 11 or not 0 <= p2 <= 11: raise ValueError(f'Pitches must be between 0 and 11 - got {p1}, {p2}')
    diff = abs(p2 - p1) % 12
    return diff if diff <= 6 else 12 - diff


# NOTE: l'utilité de cette classe est peut-être limitée. Peut être géré au niveau de Note.
# D'un autre côté, c'est la classe qui représente la similitude par octave des notes.
@dataclass
class Pitch:
    absolute_pitch: int
    accidentals: int

    def __post_init__(self):
        if not 0 <= self.absolute_pitch <= 11:
            self.absolute_pitch = self.absolute_pitch % 12

    @cached_property
    def altered_pitch(self) -> int: return (self.absolute_pitch + self.accidentals) % 12

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

    @classmethod
    def from_midi(cls, midi1: int, midi2: int, prefer_thirds: bool = False) -> Interval:
        semitones = abs(midi1 - midi2)
        return cls.from_semitones(semitones, prefer_thirds=prefer_thirds)
    
    @classmethod
    def from_notes(cls, note1: Note, note2: Note, prefer_thirds: bool = False) -> Interval:
        return cls.from_midi(note1.midi, note2.midi, prefer_thirds=prefer_thirds)
    
    @property
    def semitones(self) -> int:
        return self.value[0]
    
    @property
    def degree(self) -> int:
        return self.value[1]
    
    @property
    def dissonance(self) -> float:
        return Chord([Note(0), Note(self.semitones)]).dissonance

    @property
    def similars(self) -> set[Interval]:
        modulo_semitones = self.semitones % 12
        return {i for i in Interval if i.semitones % 12 == modulo_semitones}


class ChordShape(Enum):
    # Triads (most common)
    M = {Interval.P0, Interval.M3, Interval.P5}      # Major
    m = {Interval.P0, Interval.m3, Interval.P5}      # Minor
    aug = {Interval.P0, Interval.M3, Interval.aug5}  # Augmented
    dim = {Interval.P0, Interval.m3, Interval.b5}    # Diminished
    
    # Seventh chords (very common)
    M7 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7}      # Major 7
    m7 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7}      # Minor 7
    dom7 = {Interval.P0, Interval.M3, Interval.P5, Interval.m7}    # Dominant 7
    mM7 = {Interval.P0, Interval.m3, Interval.P5, Interval.M7}     # Minor Major 7
    aug7 = {Interval.P0, Interval.M3, Interval.aug5, Interval.m7}  # Augmented 7
    dim7 = {Interval.P0, Interval.m3, Interval.b5, Interval.b7}    # Diminished 7
    halfdim7 = {Interval.P0, Interval.m3, Interval.b5, Interval.m7}  # Half-diminished 7
    
    # Sixth chords (common)
    M6 = {Interval.P0, Interval.M3, Interval.P5, Interval.M6}      # Major 6
    m6 = {Interval.P0, Interval.m3, Interval.P5, Interval.M6}      # Minor 6
    
    # Suspended chords (common)
    sus2 = {Interval.P0, Interval.M2, Interval.P5}     # Suspended 2
    sus4 = {Interval.P0, Interval.P4, Interval.P5}     # Suspended 4
    sus47 = {Interval.P0, Interval.P4, Interval.P5, Interval.m7}  # Suspended 4 with 7
    
    # Add9 and extended chords
    add9 = {Interval.P0, Interval.M3, Interval.P5, Interval.M9}    # Major add 9
    madd9 = {Interval.P0, Interval.m3, Interval.P5, Interval.M9}   # Minor add 9
    M9 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.M9}     # Major 9
    m9 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7, Interval.M9}     # Minor 9
    dom9 = {Interval.P0, Interval.M3, Interval.P5, Interval.m7, Interval.M9}   # Dominant 9
    
    # 11th chords
    M11 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.M9, Interval.P11}    # Major 11
    m11 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7, Interval.M9, Interval.P11}    # Minor 11
    dom11 = {Interval.P0, Interval.M3, Interval.P5, Interval.m7, Interval.M9, Interval.P11}  # Dominant 11
    
    # 13th chords
    M13 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.M9, Interval.M13}    # Major 13
    m13 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7, Interval.M9, Interval.m13}    # Minor 13
    dom13 = {Interval.P0, Interval.M3, Interval.P5, Interval.m7, Interval.M9, Interval.M13}  # Dominant 13
    
    # Power chords (common in rock)
    P5 = {Interval.P0, Interval.P5}  # Power chord (no third)
    
    # Flat 5 / sharp 5 variations
    augM7 = {Interval.P0, Interval.M3, Interval.aug5, Interval.M7}  # Augmented Major 7
    dimM7 = {Interval.P0, Interval.m3, Interval.b5, Interval.M7}    # Diminished Major 7
    
    # Omitted chord variations
    M_no5 = {Interval.P0, Interval.M3}            # Major with no 5
    m_no5 = {Interval.P0, Interval.m3}            # Minor with no 5
    
    # Jazz and advanced
    aug9 = {Interval.P0, Interval.M3, Interval.aug5, Interval.M9}   # Augmented 9
    M7sharp11 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.aug11}  # Lydian
    m7b5 = {Interval.P0, Interval.m3, Interval.b5, Interval.m7}     # Half-diminished (alternate spelling)
    
    # Cluster / polychords (less common, added tones)
    M7add13 = {Interval.P0, Interval.M3, Interval.P5, Interval.M7, Interval.M13}
    m7add13 = {Interval.P0, Interval.m3, Interval.P5, Interval.m7, Interval.m13}
    
    # Extended suspensions
    sus2sus4 = {Interval.P0, Interval.M2, Interval.P4, Interval.P5}  # Sus2 Sus4
    
    # Empty/Unknown
    UNKNOWN = {Interval.P0}
    
    @classmethod
    def from_intervals(cls, intervals: list[Interval] | set[Interval]) -> ChordShape:
        target_semitones = {i.semitones % 12 for i in intervals}
        for shape in cls:
            shape_semitones = {i.semitones % 12 for i in shape.intervals}
            if target_semitones == shape_semitones:
                return shape
        return cls.UNKNOWN

    @classmethod
    def from_notes(cls, notes: list[Note]) -> ChordShape:
        if not notes:
            return cls.UNKNOWN
        
        sorted_midi = sorted([n.midi for n in notes])
        root_midi = sorted_midi[0]
        intervals = [Interval.from_semitones(m - root_midi) for m in sorted_midi]
        
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
        similar_intervals: set[Interval] = set()
        for interval in self.intervals:
            similar_intervals.union(interval.similars)
        
        for shape in ChordShape:
            if shape.intervals <= similar_intervals:
                similar_shapes.add(shape)
        
        return similar_shapes
