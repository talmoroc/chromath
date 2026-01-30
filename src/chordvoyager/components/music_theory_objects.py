from __future__ import annotations

from functools import cached_property
from dataclasses import dataclass


import src.chordvoyager.core.freq_ops as freq_op
import src.chordvoyager.core.degree_ops as deg_op
import src.chordvoyager.core.chroma_ops as chr_op
import src.chordvoyager.core.conversion as core_conv
import src.chordvoyager.core.validation as val
from typing import overload, cast
from numpy.typing import ArrayLike

import numpy as np

from src.chordvoyager.constants import (
    DefaultMusicSystem as MS,
    IONIAN_SEMITONES,
)
from src.chordvoyager.types import ChromaVec, DegreeVec, CycleMatrix, DTYPE, NDArrayInt8

import src.chordvoyager.core.cycle_ops as cycle_op


class Tone:
    def shift(t: int, n: int) -> int:
        return deg_op.shift(cast("DegreeVec", t), n)[0]

    def to_chroma(t: int) -> Chroma:
        return Chroma(core_conv.degree_to_chroma(np.array(t)))

    def isin(t: int, chroma: Chroma) -> bool:
        return chroma.vector[t] == 1


class Chroma:
    def __init__(self, data: ArrayLike):
        arr: ChromaVec = val.validate_chroma(
            np.array(data, copy=True).ravel()
        )  # Conversion
        arr.flags.writeable = False  # Immutability
        self.vector: ChromaVec = arr
        self.degrees: DegreeVec = chr_op.chroma_to_degree(self.vector)

    def shift(self, n: int) -> Chroma:
        return Chroma(chr_op.shift(self.vector, n))

    def invert(self, pivot: int = 0) -> Chroma:
        return Chroma(chr_op.invert(self.vector, pivot))


class Cycle:
    def __init__(self, step: int):
        self.step: int = step
        self.cycle, _, self.mask = cycle_op.generate(self.step)

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice) -> NDArrayInt8: ...

    def __getitem__(self, index: int | slice) -> int | NDArrayInt8:  # order -> semitone
        if isinstance(index, int):
            return (self.step * index) % MS.tones
        start = 0 if index.start is None else index.start
        dividend = start // MS.tones
        start -= MS.tones * dividend
        stop = (
            start + MS.tones if index.stop is None else index.stop - MS.tones * dividend
        )
        step = 1 if index.step is None else index.step % MS.tones
        if not self.is_complete:
            return np.array(
                [self[i] for i in range(start, stop, step)], dtype=DTYPE.Cycle
            )
        indices = np.arange(start, stop, step) * self.step % MS.tones
        return self.rank_matrix.view()[0, indices]

    @property
    def periodicity(self) -> int:
        return cycle_op.periodicity(self.step)

    @property
    def is_complete(self) -> bool:
        return cycle_op.is_complete(self.step)

    @property
    def chroma(self) -> Chroma:
        return Chroma(val.validate_chroma(self.mask))

    @property
    def cycle_matrix(self) -> CycleMatrix:
        return self.cycle

    @cached_property
    def rank_matrix(self) -> NDArrayInt8:
        idx = np.arange(MS.tones)
        cycle_semitones = (
            np.arange(0, self.step * self.periodicity, self.step) % MS.tones
        )
        mask = np.zeros(MS.tones, dtype=DTYPE.Cycle)
        cycle_asc = np.zeros(MS.tones, dtype=DTYPE.Cycle)
        mask[cycle_semitones] = 1
        cycle_asc[cycle_semitones] = np.arange(self.periodicity)
        cycle_desc = (self.periodicity - cycle_asc) % self.periodicity
        cycle = np.where(cycle_asc <= cycle_desc, cycle_asc, -cycle_desc)
        return np.ma.array(
            [idx, cycle, cycle_asc, -cycle_desc],
            mask=np.tile((1 - mask), (4, 1)),
            dtype=DTYPE,
        )

    @cached_property  # Access : pitch_to_rank[1|2, pitch] 1 = ascending, 2 = descending
    def pitch_to_rank(self) -> NDArrayInt8:
        return self.rank_matrix.view()[2:4,]

    @cached_property  # Access : rank_to_pitch[1|2, rank] 1 = ascending, 2 = descending
    def rank_to_pitch(self) -> NDArrayInt8:
        if not self.is_complete:  # We repeat the matrix to get a MS.tones dimension
            ascending_rank = [self[i] for i in range(MS.tones)]
            descending_rank = [self[-i] for i in range(MS.tones)]
        else:
            ascending_rank = np.argsort(self.rank_matrix[2, :])
            descending_rank = np.argsort(-self.rank_matrix[3, :])
        return np.array([ascending_rank, descending_rank], dtype=DTYPE)


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
        if len(self.degree_semitones) != MS.degrees:
            raise ValueError(
                f"Need {MS.degrees} notes, got {len(self.degree_semitones)}"
            )
        for st in self.degree_semitones:
            if not 0 <= st < MS.tones:
                raise ValueError(
                    f"Semitones must be between 0 and {MS.tones}, got {self.degree_semitones}"
                )

    def __str__(self):
        return f"Scale({self.degree_semitones})"

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.__str__()}>"

    def __getitem__(self, deg: int):
        return self.degree_semitones[(deg - 1) % MS.degrees]

    def __len__(self):
        return len(self.degree_semitones)

    @classmethod
    def from_mask(cls, note_mask: list[int]):
        return Scale(np.flatnonzero(note_mask).tolist())

    @classmethod
    def from_semitones_diff(cls, st_diff: list[int]):
        return Scale(np.cumsum(st_diff).tolist())

    @cached_property
    def semitones_diff(self) -> list[int]:
        return np.diff(
            self.degree_semitones, prepend=self.degree_semitones[-1] - MS.tones
        ).tolist()

    @cached_property
    def mask(self) -> list[int]:
        mask = np.zeros(MS.tones, dtype=int)
        mask[self.degree_semitones] = 1
        return mask.tolist()

    @cached_property
    def chroma(self) -> Chroma:
        return Chroma(self.mask)

    def shift(self, start_degree: int) -> Scale:
        new_semitones = np.roll(self.degree_semitones, start_degree % MS.degrees)
        new_semitones = (new_semitones - new_semitones[0]) % MS.tones
        return Scale(new_semitones.tolist())

    def transpose(self, start_pitch: int):
        return Scale([(st + start_pitch) % MS.tones for st in self.degree_semitones])

    ########### POSSIBLE EXTENSIONS ###############
    # NOTE: intervals calculation may be externalized. You need to compute intervals from scales or chords.
    # NOTE: However, the type of chord is only interpretable from within a scale, not with semitones.
    # NOTE: The scale, is really a list of semitones and the reference for everything else.
    # NOTE: Or is it the tonality ? If it was, it would allow to compare easily tonalities with different scales.
    # intervals from root

    @cached_property
    def intervals(self) -> list[Interval]:
        return [Interval(degree + 1, self[degree + 1]) for degree in range(MS.degrees)]

    # intervals from all degrees
    @cached_property
    def all_intervals(self) -> set[Interval]:
        return set(
            [
                interval
                for i in range(MS.degrees)
                for interval in self.shift(i).intervals
            ]
        )


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


@dataclass
class Pitch:
    absolute_pitch: int
    accidentals: int = 0
    MS.tones: int = MS.tones

    def __post_init__(self):
        if not 0 <= self.absolute_pitch < MS.tones:
            self.absolute_pitch = self.absolute_pitch % MS.tones

    @classmethod
    def from_midi(cls, midi_value) -> Pitch:
        return Pitch(midi_value % MS.tones, 0)

    def __str__(self):
        return f"Pitch({self.absolute_pitch}, {self.accidentals})"

    def __repr__(self):
        return self.__str__()

    @cached_property
    def sounding_pitch(self):
        return (self.absolute_pitch + self.accidentals) % 12

    @cached_property
    def is_altered(self):
        return self.accidentals != 0

    def __sub__(self, other: Pitch | int):
        if isinstance(other, Pitch):
            return Pitch(
                self.absolute_pitch - other.absolute_pitch,
                self.accidentals - other.accidentals,
            )
        return Pitch((self.absolute_pitch - other) % MS.tones, self.accidentals)

    def __add__(self, other: Pitch | int):
        if isinstance(other, Pitch):
            return Pitch(
                (self.absolute_pitch + other.absolute_pitch) % MS.tones,
                self.accidentals + other.accidentals,
            )
        return Pitch((self.absolute_pitch + other) % MS.tones, self.accidentals)

    def shift(self, semitones: int):
        return Pitch(self.absolute_pitch + semitones, self.accidentals)

    def alter(self, accidentals: int):
        return Pitch(self.absolute_pitch, self.accidentals + accidentals)


# On peut déduire les pitches probables d'un accord à partir de semitones + scale, mais ce n'est pas le sujet.
# En définissant la notion d'accord, il faut commencer à distinguer les méthodes d'analyses
# 1) Scale = générateur d'accords et d'extensions
# 2) Cycle = générateur d'accords et d'extensions.

# NOTE: un cycle, un accord, une scale sont tous représentables par une liste de [0,1] de longueur 12.
# Ils ont simplement des méthodes différentes.


# class Chord:
#     def __init__(self, intervals: list[Interval]):
#         self.chroma = Chroma([0] * 12)


# Le risque de distinguer au sein même de l'objet Chord de quels intervalles il est constitué,
# C'est qu'il faut chercher après les enharmonies.
# La représentation en semitones est incluse dans la représentation en intervalles, et plusieurs ReducedChord peuvent la partager.


# NOTE: les notions d'intervalles dans une tonalité, de degrés... Sont directement dérivées de la Scale....
# La seule différence entre la tonalité et la scale, c'est qu'elle a une root.
# Il s'agit d'une scale appliquée à une root. Ce qui va déterminer les accords concrets en termes de notes.
# Ces accords peuvent cependant être déterminés de façon relative à partir de la scale...
# Le fait de séparer les deux peut rendre compliqué des calculs du type : F = IV de C mais aussi I de F, VII de Gm, V de Bb...

# Peut-être que ces calculs doivent être faits dans une tonalité ou bien dans un autre objet qui rassemble toutes les tonalités.


class Tonality:
    def __init__(self, root: int, scale: Scale):
        if not 0 <= root <= 11:
            raise ValueError(
                f"Tonality root must be a pitch between 0 and 11, got {root}"
            )
        self.root = root  # Flatten accidentals for the root of a tonality
        self.scale = scale.transpose(self.root)  # Apply the scale to the root
        self.chroma = self.scale.chroma


# METHODS OR EXTERNAL FUNCTIONS OR EXTERNAL OBJECTS : TO BE DETERMINED
# EXTERNAL: Provide the midi notes of the tonality

# Check if a note is within the scale

# If not, give the closest notes/the possible alterations

# Provide the intervals of the tonality
# Nature
# Dissonance

# Find the closest interval, its different interpretations:
# If in tonality enharmonically : which altered interval, change of degree
# If not in tonality enharmonically : which possibilities of alteration of an interval

# Provide degrees of the tonality

# Provide the structure of the tonality in terms of degrees
# shapes, dissonance
# cycles
# Consider extensions or not
# At this point, on doit avoir des pistes pour la génération d'accords au sein d'une tonalité

# Check if a given chord is a degree
# And give the degree, and the chord
# Consider extensions or not

# If not, find the closest chord(s)
# And their distance
# And the number of notes of the chord that are within the tonality

# Il manque : notions de distance note-note, chord-chord, chord-tonalité
# Ces notions sont nécessaires pour étudier les rapports entre les degrés.

# Distance entre les gammes ==> similitude ==> Notion de relatif mineur.
# Il manque aussi les notions de distance dans le cycle des quintes. Cette notion doit dépendre (ou non, au choix) de la nature Majeur/Mineur
# Exemple : Dm = +2 car C -> G -> D mais Dm = -1 car relatif mineur de F
# C Dm Em F G Am Bd
# 0  2 4 -1 1 3  5
# 0 -1 1 -1 1 0  ?1

# Am Bd  C  Dm Em  F  G
# 0  2  -3  -1  1 -4 -2
# 0 ?-1  0  -1  1 -1  1
# Mais également (surtout ?) les common tones
# Am - C, Dm - F, Em - G, et Bd - G & Dm
# Avec les extensions : Am7 = Cm6, Dm7 = F6, Em7 = G6, Bd7 = Dm6
# G7 ~= Bd + 6, FM7 = Amb6, CM7 = Emb6,

# POSSIBLE EXTENSIONS


def tonality_mask(t: Tonality):
    return (t.scale.transpose(t.root).mask * 11)[:128]


def tonality_midi_scale(t: Tonality):
    return np.flatnonzero(tonality_mask(t)).tolist()


freq_op.dissonance([0, 6])

scale = Scales.IONIAN


class DegreeChordsFromScale:
    # Similar in terms of common notes, shape, distance to the cycles
    # common notes = equivalence between chords. (Also check scale transposition similarity ?)
    # shape = change of feeling when switching
    # cycle = feeling of closeness, root movement, logical movement
    # From this we can deduce the function of degrees in the scale.
    pass


def pitch_distance(p1: int | Pitch, p2: int | Pitch) -> int:
    """Distance between two notes (defined by semitones)
    within the 12-tones equal temperament. Values in [0-6]"""
    if isinstance(p1, Pitch):
        p1 = p1.sounding_pitch
    if isinstance(p2, Pitch):
        p2 = p2.sounding_pitch
    if not 0 <= p1 <= 11 or not 0 <= p2 <= 11:
        raise ValueError(f"Pitches must be between 0 and 11 - got {p1}, {p2}")
    diff = abs(p2 - p1) % 12
    return diff if diff <= 6 else 12 - diff


class Note:
    def __init__(self, midi_value: int):
        if midi_value < 0 or midi_value > 127:
            raise ValueError(
                f"MIDI note must be between 0 and 127 - value {midi_value}"
            )
        self.midi = midi_value
        self.note_index = self.midi % 12
        self.octave = self.midi // 12
        self.pitch = Pitch(self.note_index, 0)
        self.name = str(self.pitch)
        self.freq = freq_op.midi_freq(self.midi)

    def shift(self, semitones: int = 0, octaves: int = 0) -> Note:
        return Note(self.midi + semitones + 12 * octaves)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str((self.name, self.midi))

    def __add__(self, other) -> Note:
        if isinstance(other, Note):
            other = other.midi
        return Note(self.midi + other)

    def __sub__(self, other) -> Note:
        if isinstance(other, Note):
            other = other.midi
        return Note(self.midi - other)

    def __gt__(self, other) -> bool:
        if isinstance(other, Note):
            other = other.midi
        return self.midi > other

    def __lt__(self, other) -> bool:
        if isinstance(other, Note):
            other = other.midi
        return self.midi < other

    def __eq__(self, other) -> bool:
        if isinstance(other, Note):
            other = other.midi
        return self.midi == other

    @classmethod
    def from_pitch(cls, pitch: Pitch, octave: int = 0):
        return cls(pitch.sounding_pitch + (octave + 1) * 12)

    @cached_property
    def chroma(self) -> Chroma:
        return Tone.to_chroma(self.note_index)


class Chord:
    def __init__(self, notes: list[Note]):
        if len(notes) < 2:
            raise ValueError(
                f"{len(notes)} notes given. At least 2 notes must be provided to build a Chord."
            )
        self.notes = sorted(notes)
        self.lowest_note = min(self.notes)
        self.octave = self.lowest_note.octave  # Temporary octave assignment
        self.midi = [note.midi for note in self.notes]
        self.semitones = [(note - self.lowest_note).midi % 12 for note in self.notes]
        self.frequencies = [note.freq for note in self.notes]
        self.intervals = [
            Interval(0, 0),
            # Interval.from_semitones((i - self.lowest_note).note_index) for i in self.notes
        ]
        self.relative_periodicity = freq_op.relative_periodicity(self.semitones)
        self.dissonance = freq_op.smoothed_relative_periodicity(self.semitones)
        self._dissonance_raw = freq_op.smoothed_relative_periodicity(
            self.semitones, log=False
        )

    def __str__(self):
        return f"Root: MIDI Note {self.lowest_note.midi} with frequency {self.lowest_note.freq} Hz\n {(self.lowest_note.pitch)} {None} Chord in inversion {None}. Ambiguous root: {None}"

    # @classmethod
    # def from_shape(cls, root: Note, chord_shape: ChordShape):
    #     notes = [root + i.pitch for i in chord_shape.intervals]
    #     return cls(notes)

    # @property
    # def _qualities(self) -> list[tuple[Note, ChordShape, int, float]]:
    #     quality_tuple = []
    #     for index in range(len(self.notes)):
    #         inversion = self.invert(index)
    #         quality = ChordShape.from_intervals(inversion.intervals)
    #         if quality != ChordShape.UNKNOWN:
    #             quality_tuple.append(
    #                 (
    #                     self.notes[index],
    #                     quality,
    #                     (len(self.notes) - index) % len(self.notes),
    #                     inversion.dissonance,
    #                 )
    #             )
    #     return quality_tuple

    # @property
    # def _potential_roots(self) -> list[Note]:
    #     return [q[0] for q in self._qualities]

    # @property
    # def root_is_ambiguous(self) -> bool:
    #     if self.lowest_note in self._potential_roots:
    #         return False
    #     return len(self._qualities) > 1

    # @property
    # def most_probable_quality(self) -> tuple[Note, ChordShape, int, float]:
    #     if self.lowest_note in self._potential_roots:
    #         return self._qualities[self._potential_roots.index(self.lowest_note)]
    #     return sorted(self._qualities, key=lambda x: x[3])[-1]

    # @property
    # def root(self) -> Note:
    #     return self.most_probable_quality[0]

    # @property
    # def shape(self) -> ChordShape:
    #     return self.most_probable_quality[1]

    # @property
    # def inversion(self) -> int:
    #     return self.most_probable_quality[2]

    # @property
    # def root_position_dissonance(self) -> float:
    #     return self.most_probable_quality[3]

    # @property
    # def similar(self) -> set[Chord]:
    #     similar_chords = set()
    #     for index in range(len(self.notes)):
    #         inversion = self.invert(index)
    #         for shape in ChordShape.from_intervals(inversion.intervals).similar:
    #             if shape != ChordShape.UNKNOWN:
    #                 similar_chords.add(Chord.from_shape(
    #                     self.notes[index], shape))
    #     return similar_chords

    def invert(self, index: int = 1) -> Chord:
        new_notes = self.notes[index:]
        for i in range(index):
            new_notes += [self.notes[i] + 12]
        return Chord(new_notes)

    def pitch_eq(self, other: Chord) -> bool:
        """Compare chords by pitch class (ignoring inversion, octave, and repetition)."""
        if not isinstance(other, Chord):
            return False
        self_pitches = sorted({n.pitch for n in self.notes}, key=lambda p: p.value)
        other_pitches = sorted({n.pitch for n in other.notes}, key=lambda p: p.value)
        return self_pitches == other_pitches


@dataclass(frozen=True, order=True)
class Interval:
    functional_degree: int
    pitch: int

    def __post_init__(self):
        if not 1 <= self.functional_degree <= MS.func_degrees:
            raise ValueError(f"Degree {self.functional_degree} out of bounds")
        if not 0 <= self.pitch < MS.tones:
            raise ValueError(f"Pitch {self.pitch} out of bounds")

    def __str__(self):
        return f"Interval(degree={self.functional_degree}, pitch={self.pitch})"

    def __repr__(self):
        return self.__str__()

    @cached_property
    def degree(self):
        return self.functional_degree % MS.degrees


__all__ = ["Pitch", "Interval", "Chord", "Cycle", "Scale", "Scales", "Tonality"]
