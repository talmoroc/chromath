from __future__ import annotations

import os
if os.getcwd() == 'C:\\Users\\tomco\\dev\\repos\\chord-generator-prototype': os.chdir('./engine_utils')

from collections.abc import Sequence
from typing import Iterable
import dissonance_computation as utils
from music_system import MusicSystem, get_current_music_system, use_system
import math
from dataclasses import dataclass, replace, field
from functools import cached_property
import warnings

from typing import ClassVar, Optional, overload, Protocol, TypeVar, Any, Literal, Self, override
from numpy.typing import NDArray, DTypeLike, ArrayLike
import numpy as np
DTYPE = np.int8
NDArrayInt8 = NDArray[np.int8]


MS: MusicSystem = get_current_music_system()
IONIAN_SEMITONES: list[int] = [0, 2, 4, 5, 7, 9, 11]


class Tone:
    def rotate(t: int, n: int) -> int:
        return (t + n) % 12

    def to_chroma(t: int) -> Chroma:
        return Chroma.from_semitones([t])

    def is_in(t: int, music_object: Cycle | Scale | Chord | Tonality) -> bool:
        return music_object.chroma[t % 12] == 1


class Chroma:
    __slots__ = ('_mask', '_semitones', '_as_int')

    def _new(self, data) -> Self:  # Allows to inherit methods without redefining output. Useful for Scales
        return self.__class__(data)

    def __init__(self, data: ArrayLike):
        arr: NDArrayInt8 = np.array(data, DTYPE, copy=True).ravel()  # Conversion
        arr.flags.writeable = False  # Immutability
        if arr.shape != (MS.tones,):  # Check shape
            raise ValueError(f"Chroma must be length {MS.tones}, got {arr.shape}")
        if not set(arr) <= {0, 1}:
            raise ValueError(f"Chroma values must be 0 or 1, got {arr}")
        self._mask: NDArrayInt8 = arr
        self._semitones: NDArrayInt8 = np.flatnonzero(arr).astype(DTYPE)
        self._as_int: int = int(np.dot(arr, MS.powers))

    @classmethod
    def from_semitones(cls, semitones: ArrayLike, rolling_indices: bool = True) -> Chroma:
        semitones = np.array(semitones, dtype=DTYPE, copy=True).ravel()
        if not len(semitones) <= MS.tones:  # check if ArrayLike is too long
            raise ValueError(f'Semitones length must be between 0 and {MS.tones}, got {len(semitones)}')
        if (semitones > 12).any() or (semitones < 0).any():  # check if pitches are out of bounds
            match rolling_indices:
                case True: semitones %= MS.tones  # correction if rolling_indices
                case False: raise ValueError(f'Semitones must be between 0 and {MS.tones}, got {semitones}')
        _temp_mask = np.zeros(MS.tones, dtype=DTYPE)
        _temp_mask[semitones] = 1
        return cls(_temp_mask)

    @classmethod
    def from_bits(cls, bitwise_repr: int) -> Chroma:
        if bitwise_repr >= MS.max_int_repr:
            raise ValueError(f'Bitwise representation must be less than 2**{MS.tones} = {MS.max_int_repr}, got {bitwise_repr}')
        _temp_mask = np.array(((bitwise_repr >> i) & 1 for i in range(MS.tones)), dtype=DTYPE)
        return cls(_temp_mask)

    def __len__(self) -> int: return MS.tones
    def __int__(self) -> int: return self._as_int
    def __str__(self) -> str: return f"{self._mask}"
    def __repr__(self) -> str: return f"<Chroma({self._mask}), {self._as_int}>"
    def __hash__(self) -> int: return hash(self._as_int)
    def __iter__(self) -> Iterable: return iter(self._mask)
    def __in__(self, key: int) -> bool: return self.binary_vector[key] == 1
    def __add__(self, other: Self) -> Self: return self._new(self._mask | other._mask)
    def __mul__(self, other: Self) -> Self: return self._new(self._mask & other._mask)
    def __sub__(self, other: Self) -> Self: return self._new(self._mask & ~other._mask)
    def __invert__(self) -> Self: return self._new(1 - self._mask)
    def __matmul__(self, n: int) -> Self: return self.rotate(n)
    def __rmatmul__(self, n: int) -> Self: return self.rotate(n)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Chroma):
            return self._as_int == other._as_int
        try:
            return self._as_int == self._new(other)._as_int
        except:
            return NotImplemented

    @overload
    def __getitem__(self, index: int) -> int: ...
    @overload
    def __getitem__(self, index: slice[Any, Any, Any]) -> NDArrayInt8: ...

    def __getitem__(self, index: int | slice) -> int | NDArrayInt8:
        if isinstance(index, int):
            return self._mask[index % MS.tones]
        start = 0 if index.start is None else index.start
        stop = start + MS.tones if index.stop is None else index.stop
        step = 1 if index.step is None else index.step
        indices = np.arange(start, stop, step) % MS.tones
        return self._mask[indices]

    @cached_property
    def binary_vector(self) -> NDArrayInt8: return self._mask

    @cached_property
    def semitones_vector(self) -> NDArrayInt8:
        return np.array([np.arange(np.sum(self._mask)), self._semitones], dtype=DTYPE)

    def rotate(self, n: int) -> Self:
        return self._new(self[n: n + MS.tones])

    def invert(self, pivot: int = 0) -> Self:
        """Musical inversion around a pivot (default 0)"""
        new_mask = np.zeros(MS.tones, dtype=DTYPE)
        indices = (pivot - self._semitones) % MS.tones
        new_mask[indices] = 1
        return self._new(new_mask)


@dataclass(frozen=True)
class Cycle:
    step: int

    def __post_init__(self):
        if not 1 <= self.step < MS.tones:
            raise ValueError(f'Step must be between 1 and {MS.tones}: got {self.step}')

    def __in__(self, key: int) -> bool: return self.chroma[key] == 1

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
        stop = start + MS.tones if index.stop is None else index.stop - MS.tones * dividend
        step = 1 if index.step is None else index.step % MS.tones
        slice_size = (stop - start) // MS.tones + 1
        if not self.is_complete:
            return np.array([self[i] for i in range(start, stop, step)], dtype=DTYPE)
        indices = np.arange(start, stop, step) * self.step % MS.tones
        return self.rank_matrix.view()[0, indices]

    @cached_property
    def periodicity(self): return MS.tones // math.gcd(self.step, MS.tones)

    @cached_property
    def is_complete(self): return self.periodicity == MS.tones

    @cached_property
    def chroma(self) -> Chroma:
        if self.is_complete:
            return Chroma([1] * MS.tones)
        else:
            return Chroma(np.ndarray.astype( 1 - self.rank_matrix.mask[0], DTYPE))  # ty:ignore[unresolved-attribute]
    @cached_property
    def rank_matrix(self) -> NDArrayInt8:
        idx = np.arange(MS.tones)
        cycle_semitones = np.arange(0, self.step * self.periodicity, self.step) % MS.tones
        mask = np.zeros(MS.tones, dtype=DTYPE)
        cycle_asc = np.zeros(MS.tones, dtype=DTYPE)
        mask[cycle_semitones] = 1
        cycle_asc[cycle_semitones] = np.arange(self.periodicity)
        cycle_desc = (self.periodicity - cycle_asc) % self.periodicity
        cycle = np.where(cycle_asc <= cycle_desc, cycle_asc, -cycle_desc)
        return np.ma.array([idx, cycle, cycle_asc, -cycle_desc], mask=np.tile((1 - mask), (4, 1)), dtype=DTYPE)

    @cached_property # Access : pitch_to_rank[1|2, pitch] 1 = ascending, 2 = descending
    def pitch_to_rank(self) -> NDArrayInt8:
        return self.rank_matrix.view()[2:4,]

    @cached_property # Access : rank_to_pitch[1|2, rank] 1 = ascending, 2 = descending
    def rank_to_pitch(self) -> NDArrayInt8: 
        if not self.is_complete: # We repeat the matrix to get a MS.tones dimension
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
            raise ValueError(f'Need {MS.degrees} notes, got {len(self.degree_semitones)}')
        for st in self.degree_semitones:
            if not 0 <= st < MS.tones:
                raise ValueError(f'Semitones must be between 0 and {MS.tones}, got {self.degree_semitones}')

    def __str__(self): return f"Scale({self.degree_semitones})"
    def __repr__(self): return f"<{self.__class__.__name__}: {self.__str__()}>"
    def __getitem__(self, deg: int): return self.degree_semitones[(deg - 1) % MS.degrees]
    def __len__(self): return len(self.degree_semitones)

    @classmethod
    def from_mask(cls, note_mask: list[int]): return Scale(np.flatnonzero(note_mask).tolist())

    @classmethod
    def from_semitones_diff(cls, st_diff: list[int]): return Scale(np.cumsum(st_diff).tolist())

    @cached_property
    def semitones_diff(self) -> list[int]:
        return np.diff(self.degree_semitones, prepend=self.degree_semitones[-1] - MS.tones).tolist()

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
        return set([interval for i in range(MS.degrees) for interval in self.shift(i).intervals])


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
    _tones: int = MS.tones

    def __post_init__(self):
        if not 0 <= self.absolute_pitch < self._tones:
            self.absolute_pitch = self.absolute_pitch % self._tones

    def __str__(self):
        return f'Pitch({self.absolute_pitch}, {self.accidentals})'

    def __repr__(self): return self.__str__()

    @cached_property
    def sounding_pitch(self): return (self.absolute_pitch + self.accidentals) % 12

    @cached_property
    def is_altered(self): return self.accidentals != 0

    def __sub__(self, other: Pitch | int):
        if isinstance(other, Pitch):
            return Pitch(self.absolute_pitch - other.absolute_pitch, self.accidentals - other.accidentals)
        return Pitch((self.absolute_pitch - other) % self._tones, self.accidentals)

    def __add__(self, other: Pitch | int):
        if isinstance(other, Pitch):
            return Pitch((self.absolute_pitch + other.absolute_pitch) % self._tones, self.accidentals + other.accidentals)
        return Pitch((self.absolute_pitch + other) % self._tones, self.accidentals)

    def shift(self, semitones: int): return Pitch(self.absolute_pitch + semitones, self.accidentals)
    def alter(self, accidentals: int): return Pitch(self.absolute_pitch, self.accidentals + accidentals)


@dataclass(frozen=True, order=True)
class Interval:
    functional_degree: int
    pitch: int
    _standard_intervals: ClassVar[dict[tuple[int, int], str]] = {}
    _registered: ClassVar[bool] = False

    def __post_init__(self):
        if not 1 <= self.functional_degree <= MS.func_degrees:
            raise ValueError(f'Degree {self.functional_degree} out of bounds')
        if not 0 <= self.pitch < MS.tones:
            raise ValueError(f'Pitch {self.pitch} out of bounds')
        if self._registered and self.name is None:
            warnings.warn(f'\033[33m' + 'Non-standard interval: ({self.functional_degree, self.pitch})' + '\033[0m')

    @cached_property
    def name(self) -> Optional[str]: return self._standard_intervals.get((self.functional_degree, self.pitch))

    def __str__(self):
        if self.name:
            return f"Interval.{self.name}"
        return f"\033[33mInterval(degree={self.functional_degree}, pitch={self.pitch})\033[0m"

    def __repr__(self): return self.__str__()

    @classmethod
    def _register_standard_intervals(cls):
        """ Defines the standard intervals and injects them as class attributes. """
        for name, value in list(cls.__dict__.items()):
            match value:
                case (int(degree), int(pitch)):
                    instance = cls(degree, pitch)
                    setattr(cls, name, instance)
                    cls._standard_intervals[(degree, pitch)] = name
                case _: pass
        cls._registered = True

    # Standard intervals
    P0:    ClassVar['Interval'] = (1, 0)  # type: ignore
    m2:    ClassVar['Interval'] = (2, 1)  # type: ignore
    M2:    ClassVar['Interval'] = (2, 2)  # type: ignore
    aug2:  ClassVar['Interval'] = (2, 3)  # type: ignore
    m3:    ClassVar['Interval'] = (3, 3)  # type: ignore
    M3:    ClassVar['Interval'] = (3, 4)  # type: ignore
    P4:    ClassVar['Interval'] = (4, 5)  # type: ignore
    aug4:  ClassVar['Interval'] = (4, 6)  # type: ignore
    b5:    ClassVar['Interval'] = (5, 6)  # type: ignore
    P5:    ClassVar['Interval'] = (5, 7)  # type: ignore
    aug5:  ClassVar['Interval'] = (5, 8)  # type: ignore
    m6:    ClassVar['Interval'] = (6, 8)  # type: ignore
    M6:    ClassVar['Interval'] = (6, 9)  # type: ignore
    aug6:    ClassVar['Interval'] = (6, 10)  # type: ignore
    b7:  ClassVar['Interval'] = (7, 9)  # type: ignore
    m7:    ClassVar['Interval'] = (7, 10)  # type: ignore
    M7:    ClassVar['Interval'] = (7, 11)  # type: ignore
    P8:    ClassVar['Interval'] = (8, 0)  # type: ignore
    m9:    ClassVar['Interval'] = (9, 1)  # type: ignore
    M9:    ClassVar['Interval'] = (9, 2)  # type: ignore
    aug9:  ClassVar['Interval'] = (9, 3)  # type: ignore
    m10:   ClassVar['Interval'] = (10, 3)  # type: ignore
    M10:   ClassVar['Interval'] = (10, 4)  # type: ignore
    P11:   ClassVar['Interval'] = (11, 5)  # type: ignore
    aug11: ClassVar['Interval'] = (11, 6)  # type: ignore
    m13:   ClassVar['Interval'] = (13, 8)  # type: ignore
    M13:   ClassVar['Interval'] = (13, 9)  # type: ignore

    @cached_property
    def degree(self): return self.functional_degree % MS.degrees

    @cached_property
    def dissonance(self): return utils.dissonance([0, self.pitch])


Interval._register_standard_intervals()  # pyright: ignore[reportPrivateUsage]

# On peut déduire les pitches probables d'un accord à partir de semitones + scale, mais ce n'est pas le sujet.
# En définissant la notion d'accord, il faut commencer à distinguer les méthodes d'analyses
# 1) Scale = générateur d'accords et d'extensions
# 2) Cycle = générateur d'accords et d'extensions.

# NOTE: un cycle, un accord, une scale sont tous représentables par une liste de [0,1] de longueur 12.
# Ils ont simplement des méthodes différentes.


class Chord:
    def __init__(self, intervals: list[Interval]):
        self.chroma = Chroma([0] * 12)
    
    


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
            raise ValueError(f'Tonality root must be a pitch between 0 and 11, got {root}')
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


utils.dissonance([0, 6])

scale = Scales.IONIAN


class DegreeChordsFromScale:
    # Similar in terms of common notes, shape, distance to the cycles
    # common notes = equivalence between chords. (Also check scale transposition similarity ?)
    # shape = change of feeling when switching
    # cycle = feeling of closeness, root movement, logical movement
    # From this we can deduce the function of degrees in the scale.
    pass


def pitch_distance(p1: int | Pitch, p2: int | Pitch) -> int:
    """ Distance between two notes (defined by semitones)
    within the 12-tones equal temperament. Values in [0-6]"""
    if isinstance(p1, Pitch):
        p1 = p1.sounding_pitch
    if isinstance(p2, Pitch):
        p2 = p2.sounding_pitch
    if not 0 <= p1 <= 11 or not 0 <= p2 <= 11:
        raise ValueError(f'Pitches must be between 0 and 11 - got {p1}, {p2}')
    diff = abs(p2 - p1) % 12
    return diff if diff <= 6 else 12 - diff


__all__ = ['Pitch', 'Interval', 'Cycle', 'Scale', 'Scales', 'Tonality']


# Extensions to Scale
# def pitch_is_in_scale(self, pitch: Pitch, enharm=True) -> bool:
#     if not enharm:
#         return pitch.accidentals == 0 and self.mask[pitch.absolute_pitch] == 1
#     return self.mask[pitch.sounding_pitch] == 1

# def chord_is_in_scale(self):
#     pass
