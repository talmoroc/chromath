from enum import Enum

from base_objects import *

class DegreeChordsFromScale:
    # Similar in terms of common notes, shape, distance to the cycles
    # common notes = equivalence between chords. (Also check scale transposition similarity ?)
    # shape = change of feeling when switching
    # cycle = feeling of closeness, root movement, logical movement
    # From this we can deduce the function of degrees in the scale.
    pass


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

    # @classmethod
    # def from_notes(cls, notes: list[Note]) -> ChordShape:
    #     if not notes:
    #         return cls.UNKNOWN

    #     sorted_midi = sorted([n.midi for n in notes])
    #     root_midi = sorted_midi[0]
    #     intervals = [Intervals.from_semitones(
    #         m - root_midi) for m in sorted_midi]

    #     return cls.from_intervals(intervals)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def intervals(self) -> set[Interval]:
        return self.value

    # @property
    # def similar(self):
    #     similar_shapes: set[ChordShape] = set()
    #     similar_intervals: set[Intervals] = set()
    #     for interval in self.intervals:
    #         similar_intervals = similar_intervals.union({Intervals.P0})  # TODO

    #     for shape in ChordShape:
    #         if shape.intervals <= similar_intervals and len(shape.intervals) == len(self.intervals):
    #             similar_shapes.add(shape)

    #     return similar_shapes

    # Distinguer la distance en termes de semitones pure, prenant en compte la hauteur (!!! Elle doit être en valeur absolue)
    # et la distance en termes de gamme, qui se répète tous les 12 demi-tons.
    # Cette distance est circulaire, mais aussi ambigue (on peut prendre le degré à l'octave supérieure ou inférieure)
    # Ce qui fait qu'elle a toujours deux valeurs. Elle est non-symétrique.
    # Je ne sais pas quoi faire de ça : C-F != F-C, F-C = C-G et C-F = G-C

    # Le problème que j'ai, c'est que l'objet Note rejoint très vite celui de Pitch.
    # C'est un Pitch auquel on a ajouté une octave. Est-ce que Pitch est de trop ?
    # Mais Pitch permet de créer des intervalles, des formes d'accords.

    # De ce fait, les intervalles et les formes d'accord ont le même problème : octave-équivalence ou non ?
    # La notion d'octave équivalence rend compliqué le calcul des renversements...
    # Un accord doit se définir par les intervalles à sa fondamentale, octave-équivalents
    # Cependant il faut distinguer 2de et 9ème, Quarte et 11ème, Sixte et 13ème. Je suis donc bloqué

    # TODO: detection of pitch from midi should be done at the tonality level
    # @classmethod
    # def from_midi(cls, midi_note: int, sharp: bool = True):
    #     note_index = midi_note % 12
    #     if note_index in Scales.IONIAN.semitones:
    #         return cls(Letter(Scales.IONIAN.semitones.index(note_index)))
    #     degree = [i for i in range(
    #         len(Scales.IONIAN)) if note_index > Scales.IONIAN.semitones[i]][-1]
    #     accidental = sharp * (note_index - degree) + \
    #         (not sharp) * (degree - note_index)
    #     return cls(Letter(degree), accidental)


# class Letter(Enum):
#     C = 1
#     D = 2
#     E = 3
#     F = 4
#     G = 5
#     A = 6
#     B = 7


# @dataclass(frozen=True)
# class Pitch:
#     """
#     Class to handle traditional pitch notation, with letters and accidentals.
#     Not used for now, accidentals will be dealt with at the level of the Note object
#     and a pitch will be considered a number between 0 and 11.
#     Degrees will be the index of the list of notes in a Scale.
#     """
#     letter: Letter
#     accidental: int

#     @property
#     def semitones(self) -> int:
#         return (Scales.IONIAN.pitches[self.letter.value] + self.accidental) % 12

#     @property
#     def degree(self) -> int:
#         return self.letter.value

#     def __str__(self):
#         acc = "#" * self.accidental if self.accidental > 0 else "b" * \
#             abs(self.accidental)
#         return f"{self.letter.name}{acc}"

#     def __repr__(self):
#         return f"<{self.__class__.__name__}: {self.__str__()}>"


# Intervalle = semitones
# Interval in a given scale from a given pitch to a given pitch with alterations = semitones + degree
# PRoblem if the starting pitch has alterations => symmetrical ?
# Symmetrical intervals

# Example
# Interval1 =  3 semitones
# A scale gives us degrees. Pitches are inside or outside the scale.
# MAJOR : [1,0,1,0,1,1,0,1,0,1,0,1] -> CM[3] not inside the scale. On a CM[4]
# MAJOR[3] = 4 mais MAJOR.mode(x)[3] = 3 pour x in [1, 4, 5]
# ou (non implémenté) Interval(Scales.MAJOR, 0, 3) = 4 mais Interval(Scales.MAJOR, x, x + 3) = 3 pour x in [1, 4, 5]
# index = degree, value = semitones

# In a Scale there are no alterations. In lydian the 4th is 6 semitones. In major the 3d is 4. In minor the 3d is 3.

# Let's add accidentals.
# MAJOR[3] = (4, 3) and MAJOR.mode(2)[3] = (3, 3) and MINOR[3] =

# Il n'y a pas d'intervalles naturellement altérés. Il en existe qui sont considérés comme tels car ils
# n'apparaissent que rarement ou jamais dans les gammes classiques (aug6, b8, aug5, b7) donc la plupart
# du temps on les observe en tant qu'altérations. Mais pas de différence fondamentale avec m3 en majeur.


# The big problem I'm facing now is that a pitch is between 0 and 11, but that means a pitch 2 is a 9nth or a 2d.
# Also the degrees are between 1 and 7, but a 9nth or 13th is a degree 9 or degree 13.
# The rule is actually in notation, that there cannot be more than 2 consecutive degrees :
    # if there is C(I) + E(III) and a D, il will not be D(II) but D(IX)
    # If there is C(I) + G(V) and a D or F, it will be D(II) of F(IV) and not D(IX) not F(XI)
    # If there is C(I) + E(III) + G(V) and a A, it will be A(VI) and not A(XIII)
    # If there is C(I) + E(III) + G(V) + B(VII) and an A, it will be A(XIII) and not A(VI)

# Solution : You can deduce extensions from the reduced form of a Chord => OK
# However there needs to be 2 degrees notions : one that goes from 1 to 7 (scale, chords)
# and one that goes from 1 to 13 (extensions in a chord).


# Consequence : intervals are just (degree, semitones)
# No notion of alteration of the Pitch or the interval
    # Interval(Pitch(0, 0), Pitch(0, 4)) == Interval(Pitch(0, 4), Pitch(1, 7))
    # Interval(Pitch(0, 0), Pitch(13, 0)) == Interval(Pitch(0,0), Pitch(0, 2))
    # Wait : no notion of degree here, just semitones.

# How do I name them ?
# Semitones Interval
# Scale-dependent intervals

# Only one interval but with a function to return the degree in the scale, or the functional degree (up to 13) ?


# Clarification of the objects and the structure to access easily to what I want :
    # Know which notes are altered within a chord, and the direction of the alteration
        # C Eb G -> Eb = Pitch(4, -1) = Pitch(Scale[3], -1), diminished third degree.
    # Know which degrees are in the chord, relative to the root
        # F A C -> (5, 9, 0) => (0, 4, -5) % 12 = (0, 4, 7) in semitones
        # F A C -> (4, 6, 1) => (0, 2, -3) % 7 + 1 = (1, 3, 4) in degrees (or use scale.shift(4))
    # This allows to know which intervals are in the chord.
        # F A C -> root, M3, P5. No alteration from the scale.

    # Know if a chord is a degree of the scale
        # F A C -> root F = semitones 5, Scale[4] = 5, degree = 4
    # Know the shape of a given chord
        # A shape is a set of intervals.
        # A shape needs to consider the chords in their reduced form.
        # Root + M3 + P5 + M2 = add9 chord. cf. rule higher.
        # Here the interval object has a degree between 1 and 7. However its interpretation is between 1 and 13.
        # It may be an object "FunctionalInterval" that has an Interval as property, and a "standard position from root".


# CONCLUSION
# No need for an interval class that is just semitones and degrees.
# It is impossible to name the intervals if you don't know the scale.
# It is impossible to deduce the intervals if you don't know the scale.
# It is impossible to create an interval from pitches:
    # Interval(Pitch(0,0), Pitch(3,4)) et Interval(Pitch(0,0), Pitch(4,-1))
    # In the context of a minor scale, both are standard intervals.
    # In the context of a major scale, both are altered intervals but the first one is ambiguous (augmented 2d or bemolized third)

    # Interval(Pitch(0,0), Pitch(8,-1)) et Interval(Pitch(0,0), Pitch(7,1))
    # The first one is a minor 6th, the second one is an augmented 5th
    # If aeolian scale, the first one is not altered, the second one is altered.

    # SOOOO to get a good representation, you need to have Pitch alterations relative to degrees of a scale.

# The Pitch class is Scale-dependent.

# The Interval class is Scale-dependent if you want to find an interval from Pitches.
# Except if a Pitch is a degree of the scale (1-7) + an alteration.
# But then Pitch(1, 3) = major third ? Or fourth. Depends on the scale.
# Then the notes, if they are build on pitches, will also depend on scales...
# So it will be hard to analyse them as belonging to different scales.
# C# = Note(Pitch(3,1,Scale.AEOLIAN), root=A, octave=1) == Note(Pitch(0,0,Scale.IONIAN), root=C#, octave=1)


# TODO : calculer un intervalle n'est possible qu'en connaissant les degrés.
    # @classmethod
    # def between(cls, note1: Note, note2: Note) -> Interval:
    #     degree = 0  # TODO: deduce degree from note tonality ?
    #     semitones = note1.note_value - note2.note_value
    #     return cls(degree, semitones)



def compute_chord_shape(intervals: list[int]) -> ChordShape:
    chord_shape = ChordShape(intervals)
    if chord_shape is not None:
        return chord_shape
    return ChordShape.UNKNOWN