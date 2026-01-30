import threading
import time
import mido
from functools import cached_property

from src.chordvoyager.constants import ChordShape, Pitch, Interval
import src.chordvoyager.core.freq_ops as freq_op
import src.chordvoyager.core.degree_ops as deg_op
import src.chordvoyager.core.chroma_ops as chr_op



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
        return Chroma(deg_op.to_chroma(self.note_index))


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
            Interval.P0,
            Interval.M3,
            Interval.P5,
            # Interval.from_semitones((i - self.lowest_note).note_index) for i in self.notes
        ]
        self.freq_op.relative_periodicity = freq_op.relative_periodicity(self.semitones)
        self.dissonance = freq_op.smoothed_freq_op.relative_periodicity(self.semitones)
        self._dissonance_raw = freq_op.smoothed_freq_op.relative_periodicity(
            self.semitones, log=False
        )

    def __str__(self):
        return f"Root: MIDI Note {self.root.midi} with frequency {self.root.freq} Hz\n {(self.root.pitch)} {self.shape} Chord in inversion {self.inversion}. Ambiguous root: {self.root_is_ambiguous}"

    @classmethod
    def from_shape(cls, root: Note, chord_shape: ChordShape):
        notes = [root + i.pitch for i in chord_shape.intervals]
        return cls(notes)

    @property
    def _qualities(self) -> list[tuple[Note, ChordShape, int, float]]:
        quality_tuple = []
        for index in range(len(self.notes)):
            inversion = self.invert(index)
            quality = ChordShape.from_intervals(inversion.intervals)
            if quality != ChordShape.UNKNOWN:
                quality_tuple.append(
                    (
                        self.notes[index],
                        quality,
                        (len(self.notes) - index) % len(self.notes),
                        inversion.dissonance,
                    )
                )
        return quality_tuple

    @property
    def _potential_roots(self) -> list[Note]:
        return [q[0] for q in self._qualities]

    @property
    def root_is_ambiguous(self) -> bool:
        if self.lowest_note in self._potential_roots:
            return False
        return len(self._qualities) > 1

    @property
    def most_probable_quality(self) -> tuple[Note, ChordShape, int, float]:
        if self.lowest_note in self._potential_roots:
            return self._qualities[self._potential_roots.index(self.lowest_note)]
        return sorted(self._qualities, key=lambda x: x[3])[-1]

    @property
    def root(self) -> Note:
        return self.most_probable_quality[0]

    @property
    def shape(self) -> ChordShape:
        return self.most_probable_quality[1]

    @property
    def inversion(self) -> int:
        return self.most_probable_quality[2]

    @property
    def root_position_dissonance(self) -> float:
        return self.most_probable_quality[3]

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

    def _play_unthreaded(self, outport, duration: int = 500, velocity: int = 80):
        for n in self.midi:
            outport.send(mido.Message("note_on", note=n, velocity=velocity))
        time.sleep(duration / 1000)
        for n in self.midi:
            outport.send(mido.Message("note_off", note=n, velocity=velocity))

    def play(self, duration: int = 500, velocity: int = 80):
        thread = threading.Thread(
            target=self._play_unthreaded, args=(duration, velocity)
        )
        thread.start()
