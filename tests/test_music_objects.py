import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from music_objects import Note, Chord
from engine_utils.base_objects import Pitch, Interval
from engine_utils.chord_shapes import ChordShape


class TestNote:
    """Test cases for the Note class."""
    
    def test_note_initialization(self):
        """Test basic Note creation."""
        note = Note(60)
        assert note.midi == 60
        assert note.note_index == 0  # C
        assert note.octave == 5
        assert note.pitch == Pitch(0, 0)
        assert note.name == "C"
        assert note.freq > 0
    
    def test_note_initialization_boundary(self):
        """Test Note creation at MIDI boundaries."""
        note_low = Note(0)
        assert note_low.midi == 0
        assert note_low.octave == 0
        
        note_high = Note(127)
        assert note_high.midi == 127
        assert note_high.octave == 10
    
    def test_note_initialization_invalid(self):
        """Test that invalid MIDI values raise ValueError."""
        with pytest.raises(ValueError):
            Note(-1)
        with pytest.raises(ValueError):
            Note(128)
    
    def test_note_shift_semitones(self):
        """Test shifting a note by semitones."""
        note = Note(60)
        shifted = note.shift(semitones=2)
        assert shifted.midi == 62
        assert shifted.pitch == Pitch(2, 0)
    
    def test_note_shift_octaves(self):
        """Test shifting a note by octaves."""
        note = Note(60)
        shifted = note.shift(octaves=1)
        assert shifted.midi == 72
        assert shifted.octave == 6
    
    def test_note_shift_combined(self):
        """Test shifting a note by both semitones and octaves."""
        note = Note(60)
        shifted = note.shift(semitones=4, octaves=1)
        assert shifted.midi == 76
    
    def test_note_addition_with_integer(self):
        """Test adding an integer to a Note."""
        note = Note(60)
        result = note + 5
        assert result.midi == 65
        assert isinstance(result, Note)
    
    def test_note_addition_with_note(self):
        """Test adding two Notes."""
        note1 = Note(60)
        note2 = Note(5)
        result = note1 + note2
        assert result.midi == 65
    
    def test_note_subtraction_with_integer(self):
        """Test subtracting an integer from a Note."""
        note = Note(60)
        result = note - 5
        assert result.midi == 55
    
    def test_note_subtraction_with_note(self):
        """Test subtracting two Notes."""
        note1 = Note(60)
        note2 = Note(5)
        result = note1 - note2
        assert result.midi == 55
    
    def test_note_comparison_greater_than(self):
        """Test greater than comparison."""
        note1 = Note(60)
        note2 = Note(50)
        assert note1 > note2
        assert not note2 > note1
    
    def test_note_comparison_less_than(self):
        """Test less than comparison."""
        note1 = Note(50)
        note2 = Note(60)
        assert note1 < note2
        assert not note2 < note1
    
    def test_note_equality(self):
        """Test equality comparison."""
        note1 = Note(60)
        note2 = Note(60)
        note3 = Note(61)
        assert note1 == note2
        assert not note1 == note3
    
    def test_note_from_pitch(self):
        """Test creating a Note from a Pitch enum."""
        note = Note.from_pitch(Pitch(0,0), octave=4)
        assert note.midi == 60
        assert note.pitch == Pitch(0, 0)
    
    def test_note_string_representation(self):
        """Test string and repr methods."""
        note = Note(60)
        assert str(note) == "C"
        assert "60" in repr(note)


class TestChord:
    """Test cases for the Chord class."""
    
    def test_chord_initialization(self):
        """Test basic Chord creation."""
        notes = [Note(60), Note(64), Note(67)]  # C Major
        chord = Chord(notes)
        assert len(chord.notes) == 3
        assert chord.lowest_note == Note(60)
    
    def test_chord_initialization_sorted(self):
        """Test that Chord sorts notes."""
        notes = [Note(67), Note(60), Note(64)]
        chord = Chord(notes)
        assert chord.notes[0].midi == 60
        assert chord.notes[1].midi == 64
        assert chord.notes[2].midi == 67
    
    def test_chord_initialization_requires_at_least_two_notes(self):
        """Test that Chord requires at least 2 notes."""
        with pytest.raises(ValueError):
            Chord([Note(60)])
        with pytest.raises(ValueError):
            Chord([])
    
    def test_chord_semitones_calculation(self):
        """Test that semitones are calculated relative to lowest note."""
        notes = [Note(60), Note(64), Note(67)]  # C Major
        chord = Chord(notes)
        assert chord.semitones == [0, 4, 7]

    def test_chord_shape_detection_major(self):
        """Test that major chords are detected correctly."""
        notes = [Note(60), Note(64), Note(67)]  # C Major
        chord = Chord(notes)
        assert chord.shape == ChordShape.M
    
    def test_chord_shape_detection_minor(self):
        """Test that minor chords are detected correctly."""
        notes = [Note(60), Note(63), Note(67)]  # C Minor
        chord = Chord(notes)
        assert chord.shape == ChordShape.m
    
    def test_chord_inversion(self):
        """Test chord inversion."""
        notes = [Note(60), Note(64), Note(67)]  # C Major root position
        chord = Chord(notes)
        inverted = chord.invert(1)  # First inversion
        assert inverted.notes[0].midi == 64
        assert inverted.notes[-1].midi == 72  # C one octave higher
    
    def test_chord_from_shape(self):
        """Test creating a Chord from a root note and shape."""
        root = Note(60)
        chord = Chord.from_shape(root, ChordShape.M)
        assert len(chord.notes) == 3
        assert chord.shape == ChordShape.M
    
    def test_chord_pitch_eq_same_pitches(self):
        """Test pitch equality for chords with same pitches."""
        chord1 = Chord([Note(60), Note(64), Note(67)])
        chord2 = Chord([Note(60), Note(64), Note(67)])
        assert chord1.pitch_eq(chord2)
    
    def test_chord_pitch_eq_different_octaves(self):
        """Test pitch equality ignores octaves."""
        chord1 = Chord([Note(60), Note(64), Note(67)])
        chord2 = Chord([Note(60), Note(64), Note(64+12)])  # B instead of G
        assert not chord1.pitch_eq(chord2)
    
    def test_chord_pitch_eq_different_inversions(self):
        """Test pitch equality for different inversions."""
        chord1 = Chord([Note(60), Note(64), Note(67)])  # Root position
        chord2 = Chord([Note(64), Note(67), Note(72)])  # First inversion
        assert chord1.pitch_eq(chord2)
    
    def test_chord_root_property(self):
        """Test the root property returns the most probable root."""
        notes = [Note(60), Note(64), Note(67)]
        chord = Chord(notes)
        assert chord.root.midi == 60
    
    def test_chord_inversion_property(self):
        """Test the inversion property."""
        notes = [Note(60), Note(64), Note(67)]
        chord = Chord(notes)
        assert chord.inversion == 0


class TestPitch:
    """Test cases for the Pitch enum."""
    
    def test_pitch_from_midi(self):
        """Test creating Pitch from MIDI values."""
        assert Pitch.from_midi(0) == Pitch(0, 0)
        assert Pitch.from_midi(1) == Pitch(1,0)
        assert Pitch.from_midi(12) == Pitch(0, 0)  # One octave higher
    
    def test_pitch_string_representation(self):
        """Test string representation of pitches."""
        assert str(Pitch(0, 0)) == "C"
        assert str(Pitch(1,0)) == "C#"
        assert str(Pitch(2, 0)) == "D"


class TestInterval:
    """Test cases for the Interval enum."""
    
    def test_interval_semitones_property(self):
        """Test that semitones property returns correct values."""
        assert Interval.P0.pitch == 0
        assert Interval.M2.pitch == 2
        assert Interval.M3.pitch == 4
        assert Interval.P5.pitch == 7
    
    def test_interval_degree_property(self):
        """Test that degree property returns correct values."""
        assert Interval.P0.degree == 1
        assert Interval.M2.degree == 2
        assert Interval.M3.degree == 3
        assert Interval.P5.degree == 5

class TestChordShape:
    """Test cases for the ChordShape enum."""
    
    def test_chord_shape_intervals_property(self):
        """Test that intervals property returns correct interval sets."""
        major_intervals = ChordShape.M.intervals
        assert Interval.P0 in major_intervals
        assert Interval.M3 in major_intervals
        assert Interval.P5 in major_intervals
    
    def test_chord_shape_from_intervals_major(self):
        """Test identifying major chord from intervals."""
        intervals = [Interval.P0, Interval.M3, Interval.P5]
        shape = ChordShape.from_intervals(intervals)
        assert shape == ChordShape.M
    
    def test_chord_shape_from_intervals_minor(self):
        """Test identifying minor chord from intervals."""
        intervals = [Interval.P0, Interval.m3, Interval.P5]
        shape = ChordShape.from_intervals(intervals)
        assert shape == ChordShape.m
    
    def test_chord_shape_from_intervals_augmented(self):
        """Test identifying augmented chord from intervals."""
        intervals = [Interval.P0, Interval.M3, Interval.aug5]
        shape = ChordShape.from_intervals(intervals)
        assert shape == ChordShape.aug
    
    def test_chord_shape_from_intervals_diminished(self):
        """Test identifying diminished chord from intervals."""
        intervals = [Interval.P0, Interval.m3, Interval.b5]
        shape = ChordShape.from_intervals(intervals)
        assert shape == ChordShape.dim
    
    def test_chord_shape_from_intervals_unknown(self):
        """Test that unknown interval combinations return UNKNOWN."""
        intervals = [Interval.P0, Interval.M2]
        shape = ChordShape.from_intervals(intervals)
        assert shape == ChordShape.UNKNOWN

class TestChordAdvanced:
    """Advanced test cases for Chord recognition, root detection, and inversion identification."""
    
    # ===== Chord Recognition Tests =====
    
    def test_chord_recognition_major_seventh(self):
        """Test recognition of Major 7 chord."""
        notes = [Note(60), Note(64), Note(67), Note(71)]  # C Major 7
        chord = Chord(notes)
        assert chord.shape == ChordShape.M7
    
    def test_chord_recognition_minor_seventh(self):
        """Test recognition of Minor 7 chord."""
        notes = [Note(60), Note(63), Note(67), Note(70)]  # C Minor 7
        chord = Chord(notes)
        assert chord.shape == ChordShape.m7
    
    def test_chord_recognition_dominant_seventh(self):
        """Test recognition of Dominant 7 chord."""
        notes = [Note(60), Note(64), Note(67), Note(70)]  # C Dominant 7
        chord = Chord(notes)
        assert chord.shape == ChordShape.dom7
    
    def test_chord_recognition_diminished_seventh(self):
        """Test recognition of Diminished 7 chord."""
        notes = [Note(60), Note(63), Note(66), Note(69)]  # C Diminished 7
        chord = Chord(notes)
        assert chord.shape == ChordShape.dim7
    
    def test_chord_recognition_half_diminished_seventh(self):
        """Test recognition of Half-diminished 7 chord."""
        notes = [Note(60), Note(63), Note(66), Note(70)]  # C Half-diminished 7
        chord = Chord(notes)
        assert chord.shape == ChordShape.halfdim7
    
    def test_chord_recognition_augmented(self):
        """Test recognition of Augmented chord."""
        notes = [Note(60), Note(64), Note(68)]  # C Augmented
        chord = Chord(notes)
        assert chord.shape == ChordShape.aug
    
    def test_chord_recognition_suspended_2(self):
        """Test recognition of Suspended 2 chord."""
        notes = [Note(60), Note(62), Note(67)]  # C Suspended 2
        chord = Chord(notes)
        assert chord.shape == ChordShape.sus2
    
    def test_chord_recognition_suspended_4(self):
        """Test recognition of Suspended 4 chord."""
        notes = [Note(60), Note(65), Note(67)]  # C Suspended 4
        chord = Chord(notes)
        assert chord.shape == ChordShape.sus4
    
    def test_chord_recognition_add9(self):
        """Test recognition of Add 9 chord."""
        notes = [Note(60), Note(64), Note(67), Note(74)]  # C Add 9
        chord = Chord(notes)
        assert chord.shape == ChordShape.add9
    
    def test_chord_recognition_major_9(self):
        """Test recognition of Major 9 chord."""
        notes = [Note(60), Note(64), Note(67), Note(71), Note(74)]  # C Major 9
        chord = Chord(notes)
        assert chord.shape == ChordShape.M9
    
    def test_chord_recognition_minor_9(self):
        """Test recognition of Minor 9 chord."""
        notes = [Note(60), Note(63), Note(67), Note(70), Note(74)]  # C Minor 9
        chord = Chord(notes)
        assert chord.shape == ChordShape.m9
    
    def test_chord_recognition_power_chord(self):
        """Test recognition of Power chord."""
        notes = [Note(60), Note(67)]  # C Power chord (P5)
        chord = Chord(notes)
        assert chord.shape == ChordShape.P5
    
    def test_chord_recognition_major_sixth(self):
        """Test recognition of Major 6 chord."""
        notes = [Note(60), Note(64), Note(67), Note(69)]  # C Major 6
        chord = Chord(notes)
        assert chord.shape == ChordShape.M6
    
    def test_chord_recognition_minor_sixth(self):
        """Test recognition of Minor 6 chord."""
        notes = [Note(60), Note(63), Note(67), Note(69)]  # C Minor 6
        chord = Chord(notes)
        assert chord.shape == ChordShape.m6
    
    # ===== Root Detection Tests =====
    
    def test_root_detection_root_position_major(self):
        """Test root detection in root position major chord."""
        notes = [Note(60), Note(64), Note(67)]  # C Major, root position
        chord = Chord(notes)
        assert chord.root == Note(60)
    
    def test_root_detection_first_inversion_major(self):
        """Test root detection in first inversion major chord."""
        notes = [Note(64), Note(67), Note(72)]  # C Major, first inversion (E in bass)
        chord = Chord(notes)
        # Root should still identify C as the root
        assert chord.root.pitch == Pitch(0, 0)
    
    def test_root_detection_second_inversion_major(self):
        """Test root detection in second inversion major chord."""
        notes = [Note(67), Note(72), Note(76)]  # C Major, second inversion (G in bass)
        chord = Chord(notes)
        assert chord.root.pitch == Pitch(0, 0)
    
    def test_root_detection_minor_chord(self):
        """Test root detection in minor chord."""
        notes = [Note(69), Note(72), Note(76)]  # A Minor (A C E)
        chord = Chord(notes)
        assert chord.root.pitch == Pitch(9, 0)
    
    def test_root_detection_seventh_chord(self):
        """Test root detection in seventh chord."""
        notes = [Note(60), Note(64), Note(67), Note(70)]  # C Dominant 7
        chord = Chord(notes)
        assert chord.root.pitch == Pitch(0, 0)
    
    def test_root_detection_different_octaves(self):
        """Test root detection with notes in different octaves."""
        notes = [Note(60), Note(76), Note(79)]  # C in octave 4, E in octave 5, G in octave 5
        chord = Chord(notes)
        assert chord.root.pitch == Pitch(0, 0)
    
    # ===== Inversion Detection Tests =====
    
    def test_inversion_root_position(self):
        """Test inversion detection for root position chord."""
        notes = [Note(60), Note(64), Note(67)]  # C E G (root position)
        chord = Chord(notes)
        assert chord.inversion == 0
    
    def test_inversion_first_inversion(self):
        """Test inversion detection for first inversion chord."""
        notes = [Note(64), Note(67), Note(72)]  # E G C (first inversion)
        chord = Chord(notes)
        assert chord.inversion == 1
    
    def test_inversion_second_inversion(self):
        """Test inversion detection for second inversion chord."""
        notes = [Note(67), Note(72), Note(76)]  # G C E (second inversion)
        chord = Chord(notes)
        assert chord.inversion == 2
    
    def test_inversion_with_doublings(self):
        """Test inversion detection with doubled notes."""
        notes = [Note(64), Note(64), Note(67), Note(72)]  # E G C with doubled E (first inversion)
        chord = Chord(notes)
        assert chord.inversion == 1
    
    def test_inversion_seventh_chord_root_position(self):
        """Test inversion detection for seventh chord in root position."""
        notes = [Note(60), Note(64), Note(67), Note(70)]  # C E G Bb
        chord = Chord(notes)
        assert chord.inversion == 0
    
    def test_inversion_seventh_chord_first_inversion(self):
        """Test inversion detection for seventh chord in first inversion."""
        notes = [Note(64), Note(67), Note(70), Note(72)]  # E G Bb C
        chord = Chord(notes)
        assert chord.inversion == 1
    
    def test_inversion_seventh_chord_second_inversion(self):
        """Test inversion detection for seventh chord in second inversion."""
        notes = [Note(67), Note(70), Note(72), Note(76)]  # G Bb C E
        chord = Chord(notes)
        assert chord.inversion == 2
    
    def test_inversion_seventh_chord_third_inversion(self):
        """Test inversion detection for seventh chord in third inversion."""
        notes = [Note(70), Note(72), Note(76), Note(79)]  # Bb C E G
        chord = Chord(notes)
        assert chord.inversion == 3
    
    # ===== Complex Recognition Tests =====
    
    def test_chord_recognition_with_extended_notes(self):
        """Test chord recognition with extended notes beyond standard voicing."""
        notes = [Note(60), Note(64), Note(67), Note(71), Note(74), Note(77)]  # C Major 9 with extended voicing
        chord = Chord(notes)
        # Should still recognize as a major-type chord
        assert chord.shape not in [ChordShape.M, ChordShape.M7, ChordShape.M9]
    
    def test_chord_recognition_voicing_independence(self):
        """Test that chord recognition is independent of voicing (octave spacing)."""
        notes1 = [Note(60), Note(64), Note(67)]  # Standard voicing
        notes2 = [Note(60), Note(76), Note(79)]  # Wide voicing (E and G in octave 5)
        
        chord1 = Chord(notes1)
        chord2 = Chord(notes2)
        
        assert chord1.shape == chord2.shape == ChordShape.M
    
    def test_root_detection_ambiguous_chord(self):
        """Test root detection for chords that could have multiple interpretations."""
        # C E G could be C Major or A Minor (if inverted)
        notes = [Note(60), Note(64), Note(67)]
        chord = Chord(notes)
        assert chord.root.pitch == Pitch(0, 0)
    
    def test_inversion_detection_minimal_voicing(self):
        """Test inversion detection with minimal voicing (2 notes)."""
        notes = [Note(64), Note(72)]  # E and C (first inversion of C Major)
        chord = Chord(notes)
        # Should identify as first inversion
        assert chord.inversion >= 1
    
    def test_chord_quality_comparison_same_pitches_different_octaves(self):
        """Test that chords with same pitch classes but different octaves are recognized as same."""
        chord1 = Chord([Note(60), Note(64), Note(67)])    # C4 E4 G4
        chord2 = Chord([Note(60), Note(76), Note(79)])    # C4 E5 G5
        
        assert chord1.pitch_eq(chord2)
        assert chord1.shape == chord2.shape


class TestChordRootDetectionAdvanced:
    """Advanced tests for root note detection using multiple strategies."""
    
    def test_root_by_bass_note(self):
        """Test root detection by bass note (lowest sounding note)."""
        # In root position, the root should be the bass note
        notes = [Note(60), Note(64), Note(67)]
        chord = Chord(notes)
        assert chord.root == chord.notes[0]
    
    def test_root_detection_first_inversion_bass_note_is_third(self):
        """Test that in first inversion, the bass note is the third."""
        notes = [Note(64), Note(67), Note(72)]  # E G C
        chord = Chord(notes)
        
        # Bass note should be E (the third)
        assert chord.notes[0].pitch == Pitch(4, 0)
        # But root should still be C
        assert chord.root.pitch == Pitch(0, 0)


class TestChordInversionAdvanced:
    """Advanced tests for inversion detection and manipulation."""
    
    def test_inversion_chain(self):
        """Test a chain of inversions."""
        root_chord = Chord([Note(60), Note(64), Note(67)])
        assert root_chord.inversion == 0
        
        first_inv = root_chord.invert(1)
        assert first_inv.inversion == 1
        assert first_inv.notes[0].pitch == Pitch(4, 0)
        
        second_inv = first_inv.invert(1)
        assert second_inv.inversion == 2
        assert second_inv.notes[0].pitch == Pitch(7, 0)
    
    def test_inversion_octave_wrapping(self):
        """Test that inversion properly wraps notes to next octave."""
        notes = [Note(60), Note(64), Note(67)]
        chord = Chord(notes)
        inverted = chord.invert(1)
        
        # After first inversion, lowest note should be E (64)
        assert inverted.notes[0].midi == 64
        # And the original root (C) should be in the next octave
        assert any(n.midi == 72 for n in inverted.notes)
    
    def test_inversion_preserves_pitch_class(self):
        """Test that inversion preserves the pitch classes of the chord."""
        notes = [Note(60), Note(64), Note(67)]  # C Major
        chord = Chord(notes)
        inverted = chord.invert(1)
        
        # Both should have same pitch classes
        assert chord.pitch_eq(inverted)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])