"""The objects of chromath.core: what a chroma is, and what a cycle is."""

import numpy as np
import pytest
from conftest import (
    BASED_IDS,
    BASES,
    SAMPLE_BASED,
    SAMPLE_KEYS,
    chroma_of,
    each,
    members_of,
    note_of,
    notes,
    period_of,
    start_of,
    tones_of,
    turn,
)
from hypothesis import given
from utils.fixtures import ALL_CHROMAS, ALL_KEYS, fixtures, keys

from chromath.constants import MS
from chromath.core import chroma, cycle
from chromath.types import DT, SENTINEL

#: step -> how many steps it takes to close, starting on a single note.
PERIODS = {0: 1, 1: 12, 2: 6, 3: 4, 4: 3, 5: 12, 6: 2, 7: 12, 8: 3, 9: 4, 10: 6, 11: 12}


# ── chroma ──────────────────────────────────────────────────────────────


def test_a_chroma_is_a_little_endian_twelve_bit_key():
    """Index 0 is C, so pitch class n is bit n."""
    assert chroma_of([]) == keys.edge_cases.empty == 0
    assert note_of(0) == keys.edge_cases.single_C == 1
    assert chroma_of([0, 4, 7]) == keys.chords.C_maj == 0b000010010001
    assert note_of(range(MS.tones)).tolist() == [2**n for n in range(MS.tones)]
    assert np.array_equal(
        fixtures.chords.C_maj,
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool_),
    )


def test_from_members_reads_one_chroma_per_row():
    """The last axis lists the members; the others are the batch."""
    chords = keys.chords
    assert chroma_of([0, 4, 7]).shape == ()
    assert chroma_of([[0, 4, 7], [0, 3, 7]]).tolist() == [int(chords.C_maj), int(chords.C_min)]
    assert chroma_of([[0], [7], [2]]).tolist() == [1 << 0, 1 << 7, 1 << 2]  # one note each
    assert chroma.from_members(np.zeros((2, 3, 4), dtype=DT.Note)).shape == (2, 3)
    assert chroma.from_members(np.zeros((0, 3), dtype=DT.Note)).shape == (0,)
    assert chroma_of([4, 0, 7, 4]) == chords.C_maj  # order and repeats do not matter


def test_from_members_skips_padding_so_chromas_of_any_size_share_a_batch():
    """SENTINEL slots hold no note."""
    chords = keys.chords
    ragged = [[0, 4, 7, 10], [0, 4, 7, SENTINEL], [SENTINEL] * 4]
    assert chroma_of(ragged).tolist() == [int(chords.C_dom7), int(chords.C_maj), 0]
    fits = ALL_KEYS[chroma.cardinality(ALL_KEYS) <= MS.max_chroma_members]
    assert np.array_equal(chroma.from_members(chroma.to_members(fits)), fits)


def test_a_note_converts_between_its_key_and_its_index():
    """from_st and to_st convert single notes."""
    every = chroma.validate_note_index_array(range(MS.tones))
    assert chroma.from_st(every).dtype == DT.Key
    assert chroma.to_st(chroma.from_st(every)).tolist() == list(range(MS.tones))
    assert chroma.to_st(chroma.from_st(every)).dtype == DT.Note
    assert chroma.to_st(chroma.from_st(every.reshape(3, 4))).shape == (3, 4)
    for got in (note_of(7), chroma.to_st(note_of(7))):
        assert isinstance(got, np.ndarray) and got.ndim == 0


def test_a_chroma_converts_between_its_three_forms():
    """Key, twelve-lane view and semitones, over the whole space."""
    assert np.array_equal(chroma.to_vector(ALL_KEYS), ALL_CHROMAS)
    assert np.array_equal(chroma.from_vector(ALL_CHROMAS), ALL_KEYS)
    for key in each(SAMPLE_KEYS):
        members = chroma.to_members(key, width=MS.tones)
        assert np.array_equal(members[members != SENTINEL], np.flatnonzero(chroma.to_vector(key)))
        assert chroma.from_members(members) == key
        assert set(members[members != SENTINEL].tolist()) == tones_of(key)


@given(notes())
def test_a_note_is_exactly_one_pitch_class(note):
    assert chroma.cardinality(note) == 1
    members = chroma.to_members(note)
    assert len(members[members != SENTINEL]) == 1
    assert members[0] == chroma.to_st(note)


# ── entry validation ────────────────────────────────────────────────────


def test_validation_returns_the_array_the_core_expects():
    for got, dtype in (
        (chroma.validate_chroma_keys(4095), DT.Key),
        (chroma.validate_chroma_keys([0, 1, 145]), DT.Key),
        (chroma.validate_note_keys([1, 2, 2048]), DT.Key),
        (chroma.validate_note_index_array(11), DT.Note),
        (chroma.validate_chroma_members_array([[0, 4, 7], [0, 7, SENTINEL]]), DT.Note),
        (chroma.validate_chroma_vecs([0, 1] * 6), DT.Bool),
        (chroma.validate_note_vecs([1] + [0] * 11), DT.Bool),
    ):
        assert isinstance(got, np.ndarray) and got.dtype == dtype
    assert chroma.cardinality(chroma.validate_chroma_keys(4095)) == MS.tones


def test_a_key_outside_twelve_bits_is_rejected():
    for bad in (1 << MS.tones, -1, [0, 5000]):
        with pytest.raises(ValueError):
            chroma.validate_chroma_keys(bad)
    with pytest.raises(TypeError):
        chroma.validate_chroma_keys(1.0)


def test_a_note_key_must_hold_exactly_one_note():
    for bad in (0, 3, [1, 2, 145]):
        with pytest.raises(ValueError):
            chroma.validate_note_keys(bad)


def test_a_note_index_outside_the_octave_is_rejected():
    """Nothing wraps: 12 is an error."""
    for bad in (MS.tones, -1, [0, 12]):
        with pytest.raises(ValueError):
            chroma.validate_note_index_array(bad)
    with pytest.raises(TypeError):
        chroma.validate_note_index_array(0.5)


def test_members_must_be_pitch_classes_or_padding():
    for bad in ([0, MS.tones], [0, -1], 3):  # the last has no member axis
        with pytest.raises(ValueError):
            chroma.validate_chroma_members_array(bad)
    with pytest.raises(TypeError):
        chroma.validate_chroma_members_array([0.5, 4.0])


def test_a_vector_must_be_twelve_lanes_of_zeros_and_ones():
    with pytest.raises(ValueError):
        chroma.validate_chroma_vecs([0, 1, 0])
    with pytest.raises(ValueError):
        chroma.validate_chroma_vecs([2] + [0] * 11)
    with pytest.raises(ValueError):
        chroma.validate_note_vecs([1, 1] + [0] * 10)


def test_a_single_value_narrows_to_a_hashable_scalar():
    """as_key and as_note_index give scalars a class can store."""
    assert hash(chroma.as_key(note_of(0))) == hash(1)
    with pytest.raises(ValueError):
        chroma.as_key(ALL_KEYS[:2])
    with pytest.raises(ValueError):
        chroma.as_note_index(chroma.validate_note_index_array([0, 7]))


# ── cycle ───────────────────────────────────────────────────────────────


def test_a_cycle_closes_when_its_start_returns():
    """A symmetric start returns before the full orbit."""
    for step, period in PERIODS.items():
        assert cycle.period(step, note_of(0)) == period
        assert cycle.period((-step) % MS.tones, note_of(0)) == period

    for name, base in BASES.items():
        for step in range(MS.tones):
            assert cycle.period(step, np.asarray(base)) == period_of(step, int(base))
            assert cycle.period(step, start_of(5, name)) == period_of(step, int(base))  # where it starts is irrelevant


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_one_turn_is_the_start_transposed_step_by_step(step, start, name):
    """Rank r holds the start moved up r steps; no member repeats."""
    base = BASES[name]
    members = turn(step, start, name)

    assert members.tolist() == members_of(step, start, int(base))
    assert len(set(members.tolist())) == len(members) == period_of(step, int(base))
    for r, member in enumerate(members):
        assert tones_of(member) == {(t + start + r * step) % MS.tones for t in tones_of(base)}


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_a_cycle_masks_every_tone_it_touches(step, start, name):
    """The mask is the union over one turn, whatever the direction or start."""
    members = turn(step, start, name)
    mask = cycle.mask(members)
    assert tones_of(mask) == set().union(*(tones_of(x) for x in members))
    assert cycle.mask(turn((-step) % MS.tones, start, name)) == mask
    assert cycle.mask(turn(step, (start + step) % MS.tones, name)) == mask
    assert cycle.is_complete(members) == (tones_of(mask) == set(range(MS.tones)))


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_semitones_spells_out_every_member(step, start, name):
    """One row per rank, padded or one column per note."""
    members = turn(step, start, name)
    size = int(chroma.cardinality(np.asarray(BASES[name])))

    tight = cycle.semitones(members, padded=False)
    assert tight.shape == (len(members), size)
    for row, member in zip(tight, members, strict=True):
        assert set(row.tolist()) == tones_of(member)

    padded = cycle.semitones(members)
    assert padded.shape == (len(members), MS.max_chroma_members)
    assert np.array_equal(chroma.from_members(padded), members)


def test_cycle_anchors():
    assert turn(7).tolist() == note_of([0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]).tolist()  # the circle of fifths
    assert turn(3, 9).tolist() == note_of([9, 0, 3, 6]).tolist()
    assert turn(0, 4).tolist() == note_of([4]).tolist()  # standstill
    assert [tones_of(x) for x in turn(4, 0, "major")] == [{0, 4, 7}, {4, 8, 11}, {8, 0, 3}]  # C, E, Ab
    assert [tones_of(x) for x in turn(1, 0, "whole_tone")] == [{0, 2, 4, 6, 8, 10}, {1, 3, 5, 7, 9, 11}]
    assert turn(3, 0, "dim7").tolist() == [int(BASES["dim7"])]  # one chord, not four positions holding it

    assert cycle.period(7, start_of(0, "major")) == MS.tones  # twelve major triads
    assert cycle.period(7, start_of(0, "whole_tone")) == 2  # only two whole-tone scales
    assert cycle.period(2, start_of(0, "tritone")) == 3

    assert tones_of(cycle.mask(turn(2))) == {0, 2, 4, 6, 8, 10}  # whole tone
    assert tones_of(cycle.mask(turn(3))) == {0, 3, 6, 9}  # diminished seventh
    assert tones_of(cycle.mask(turn(3, 0, "dim7"))) == {0, 3, 6, 9}  # one chord, four tones
    assert cycle.is_complete(turn(7)) and not cycle.is_complete(turn(3))

    assert cycle.semitones(turn(4, 0, "major"), padded=False).tolist() == [[0, 4, 7], [4, 8, 11], [0, 3, 8]]
    assert cycle.semitones(turn(6)).tolist() == [[0] + [SENTINEL] * 6, [6] + [SENTINEL] * 6]
