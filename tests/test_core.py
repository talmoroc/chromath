"""The objects of chromath.core: what a chroma is, and what a cycle is.

Representation and conversion for chromas; construction, period, orbit and
indexing for cycles. What those objects *do* to each other lives in
test_operations.py.
"""

import numpy as np
import pytest
from conftest import CYCLE_IDS, SAMPLE_CYCLES, SAMPLE_KEYS, notes, orbit_of, shifts, tones_of
from hypothesis import given
from utils.fixtures import ALL_CHROMAS, ALL_KEYS, fixtures, keys

from chromath.constants import MS
from chromath.core import chroma
from chromath.core.cycle import Cycle

#: step -> how many steps it takes to close.
PERIODS = {0: 1, 1: 12, 2: 6, 3: 4, 4: 3, 5: 12, 6: 2, 7: 12, 8: 3, 9: 4, 10: 6, 11: 12}


# ── chroma ──────────────────────────────────────────────────────────────


def test_a_chroma_is_a_little_endian_twelve_bit_key():
    """Index 0 is C, so pitch class n is bit n."""
    assert chroma.from_st([]) == keys.edge_cases.empty == 0
    assert chroma.from_st(0) == keys.edge_cases.single_C == 1
    assert chroma.from_st(0, 4, 7) == keys.chords.C_maj == 0b000010010001
    assert all(chroma.from_st(n) == 2**n for n in range(MS.tones))
    assert np.array_equal(
        fixtures.chords.C_maj,
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool_),
    )


@given(shifts)
def test_semitones_wrap_at_the_octave(n):
    assert chroma.from_st(n) == chroma.from_st(n % MS.tones)


def test_a_chroma_converts_between_its_three_forms():
    """Key, twelve-lane view and semitones, over the whole space."""
    assert np.array_equal(chroma.to_vector(ALL_KEYS), ALL_CHROMAS)
    assert np.array_equal(chroma.from_vector(ALL_CHROMAS), ALL_KEYS)
    for key in SAMPLE_KEYS:
        assert np.array_equal(chroma.to_st(key), np.flatnonzero(chroma.to_vector(key)))
        assert chroma.from_st(chroma.to_st(key)) == key
        assert set(chroma.to_st(key).tolist()) == tones_of(key)


def test_a_key_outside_twelve_bits_is_rejected():
    assert chroma.cardinality(chroma.validate_chroma(np.uint16(4095))) == MS.tones
    with pytest.raises(ValueError):
        chroma.validate_chroma(np.uint16(1 << MS.tones))


@given(notes())
def test_a_note_is_exactly_one_pitch_class(note):
    assert chroma.cardinality(note) == 1
    assert len(chroma.to_st(note)) == 1


# ── cycle ───────────────────────────────────────────────────────────────


def test_a_cycle_refuses_a_step_or_start_outside_the_system():
    for step in (-1, MS.tones, 24):
        with pytest.raises(ValueError):
            Cycle(step)
    for start in (-1, MS.tones, 13):
        with pytest.raises(ValueError):
            Cycle(1, start)


def test_a_cycle_closes_after_a_period_dividing_the_system():
    """Chromatic and fifths span all twelve; thirds and tritones do not."""
    for step, period in PERIODS.items():
        c = Cycle(step)
        assert c.period == period
        assert MS.tones % c.period == 0
        assert c.is_complete == (period == MS.tones)
        assert Cycle((-step) % MS.tones).period == period


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_a_cycle_covers_the_orbit_of_its_start(step, start):
    """Mask and semitones agree, and neither depends on which way it is walked."""
    c = Cycle(step, start)
    orbit = orbit_of(step, start)
    assert tones_of(c.mask) == orbit
    assert set(c.semitones.tolist()) == orbit
    assert len(orbit) == c.period
    assert start in orbit
    assert Cycle((-step) % MS.tones, start).mask == c.mask
    assert Cycle(step, (start + step) % MS.tones).mask == c.mask


def test_cycle_orbit_anchors():
    assert tones_of(Cycle(1).mask) == set(range(MS.tones))  # chromatic
    assert tones_of(Cycle(2).mask) == {0, 2, 4, 6, 8, 10}  # whole tone
    assert tones_of(Cycle(3).mask) == {0, 3, 6, 9}  # diminished seventh
    assert tones_of(Cycle(4).mask) == {0, 4, 8}  # augmented triad
    assert tones_of(Cycle(6).mask) == {0, 6}  # tritone
    assert tones_of(Cycle(7).mask) == set(range(MS.tones))  # circle of fifths
    assert tones_of(Cycle(0, 4).mask) == {4}  # standstill


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_indexing(step, start):
    """A rank gives one tone, a range of ranks an array; both are unbounded."""
    c = Cycle(step, start)
    assert np.ndim(c[0]) == 0
    assert c[0] == start
    assert [int(c[r]) for r in range(MS.tones)] == [(start + r * step) % MS.tones for r in range(MS.tones)]
    assert [int(c[-r]) for r in range(MS.tones)] == [(start - r * step) % MS.tones for r in range(MS.tones)]

    assert len(c) == c.period
    assert c[:].shape == (c.period,)
    assert [int(t) for t in c] == c[:].tolist()
    assert c[:].tolist() == c.semitones.tolist()

    assert len(c[:25]) == 25
    assert c[: 2 * c.period].tolist() == c[: c.period].tolist() * 2
    assert set(c[:25].tolist()) <= tones_of(c.mask)
    assert set(c[:-25:-1].tolist()) <= tones_of(c.mask)


def test_cycle_indexing_anchors():
    assert Cycle(7)[:4].tolist() == [0, 7, 2, 9]  # C G D A
    assert Cycle(7)[:-4:-1].tolist() == [0, 5, 10, 3]  # C F Bb Eb
    assert Cycle(3, 9)[:5].tolist() == [9, 0, 3, 6, 9]  # wraps after four
    assert Cycle(1, 11)[:3].tolist() == [11, 0, 1]  # wraps past B
    assert Cycle(0, 4)[:3].tolist() == [4, 4, 4]  # standstill
    assert Cycle(7)[:].tolist() == [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_symmetric_indexing(step, start):
    """`sym` is a sequence of pairs, so its result is always (..., 2)."""
    c = Cycle(step, start)
    assert len(c.sym) == c.period // 2 + 1
    assert c.sym[:].shape == (len(c.sym), 2)
    assert c.sym[2].tolist() == [int(c[2]), int(c[-2])]
    assert c.sym[1:3].tolist() == [c.sym[1].tolist(), c.sym[2].tolist()]
    assert [pair.tolist() for pair in c.sym] == c.sym[:].tolist()
    assert set(c.sym[:].ravel().tolist()) == tones_of(c.mask)

    doubled = {d for d in range(len(c.sym)) if c.sym[d][0] == c.sym[d][1]}
    antipode = {c.period // 2} if c.period % 2 == 0 else set()
    assert doubled == {0} | antipode


def test_cycle_symmetric_anchors():
    assert Cycle(7).sym[:].tolist() == [[0, 0], [7, 5], [2, 10], [9, 3], [4, 8], [11, 1], [6, 6]]
    assert Cycle(3, 9).sym[:].tolist() == [[9, 9], [0, 6], [3, 3]]
    assert Cycle(4).sym[:].tolist() == [[0, 0], [4, 8]]  # odd period, no antipode
    assert Cycle(0, 4).sym[:].tolist() == [[4, 4]]  # standstill
