"""Operations relating one object to another."""

from itertools import pairwise, product

import numpy as np
import pytest
from conftest import (
    BAD_CHOICE,
    BASED_IDS,
    BASES,
    SAMPLE_BASED,
    SAMPLE_KEYS,
    chroma_of,
    chromas,
    each,
    foreign_to,
    note_of,
    shifts,
    tones_of,
    turn,
)
from hypothesis import given
from utils.fixtures import ALL_KEYS, keys

from chromath.constants import MS
from chromath.core import chroma, cycle
from chromath.core.chroma import ALL_DISTANCES, DISTANCE_IDS, METRICS, SYMMETRIC
from chromath.types import DT

# ── moving a chroma ─────────────────────────────────────────────────────


def test_transposition_shifts_every_pitch_class():
    chords = keys.chords
    assert chroma.transpose(chords.C_maj, 5) == chords.F_maj
    assert chroma.transpose(chords.C_maj, -7) == chords.F_maj
    assert chroma.transpose(chords.C_min, 2) == chords.D_min
    assert chroma.transpose(chords.C_min, 2 + MS.tones) == chords.D_min

    for key in each(SAMPLE_KEYS):
        src = tones_of(key)
        for n in range(MS.tones):
            assert tones_of(chroma.transpose(key, n)) == {(i + n) % MS.tones for i in src}


def test_transposition_permutes_the_whole_space():
    """Every shift is a bijection, and twelve of them return home."""
    assert np.array_equal(chroma.transpose(ALL_KEYS, MS.tones), ALL_KEYS)
    for n in range(MS.tones):
        moved = np.asarray(chroma.transpose(ALL_KEYS, n))
        assert set(moved.tolist()) == set(ALL_KEYS.tolist())
        assert np.array_equal(chroma.cardinality(moved), chroma.cardinality(ALL_KEYS))


@given(chromas, shifts, shifts)
def test_transposition_is_additive(c, a, b):
    assert chroma.transpose(chroma.transpose(c, a), b) == chroma.transpose(c, a + b)


def test_inversion_is_its_own_undoing():
    """Inverting twice about the same pivot is the identity."""
    assert np.array_equal(chroma.invert(chroma.invert(ALL_KEYS)), ALL_KEYS)
    assert set(np.asarray(chroma.invert(ALL_KEYS).tolist())) == set(ALL_KEYS.tolist())
    assert np.array_equal(chroma.cardinality(chroma.invert(ALL_KEYS)), chroma.cardinality(ALL_KEYS))
    assert chroma.invert(keys.chords.C_maj, 7) == keys.chords.C_min


# ── comparing two chromas ───────────────────────────────────────────────


def test_set_measures_count_shared_and_divergent_tones():
    chords, scales = keys.chords, keys.scales
    assert chroma.cardinality(chords.C_maj) == 3
    assert chroma.cardinality(keys.edge_cases.empty) == 0
    assert chroma.common_tones(chords.C_maj, chords.C_maj7) == 3
    assert chroma.diverging_tones(chords.C_maj, chords.C_maj7) == 1
    assert chroma.isin(chords.C_maj, scales.C_major)
    assert not chroma.isin(chords.Csharp_maj, scales.C_major)


@pytest.mark.parametrize("d", ALL_DISTANCES, ids=DISTANCE_IDS)
@given(chromas, chromas, chromas)
def test_distance_axioms(d, a, b, c):
    """Identity and range for every distance; symmetry and triangle where claimed."""
    assert d(a, a) == 0
    assert 0.0 <= d(a, b) <= 1.0
    assert d(keys.edge_cases.empty, keys.edge_cases.empty) == 0
    if d in SYMMETRIC:
        assert d(a, b) == d(b, a)
    if d in METRICS:
        assert d(a, c)[()] <= d(a, b) + d(b, c) + 4 * np.finfo(DT.Score).eps  # equality case rounds in float32


@pytest.mark.parametrize("d", ALL_DISTANCES, ids=DISTANCE_IDS)
@given(chromas, chromas)
def test_a_distance_is_invariant_under_transposition(d, a, b):
    for n in range(MS.tones):
        assert d(chroma.transpose(a, n), chroma.transpose(b, n)) == d(a, b)


def test_tversky_symm_trades_the_triangle_inequality_for_inclusion():
    """Above alpha 0.5, symmetric Tversky is not a metric."""
    c, c_d, d_ = note_of(0), chroma_of([0, 2]), note_of(2)
    assert chroma.tversky_symm(c, d_) == 1.0
    assert chroma.tversky_symm(c, c_d) + chroma.tversky_symm(c_d, d_) < 1.0

    at_half = chroma.tversky_symm(c, c_d, alpha=0.5) + chroma.tversky_symm(c_d, d_, alpha=0.5)
    assert at_half >= chroma.tversky_symm(c, d_, alpha=0.5)


def test_closest_ranks_candidates_by_distance():
    chords = keys.chords
    candidates = np.array([chords.C_min, chords.Csharp_maj, chords.C_maj, chords.F_maj], dtype=np.uint16)
    assert chroma.closest(chords.C_maj, candidates) == 2
    assert chroma.closest(chords.C_maj, candidates, n=3).tolist() == [2, 0, 3]
    assert chroma.closest(chords.C_maj, candidates, dist="jaccard") == 2
    with pytest.raises(BAD_CHOICE):
        chroma.closest(chords.C_maj, candidates, dist="not a distance")  # type: ignore


# ── locating a chroma in a cycle ────────────────────────────────────────


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_cycle_rank(step, start, name):
    """Ranks are signed: `members[rank(x)] == x` either way."""
    members = turn(step, start, name)
    period = len(members)
    for r, member in enumerate(each(members)):
        forward, backward, nearest = (int(cycle.rank(members, member, d)) for d in ("forward", "backward", "min"))
        assert forward == r
        assert int(members[backward]) == int(members[nearest]) == int(member)
        assert -period < backward <= 0
        assert nearest == (forward if forward < abs(backward) else backward)

    with pytest.raises(IndexError):
        cycle.rank(members, foreign_to(BASES[name]))


def test_cycle_rank_anchors():
    fifths = turn(7)
    assert cycle.rank(fifths, note_of(2), "forward") == 2  # C -> G -> D
    assert cycle.rank(fifths, note_of(2), "backward") == -10
    assert cycle.rank(fifths, note_of(5), "forward") == 11  # F is eleven fifths up
    assert cycle.rank(fifths, note_of(5), "min") == -1  # but one fourth down
    assert cycle.rank(turn(3), note_of(6), "min") == -2  # a tie; backward wins
    assert cycle.rank(turn(7, 0, "major"), chroma_of([0, 5, 9])) == -1  # F major, one fourth down
    with pytest.raises(BAD_CHOICE):
        cycle.rank(fifths, note_of(0), "sideways")  # type: ignore


# ── measuring along a cycle ─────────────────────────────────────────────


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_cycle_dist_is_a_metric_counting_steps(step, start, name):
    """Steps between two members the short way round."""
    members = turn(step, start, name)
    inside = each(members)

    for a, b in product(inside, repeat=2):
        d = int(cycle.dist(members, a, b))
        here = cycle.cycle(step, a)
        assert int(b) in (int(here[d]), int(here[-d]))
        assert d == int(cycle.dist(members, b, a))
        assert (d == 0) == (int(a) == int(b))
        assert 0 <= d <= len(members) // 2
    assert all(cycle.dist(members, x, y) == 1 for x, y in pairwise(inside))

    for a, b, e in product(inside, repeat=3):
        assert cycle.dist(members, a, e) <= cycle.dist(members, a, b) + cycle.dist(members, b, e)

    with pytest.raises(IndexError):
        cycle.dist(members, inside[0], foreign_to(BASES[name]))


def test_cycle_dist_anchors():
    fifths = turn(7)
    assert cycle.dist(fifths, note_of(0), note_of(7)) == 1  # C to G
    assert cycle.dist(fifths, note_of(0), note_of(5)) == 1  # C to F, a fifth the other way
    assert cycle.dist(fifths, note_of(0), note_of(2)) == 2  # C -> G -> D
    assert cycle.dist(fifths, note_of(0), note_of(6)) == 6  # the antipode, as far as a fifth-cycle reaches
    assert cycle.dist(turn(3), note_of(0), note_of(9)) == 1  # one step backward
    assert cycle.dist(turn(0, 4), note_of(4), note_of(4)) == 0  # standstill

    triads = turn(7, 0, "major")
    assert cycle.dist(triads, chroma_of([0, 4, 7]), chroma_of([0, 5, 9])) == 1  # C to F major
    assert cycle.dist(triads, chroma_of([0, 4, 7]), chroma_of([2, 6, 9])) == 2  # C to D major

    chromatic = turn(1)  # with step one, the cycle metric is the circular semitone gap
    for a, b in product(range(MS.tones), repeat=2):
        assert cycle.dist(chromatic, note_of(a), note_of(b)) == min(abs(a - b), MS.tones - abs(a - b))
