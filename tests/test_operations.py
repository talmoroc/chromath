"""Operations relating one object to another.

Transposition and inversion move a chroma; set measures and distances compare
two of them; rank locates a semitone within a cycle. What the objects *are*
lives in test_core.py.
"""

from collections import Counter
from itertools import pairwise, product

import numpy as np
import pytest
from conftest import CYCLE_IDS, SAMPLE_CYCLES, SAMPLE_KEYS, chromas, shifts, tones_of
from hypothesis import given
from utils.fixtures import ALL_KEYS, keys

from chromath.constants import MS
from chromath.core import chroma
from chromath.core.cycle import Cycle

#: Those that obey the triangle inequality.
METRICS = [chroma.hamming, chroma.jaccard]
#: Those that are symmetric. Tversky measures inclusion, so it is not.
SYMMETRIC = [*METRICS, chroma.tversky_symm]
ALL_DISTANCES = [*SYMMETRIC, chroma.tversky]
DISTANCE_IDS = [d.__name__ for d in ALL_DISTANCES]


# ── moving a chroma ─────────────────────────────────────────────────────


def test_transposition_shifts_every_pitch_class():
    chords = keys.chords
    assert chroma.transpose(chords.C_maj, 5) == chords.F_maj
    assert chroma.transpose(chords.C_maj, -7) == chords.F_maj
    assert chroma.transpose(chords.C_min, 2) == chords.D_min
    assert chroma.transpose(chords.C_min, 2 + MS.tones) == chords.D_min

    for key in SAMPLE_KEYS:
        src = tones_of(key)
        for n in range(MS.tones):
            assert tones_of(chroma.transpose(key, n)) == {(i + n) % MS.tones for i in src}


def test_transposition_permutes_the_whole_space():
    """Every shift is a bijection, and twelve of them return home."""
    assert np.array_equal(chroma.transpose(ALL_KEYS, MS.tones), ALL_KEYS)
    for n in range(MS.tones):
        moved = chroma.transpose(ALL_KEYS, n)
        assert set(moved.tolist()) == set(ALL_KEYS.tolist())
        assert np.array_equal(chroma.cardinality(moved), chroma.cardinality(ALL_KEYS))


@given(chromas, shifts, shifts)
def test_transposition_is_additive(c, a, b):
    assert chroma.transpose(chroma.transpose(c, a), b) == chroma.transpose(c, a + b)


def test_inversion_is_its_own_undoing():
    """Inverting twice about the same pivot is the identity."""
    assert np.array_equal(chroma.invert(chroma.invert(ALL_KEYS)), ALL_KEYS)
    assert set(chroma.invert(ALL_KEYS).tolist()) == set(ALL_KEYS.tolist())
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
        assert d(a, c) <= d(a, b) + d(b, c) + 1e-12


@pytest.mark.parametrize("d", ALL_DISTANCES, ids=DISTANCE_IDS)
@given(chromas, chromas)
def test_a_distance_is_invariant_under_transposition(d, a, b):
    for n in range(MS.tones):
        assert d(chroma.transpose(a, n), chroma.transpose(b, n)) == d(a, b)


def test_tversky_symm_trades_the_triangle_inequality_for_inclusion():
    """Alpha above 0.5 makes inclusion cheap, and inclusion chains.

    C and D are a full unit apart, but each is nearly included in {C, D}, so
    the detour is cheaper than the direct route. At alpha 0.5 it is a metric
    and belongs back in METRICS.
    """
    c, c_d, d_ = chroma.from_st(0), chroma.from_st(0, 2), chroma.from_st(2)
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
    with pytest.raises(ValueError):
        chroma.closest(chords.C_maj, candidates, dist="not a distance")  # type: ignore


# ── locating a tone in a cycle ──────────────────────────────────────────


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_rank(step, start):
    """Ranks are signed, so `c[c.rank(t)] == t` whichever way the tone was reached."""
    c = Cycle(step, start)
    for semitone in tones_of(c.mask):
        forward, backward, nearest = (int(c.rank(semitone, d)) for d in ("forward", "backward", "min"))
        assert c[forward] == c[backward] == c[nearest] == semitone
        assert 0 <= forward < c.period
        assert -c.period < backward <= 0
        assert nearest == (forward if forward < abs(backward) else backward)
        if semitone != start:
            assert forward - backward == c.period

    for outside in set(range(MS.tones)) - tones_of(c.mask):
        with pytest.raises(IndexError):
            c.rank(outside)


def test_cycle_rank_anchors():
    fifths = Cycle(7)
    assert fifths.rank(0) == 0
    assert fifths.rank(2, "forward") == 2  # C -> G -> D
    assert fifths.rank(2, "backward") == -10
    assert fifths.rank(2, "min") == 2
    assert fifths.rank(5, "forward") == 11  # F is eleven fifths up
    assert fifths.rank(5, "backward") == -1  # but one fourth down
    assert fifths.rank(5, "min") == -1
    assert Cycle(3).rank(6, "min") == -2  # a tie; backward wins
    with pytest.raises(ValueError):
        fifths.rank(0, "sideways")  # type: ignore


# ── measuring along a cycle ─────────────────────────────────────────────


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_dist(step, start):
    """Steps between two tones the short way round, so never more than half a turn."""
    c = Cycle(step, start)
    inside = sorted(tones_of(c.mask))

    for a, b in product(inside, repeat=2):
        d = int(c.dist(a, b))
        assert d == int(c.dist(b, a))
        assert (d == 0) == (a == b)
        assert 0 <= d <= c.period // 2
        assert d == abs(int(Cycle(step, a).rank(b, "min")))

    for a, b, e in product(inside, repeat=3):
        assert c.dist(a, e) <= c.dist(a, b) + c.dist(b, e)

    for outside in set(range(MS.tones)) - set(inside):
        with pytest.raises(IndexError):
            c.dist(inside[0], outside)
        with pytest.raises(IndexError):
            c.dist(outside, inside[0])


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_dist_counts_steps_along_the_walk(step, start):
    """Walking *d* steps either way from a lands on b."""
    c = Cycle(step, start)
    inside = sorted(tones_of(c.mask))
    for a, b in product(inside, repeat=2):
        here = Cycle(step, a)
        d = int(c.dist(a, b))
        assert b in (int(here[d]), int(here[-d]))
    assert all(c.dist(x, y) == 1 for x, y in pairwise(c[:].tolist()))


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_dist_belongs_to_the_step_not_the_start(step, start):
    """Two cycles over the same tones measure them the same way."""
    c = Cycle(step, start)
    inside = sorted(tones_of(c.mask))
    pairs = list(product(inside, repeat=2))

    for other in inside:
        elsewhere = Cycle(step, other)
        assert elsewhere.mask == c.mask
        assert [elsewhere.dist(a, b) for a, b in pairs] == [c.dist(a, b) for a, b in pairs]

    backwards = Cycle((-step) % MS.tones, start)
    assert [backwards.dist(a, b) for a, b in pairs] == [c.dist(a, b) for a, b in pairs]


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
@pytest.mark.parametrize("shift", [1, 5, 7])
def test_cycle_dist_moves_with_the_music(step, start, shift):
    """Transposing the cycle and both tones together changes nothing."""
    c = Cycle(step, start)
    moved = Cycle(step, (start + shift) % MS.tones)
    for a, b in product(sorted(tones_of(c.mask)), repeat=2):
        assert moved.dist((a + shift) % MS.tones, (b + shift) % MS.tones) == c.dist(a, b)


@pytest.mark.parametrize(("step", "start"), SAMPLE_CYCLES, ids=CYCLE_IDS)
def test_cycle_dist_partitions_the_orbit(step, start):
    """One tone at distance zero, two at each distance after it, one at the antipode."""
    c = Cycle(step, start)
    inside = sorted(tones_of(c.mask))
    at = Counter(int(c.dist(start, t)) for t in inside)

    assert at[0] == 1
    assert all(at[d] == 2 for d in range(1, (c.period + 1) // 2))
    if c.period % 2 == 0 and c.period > 1:
        assert at[c.period // 2] == 1
    assert sum(at.values()) == c.period

    for d in range(len(c.sym)):
        assert {int(t) for t in c.sym[d]} == {t for t in inside if c.dist(start, t) == d}


def test_cycle_dist_anchors():
    fifths = Cycle(7)
    assert fifths.dist(0, 0) == 0
    assert fifths.dist(0, 7) == 1  # C to G
    assert fifths.dist(0, 5) == 1  # C to F, a fifth the other way
    assert fifths.dist(0, 2) == 2  # C -> G -> D
    assert fifths.dist(7, 2) == 1
    assert fifths.dist(0, 6) == 6  # the antipode, as far as a fifth-cycle reaches

    dim = Cycle(3)
    assert dim.dist(0, 3) == 1
    assert dim.dist(0, 9) == 1  # one step backward
    assert dim.dist(0, 6) == 2  # the antipode of a four-step cycle

    assert Cycle(6).dist(0, 6) == 1  # a two-tone cycle has only one distance
    assert Cycle(0, 4).dist(4, 4) == 0  # standstill

    with pytest.raises(IndexError):
        dim.dist(0, 1)


def test_chromatic_dist_is_the_circular_semitone_gap():
    """With step one, rank is pitch and the cycle metric is the familiar one."""
    chromatic = Cycle(1)
    for a, b in product(range(MS.tones), repeat=2):
        gap = abs(a - b)
        assert chromatic.dist(a, b) == min(gap, MS.tones - gap)
