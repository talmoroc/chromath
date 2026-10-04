"""The public classes built on chromath.core."""

import numpy as np
import pytest
from conftest import BASED_IDS, BASES, SAMPLE_BASED, chroma_of, members_of, note_of, start_of, tones_of, turn

from chromath.constants import MS
from chromath.core import chroma, cycle
from chromath.objects.cycle_object import Cycle

# ── construction ────────────────────────────────────────────────────────


def test_a_cycle_refuses_what_is_not_a_step_and_one_chroma_key():
    for step in (-1, MS.tones, 24):
        with pytest.raises(ValueError):
            Cycle(step)
    for start in (-1, 1 << MS.tones, [1, 2]):
        with pytest.raises(ValueError):
            Cycle(7, start)  # type: ignore
    with pytest.raises(TypeError):
        Cycle(7, 1.0)  # type: ignore


def test_a_cycle_is_a_hashable_value_with_one_spelling():
    """The start is stored as a ChromaKey, so cycles hash and compare."""
    major = start_of(0, "major")  # a 0-d array, as the core returns it
    d_major = chroma_of([2, 6, 9])
    assert Cycle(7) == Cycle(7, note_of(0))  # C unless told otherwise
    assert Cycle(7, major) == Cycle(7, int(major)) == Cycle(7, BASES["major"])
    assert Cycle(7, chroma.transpose(major, 2)) == Cycle(7, d_major)
    assert Cycle(7, d_major) != Cycle(7, major)  # same members, another walk
    assert len({Cycle(7, major), Cycle(7, int(major)), Cycle(7)}) == 2


# ── agreement with the core ─────────────────────────────────────────────


@pytest.mark.parametrize(("step", "start", "name"), [(7, 0, "unison"), (3, 9, "dom7"), (1, 0, "whole_tone")])
def test_a_cycle_reports_what_the_core_computes(step, start, name):
    c = Cycle(step, start_of(start, name))
    members = turn(step, start, name)

    assert np.array_equal(c.members, members)
    assert c.period == len(c) == len(members)
    assert c.mask == cycle.mask(members)
    assert c.is_complete == cycle.is_complete(members)
    assert np.array_equal(c.semitones(), cycle.semitones(members))
    assert np.array_equal(c.rank(members, "backward"), cycle.rank(members, members, "backward"))
    assert np.array_equal(c.dist(members[:, np.newaxis], members), cycle.dist(members, members[:, np.newaxis], members))


def test_cycle_methods_validate_the_chromas_they_are_given():
    """Raw values are parsed, not trusted."""
    fifths = Cycle(7)
    assert fifths.rank([1, 128, 4]).tolist() == [0, 1, 2]  # plain ints are keys: C, G, D
    assert fifths.dist(1, [128, 32]).tolist() == [1, 1]
    with pytest.raises(ValueError):
        fifths.rank(1 << MS.tones)
    with pytest.raises(ValueError):
        fifths.dist(1, -1)
    with pytest.raises(TypeError):
        fifths.rank(1.0)


# ── indexing by rank ────────────────────────────────────────────────────


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_cycle_indexing(step, start, name):
    """A rank gives one chroma, a range of ranks an array; both are unbounded."""
    c = Cycle(step, start_of(start, name))
    assert np.ndim(c[0]) == 0
    for r in range(-MS.tones, MS.tones):
        assert tones_of(c[r]) == {(t + start + r * step) % MS.tones for t in tones_of(BASES[name])}
        assert c.at(c.rank(c[r])) == c[r]

    assert [int(x) for x in c] == c[:].tolist() == members_of(step, start, int(BASES[name]))
    assert c[: 2 * c.period].tolist() == c[:].tolist() * 2
    assert c[:-25:-1].tolist() == [int(c[-r]) for r in range(25)]


@pytest.mark.parametrize(("step", "start", "name"), SAMPLE_BASED, ids=BASED_IDS)
def test_cycle_symmetric_indexing(step, start, name):
    """Entry d of `sym` holds everything d steps from the start."""
    c = Cycle(step, start_of(start, name))
    assert len(c.sym) == c.period // 2 + 1
    assert c.sym[:].shape == (len(c.sym), 2)
    assert [pair.tolist() for pair in c.sym] == c.sym[:].tolist()
    for d in range(len(c.sym)):
        assert c.sym[d].tolist() == [int(c[d]), int(c[-d])]
        assert set(c.sym[d].tolist()) == {int(x) for x in c if c.dist(c[0], x) == d}


def test_cycle_indexing_anchors():
    assert Cycle(7)[:4].tolist() == note_of([0, 7, 2, 9]).tolist()  # C G D A
    assert Cycle(7)[:-4:-1].tolist() == note_of([0, 5, 10, 3]).tolist()  # C F Bb Eb
    assert Cycle(3, note_of(9))[:5].tolist() == note_of([9, 0, 3, 6, 9]).tolist()  # wraps after four
    assert tones_of(Cycle(7, BASES["major"])[-1]) == {5, 9, 0}  # F major
    assert Cycle(3, BASES["dim7"])[:4].tolist() == [int(BASES["dim7"])] * 4  # never moves

    assert Cycle(7).sym[:3].tolist() == note_of([(0, 0), (7, 5), (2, 10)]).tolist()
    assert Cycle(4).sym[:].tolist() == note_of([(0, 0), (4, 8)]).tolist()  # odd period, no antipode
