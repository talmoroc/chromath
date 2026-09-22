"""Every chroma operation is elementwise.

An operation on an array of keys must agree, entry for entry, with the same
operation applied to each key on its own. That one rule covers batching,
broadcasting and shape preservation together.
"""

import numpy as np
import pytest
from conftest import chroma_key_arrays, chromas, tones_of
from hypothesis import given

from chromath.constants import MS
from chromath.core import chroma
from chromath.core.cycle import Cycle

#: (name, f) where f takes one key or an array of keys.
UNARY = [
    ("transpose", lambda c: chroma.transpose(c, 5)),
    ("transpose_zero", lambda c: chroma.transpose(c, 0)),
    ("invert", lambda c: chroma.invert(c, 3)),
    ("cardinality", chroma.cardinality),
    ("to_vector", chroma.to_vector),
]

#: (name, f) where f takes two keys or two arrays of keys.
BINARY = [
    ("common_tones", chroma.common_tones),
    ("isin", chroma.isin),
    ("hamming", chroma.hamming),
    ("jaccard", chroma.jaccard),
    ("tversky", chroma.tversky),
]

KEY_RETURNING = [("transpose", lambda c: chroma.transpose(c, 5)), ("invert", chroma.invert)]


@pytest.mark.parametrize(("name", "f"), UNARY, ids=[n for n, _ in UNARY])
@given(chroma_key_arrays(), chroma_key_arrays(shape2d=True))
def test_a_unary_operation_is_elementwise(name, f, arr, grid):
    """f(array)[i] == f(array[i]), and the leading shape survives."""
    batched = f(arr)
    assert len(batched) == len(arr)
    for got, key in zip(batched, arr, strict=True):
        assert np.array_equal(got, f(np.uint16(key)))
    assert np.asarray(f(grid)).shape[: grid.ndim] == grid.shape


@pytest.mark.parametrize(("name", "f"), BINARY, ids=[n for n, _ in BINARY])
@given(chroma_key_arrays(1, 5), chroma_key_arrays(1, 5), chromas)
def test_a_binary_operation_is_elementwise(name, f, a, b, scalar):
    """Pairwise over equal lengths, broadcast against a scalar, table when nested."""
    n = min(len(a), len(b))
    for i in range(n):
        assert np.array_equal(f(a[:n], b[:n])[i], f(np.uint16(a[i]), np.uint16(b[i])))

    right, left = f(a, scalar), f(scalar, a)
    for i, key in enumerate(a):
        assert np.array_equal(right[i], f(np.uint16(key), np.uint16(scalar)))
        assert np.array_equal(left[i], f(np.uint16(scalar), np.uint16(key)))

    table = f(a[:, np.newaxis], b)
    assert table.shape == (len(a), len(b))
    for i in range(len(a)):
        for j in range(len(b)):
            assert np.array_equal(table[i, j], f(np.uint16(a[i]), np.uint16(b[j])))


@pytest.mark.parametrize(("name", "f"), KEY_RETURNING, ids=[n for n, _ in KEY_RETURNING])
@given(chroma_key_arrays())
def test_an_operation_returning_keys_keeps_the_key_dtype(name, f, arr):
    """uint16 in, uint16 out — no silent promotion to int64."""
    assert np.asarray(f(arr)).dtype == np.uint16
    assert np.asarray(f(np.uint16(1))).dtype == np.uint16


@given(chroma_key_arrays(), chroma_key_arrays(shape2d=True))
def test_conversion_round_trips_over_a_batch(arr, grid):
    """to_vector adds a trailing pitch-class axis and from_vector removes it."""
    assert chroma.to_vector(arr).shape == (*arr.shape, MS.tones)
    assert np.array_equal(chroma.from_vector(chroma.to_vector(arr)), arr)
    assert np.array_equal(chroma.from_vector(chroma.to_vector(grid)), grid)


def test_to_st_rejects_a_batch():
    """Semitone lists are ragged, so to_st takes one chroma and says so."""
    with pytest.raises(ValueError):
        chroma.to_st(np.array([chroma.from_st(0, 4, 7), chroma.from_st(0, 3, 7)]))


# ── cycle methods ───────────────────────────────────────────────────────

#: Cycles with a period of at least four, so a batch can be reshaped.
CYCLES = [Cycle(7), Cycle(3, 9), Cycle(1, 11)]
CYCLE_IDS = [f"step{c.step}_from{c.start}" for c in CYCLES]


EMPTY = np.array([], dtype=np.int8)


@pytest.mark.parametrize("c", CYCLES, ids=CYCLE_IDS)
@pytest.mark.parametrize("direction", ["forward", "backward", "min"])
def test_cycle_rank_is_elementwise(c, direction):
    """rank(array)[i] == rank(array[i]), whatever shape the array is."""
    tones = np.array(sorted(tones_of(c.mask)))
    one_by_one = [int(c.rank(int(t), direction)) for t in tones]

    batched = np.asarray(c.rank(tones, direction))
    assert batched.shape == tones.shape
    assert batched.tolist() == one_by_one

    grid = tones[:4].reshape(2, 2)
    assert np.asarray(c.rank(grid, direction)).tolist() == [[int(c.rank(int(t), direction)) for t in row] for row in grid]

    cube = np.stack([grid, grid])
    assert np.asarray(c.rank(cube, direction)).shape == cube.shape


@pytest.mark.parametrize("c", CYCLES, ids=CYCLE_IDS)
def test_cycle_rank_keeps_scalars_scalar(c):
    """A lone semitone is not silently promoted to a one-element array."""
    tone = sorted(tones_of(c.mask))[0]
    assert np.ndim(c.rank(tone)) == 0
    assert np.asarray(c.rank(EMPTY)).shape == (0,)


@pytest.mark.parametrize("c", CYCLES, ids=CYCLE_IDS)
def test_cycle_dist_is_elementwise(c):
    """Pairwise over matching shapes, and still symmetric entry by entry."""
    tones = np.array(sorted(tones_of(c.mask)))
    a, b = tones[:4], tones[-4:]

    paired = np.asarray(c.dist(a, b))
    assert paired.shape == a.shape
    assert paired.tolist() == [int(c.dist(int(x), int(y))) for x, y in zip(a, b, strict=True)]
    assert np.array_equal(paired, np.asarray(c.dist(b, a)))

    assert np.asarray(c.dist(a.reshape(2, 2), b.reshape(2, 2))).tolist() == paired.reshape(2, 2).tolist()
    assert np.asarray(c.dist(EMPTY, EMPTY)).shape == (0,)


@pytest.mark.parametrize("c", CYCLES, ids=CYCLE_IDS)
def test_cycle_dist_broadcasts(c):
    """A lone semitone against an array, and a column against a row for every pair."""
    tones = np.array(sorted(tones_of(c.mask)))
    a, b = tones[:4], tones[-4:]
    lone = int(tones[0])

    assert np.asarray(c.dist(lone, b)).tolist() == [int(c.dist(lone, int(y))) for y in b]
    assert np.asarray(c.dist(a, lone)).tolist() == [int(c.dist(int(x), lone)) for x in a]
    assert np.ndim(c.dist(lone, lone)) == 0

    table = np.asarray(c.dist(a[:, np.newaxis], b))
    assert table.shape == (len(a), len(b))
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            assert table[i, j] == c.dist(int(x), int(y))

    assert np.asarray(c.dist(tones[:3, np.newaxis], tones[:2])).shape == (3, 2)


@pytest.mark.parametrize("c", CYCLES, ids=CYCLE_IDS)
def test_batched_ranks_and_distances_obey_the_scalar_bounds(c):
    """The laws hold over a whole table at once, not only one pair at a time."""
    tones = np.array(sorted(tones_of(c.mask)))
    table = np.asarray(c.dist(tones[:, np.newaxis], tones))
    assert np.array_equal(table, table.T)
    assert np.array_equal(np.diagonal(table), np.zeros(len(tones)))
    assert table.min() == 0
    assert table.max() <= c.period // 2

    forward = np.asarray(c.rank(tones, "forward"))
    backward = np.asarray(c.rank(tones, "backward"))
    assert ((forward >= 0) & (forward < c.period)).all()
    assert ((backward > -c.period) & (backward <= 0)).all()
    assert sorted(forward.tolist()) == list(range(c.period))


def test_a_batch_holding_a_tone_outside_the_cycle_raises():
    """One bad entry fails the call rather than returning a sentinel."""
    dim = Cycle(3)  # {0, 3, 6, 9}
    inside, outside = 0, 1

    with pytest.raises(IndexError):
        dim.rank(outside)
    with pytest.raises(IndexError):
        dim.rank(np.array([inside, outside]))
    with pytest.raises(IndexError):
        dim.rank(np.array([[inside, inside], [inside, outside]]))

    with pytest.raises(IndexError):
        dim.dist(inside, outside)
    with pytest.raises(IndexError):
        dim.dist(outside, inside)
    with pytest.raises(IndexError):
        dim.dist(np.array([inside, inside]), np.array([3, outside]))
    with pytest.raises(IndexError):
        dim.dist(np.array([inside, outside]), np.array([3, 3]))
    with pytest.raises(IndexError):
        dim.dist(np.array([inside, outside])[:, np.newaxis], np.array([3, 6]))
