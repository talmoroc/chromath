"""Tests for batch operations"""

import numpy as np
import pytest
from conftest import chroma_key_arrays, chroma_of, chromas, each, foreign_to, note_of, turn
from hypothesis import given

from chromath.constants import MS
from chromath.core import chroma, cycle
from chromath.types import DT

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
    for got, key in zip(batched, each(arr), strict=True):
        assert np.array_equal(got, f(key))
    assert np.asarray(f(grid)).shape[: grid.ndim] == grid.shape


@pytest.mark.parametrize(("name", "f"), BINARY, ids=[n for n, _ in BINARY])
@given(chroma_key_arrays(1, 5), chroma_key_arrays(1, 5), chromas)
def test_a_binary_operation_is_elementwise(name, f, a, b, scalar):
    """Pairwise over equal lengths, broadcast against a scalar, table when nested."""
    n = min(len(a), len(b))
    for i in range(n):
        assert np.array_equal(f(a[:n], b[:n])[i], f(a[i, ...], b[i, ...]))

    right, left = f(a, scalar), f(scalar, a)
    for i, key in enumerate(each(a)):
        assert np.array_equal(right[i], f(key, scalar))
        assert np.array_equal(left[i], f(scalar, key))

    table = f(a[:, np.newaxis], b)
    assert table.shape == (len(a), len(b))
    for i in range(len(a)):
        for j in range(len(b)):
            assert np.array_equal(table[i, j], f(a[i, ...], b[j, ...]))


@pytest.mark.parametrize(("name", "f"), KEY_RETURNING, ids=[n for n, _ in KEY_RETURNING])
@given(chroma_key_arrays())
def test_an_operation_returning_keys_keeps_the_key_dtype(name, f, arr):
    """uint16 in, uint16 out — no silent promotion to int64."""
    assert f(arr).dtype == np.uint16
    assert f(np.array(1, dtype=DT.Key)).dtype == np.uint16


# ── one chroma is a 0-d array, never a NumPy scalar ─────────────────────

ONE = [*UNARY, ("from_vector", lambda c: chroma.from_vector(chroma.to_vector(c)))]


@pytest.mark.parametrize(("name", "f"), ONE, ids=[n for n, _ in ONE])
@given(chromas)
def test_a_unary_operation_on_one_chroma_returns_an_array(name, f, c):
    """NumPy demotes 0-d results to scalars; the library must not."""
    assert isinstance(c, np.ndarray) and c.ndim == 0
    assert isinstance(f(c), np.ndarray)


@pytest.mark.parametrize(("name", "f"), BINARY, ids=[n for n, _ in BINARY])
@given(chromas, chromas)
def test_a_binary_operation_on_two_chromas_returns_a_0d_array(name, f, a, b):
    got = f(a, b)
    assert isinstance(got, np.ndarray) and got.ndim == 0


def test_parsing_and_generation_return_arrays():
    for got in (chroma.validate_chroma_keys(5), chroma_of([0, 4, 7]), note_of(3), chroma_of([])):
        assert isinstance(got, np.ndarray) and got.ndim == 0 and got.dtype == DT.Key


@given(chroma_key_arrays(), chroma_key_arrays(shape2d=True))
def test_conversion_round_trips_over_a_batch(arr, grid):
    """to_vector adds a trailing pitch-class axis and from_vector removes it."""
    assert chroma.to_vector(arr).shape == (*arr.shape, MS.tones)
    assert np.array_equal(chroma.from_vector(chroma.to_vector(arr)), arr)
    assert np.array_equal(chroma.from_vector(chroma.to_vector(grid)), grid)


# ── cycle functions ─────────────────────────────────────────────────────

#: Cycles with a period of at least four, so a batch can be reshaped.
CYCLES = [(7, 0, "unison"), (3, 9, "unison"), (7, 0, "major"), (3, 9, "dom7")]
CYCLE_IDS = ["step7", "step3_from9", "step7_major", "step3_from9_dom7"]


@pytest.fixture(params=CYCLES, ids=CYCLE_IDS)
def members(request):
    """One turn of a cycle."""
    return turn(*request.param)


@pytest.mark.parametrize("direction", ["forward", "backward", "min"])
def test_cycle_rank_is_elementwise(members, direction):
    """rank(array)[i] == rank(array[i]), whatever shape the array is."""
    grid = members[:4].reshape(2, 2)
    assert cycle.rank(members, members, direction).tolist() == [int(cycle.rank(members, x, direction)) for x in each(members)]
    assert cycle.rank(members, grid, direction).tolist() == [[int(cycle.rank(members, x, direction)) for x in each(row)] for row in grid]
    assert cycle.rank(members, members[:0], direction).shape == (0,)


def test_cycle_dist_is_elementwise_and_broadcasts(members):
    """Pairwise, lone against array, column against row."""
    a, b = members[:4], members[-4:]
    lone = members[0, ...]

    assert cycle.dist(members, a, b).tolist() == [int(cycle.dist(members, x, y)) for x, y in zip(each(a), each(b), strict=True)]
    assert cycle.dist(members, lone, b).tolist() == [int(cycle.dist(members, lone, y)) for y in each(b)]

    table = cycle.dist(members, a[:, np.newaxis], b)
    assert table.shape == (len(a), len(b))
    assert all(table[i, j] == cycle.dist(members, x, y) for i, x in enumerate(each(a)) for j, y in enumerate(each(b)))


def test_cycle_functions_on_one_chroma_return_a_0d_array(members):
    """One chroma in, one 0-d array out."""
    one = members[0, ...]
    for got in (cycle.mask(members), cycle.rank(members, one), cycle.dist(members, one, one)):
        assert isinstance(got, np.ndarray) and got.ndim == 0


def test_a_batch_holding_a_chroma_outside_the_cycle_raises(members):
    """One bad entry fails the call rather than returning a sentinel."""
    inside, outside = members[0, ...], foreign_to(members[0, ...])
    mixed = np.array([inside, outside], dtype=DT.Key)

    with pytest.raises(IndexError):
        cycle.rank(members, mixed)
    with pytest.raises(IndexError):
        cycle.dist(members, members[:2], mixed)
    with pytest.raises(IndexError):
        cycle.dist(members, mixed[:, np.newaxis], members[:2])
