"""Every chroma operation is elementwise.

An operation on an array of keys must agree, entry for entry, with the same
operation applied to each key on its own. That one rule covers batching,
broadcasting and shape preservation together.
"""

import numpy as np
import pytest
from conftest import chroma_key_arrays, chromas
from hypothesis import given

from chromath.constants import MS
from chromath.core import chroma

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
