"""Every chroma operation is elementwise.

An operation on an array of keys must agree, entry for entry, with the same
operation applied to each key on its own. That one law covers batching,
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


# ── The elementwise law ─────────────────────────────────────────────────


@pytest.mark.law
@pytest.mark.parametrize(("name", "f"), UNARY, ids=[n for n, _ in UNARY])
@given(chroma_key_arrays())
def test_a_unary_operation_is_elementwise(name, f, arr):
    """f(array)[i] == f(array[i])."""
    batched = f(arr)
    one_by_one = [f(np.uint16(k)) for k in arr]
    assert len(batched) == len(arr)
    for got, expected in zip(batched, one_by_one, strict=True):
        assert np.array_equal(got, expected)


@pytest.mark.law
@pytest.mark.parametrize(("name", "f"), BINARY, ids=[n for n, _ in BINARY])
@given(chroma_key_arrays(), chroma_key_arrays())
def test_a_binary_operation_is_elementwise(name, f, a, b):
    """f(x, y)[i] == f(x[i], y[i]), over the common length."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    batched = f(a, b)
    for i in range(n):
        assert np.array_equal(batched[i], f(np.uint16(a[i]), np.uint16(b[i])))


@pytest.mark.law
@pytest.mark.parametrize(("name", "f"), BINARY, ids=[n for n, _ in BINARY])
@given(chroma_key_arrays(), chromas)
def test_a_binary_operation_broadcasts_a_scalar(name, f, arr, scalar):
    """A lone key on either side broadcasts across the array."""
    right = f(arr, scalar)
    left = f(scalar, arr)
    for i, k in enumerate(arr):
        assert np.array_equal(right[i], f(np.uint16(k), np.uint16(scalar)))
        assert np.array_equal(left[i], f(np.uint16(scalar), np.uint16(k)))


@pytest.mark.law
@pytest.mark.parametrize(("name", "f"), BINARY, ids=[n for n, _ in BINARY])
@given(chroma_key_arrays(1, 5), chroma_key_arrays(1, 5))
def test_a_binary_operation_builds_a_pairwise_table(name, f, a, b):
    """a[:, None] against b gives every pair, shape (len(a), len(b))."""
    table = f(a[:, np.newaxis], b)
    assert table.shape == (len(a), len(b))
    for i in range(len(a)):
        for j in range(len(b)):
            assert np.array_equal(table[i, j], f(np.uint16(a[i]), np.uint16(b[j])))


# ── Shape and dtype ─────────────────────────────────────────────────────


@pytest.mark.law
@pytest.mark.parametrize(("name", "f"), UNARY, ids=[n for n, _ in UNARY])
@given(chroma_key_arrays(shape2d=True))
def test_a_unary_operation_preserves_leading_shape(name, f, arr):
    """A (r, 2) batch stays (r, 2), plus any axes the result adds."""
    out = np.asarray(f(arr))
    assert out.shape[: arr.ndim] == arr.shape


@pytest.mark.law
@pytest.mark.parametrize(("name", "f"), KEY_RETURNING, ids=[n for n, _ in KEY_RETURNING])
@given(chroma_key_arrays())
def test_an_operation_returning_keys_keeps_the_key_dtype(name, f, arr):
    """uint16 in, uint16 out — no silent promotion to int64."""
    assert np.asarray(f(arr)).dtype == np.uint16
    assert np.asarray(f(np.uint16(1))).dtype == np.uint16


@pytest.mark.law
@given(chroma_key_arrays())
def test_to_vector_adds_a_trailing_pitch_class_axis(arr):
    assert chroma.to_vector(arr).shape == (*arr.shape, MS.tones)


@pytest.mark.law
@given(chroma_key_arrays())
def test_from_vector_removes_it(arr):
    """Round trip over a whole batch at once."""
    assert np.array_equal(chroma.from_vector(chroma.to_vector(arr)), arr)


@pytest.mark.law
@given(chroma_key_arrays(shape2d=True))
def test_the_round_trip_survives_two_leading_axes(arr):
    assert np.array_equal(chroma.from_vector(chroma.to_vector(arr)), arr)


# ── The one operation that is not elementwise ───────────────────────────


@pytest.mark.law
def test_to_st_rejects_a_batch():
    """Semitone lists are ragged, so to_st takes one chroma and says so.

    Silently flattening a batch is how the old array-based invert produced a
    plausible wrong answer instead of an error.
    """
    with pytest.raises(ValueError):
        chroma.to_st(np.array([chroma.from_st(0, 4, 7), chroma.from_st(0, 3, 7)]))
