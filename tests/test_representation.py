import numpy as np
import pytest
from conftest import notes, shifts
from hypothesis import given
from utils.fixtures import fixtures, keys

from chromath.constants import MS
from chromath.core import chroma


@pytest.mark.law
def test_the_empty_chroma_is_representable():
    """Zero notes is the key 0."""
    assert chroma.from_st([]) == keys.edge_cases.empty == 0


@pytest.mark.law
def test_chroma_zero_is_c():
    """Pitch class 0 is C, so it occupies bit 0."""
    assert chroma.from_st(0) == keys.edge_cases.single_C == 1


@pytest.mark.law
def test_index_zero_is_c():
    """The twelve-lane view puts C first."""
    assert np.array_equal(
        fixtures.chords.C_maj,
        np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.bool_),
    )


@pytest.mark.law
def test_chroma_is_c_major():
    """C major is C, E, G -> bits 0, 4, 7."""
    assert chroma.from_st(0, 4, 7) == keys.chords.C_maj == 0b000010010001


@pytest.mark.law
@given(notes())
def test_a_note_has_exactly_one_pitch_class(c):
    assert chroma.cardinality(c) == 1


@pytest.mark.law
@given(shifts)
def test_chroma_is_modulo_12(n):
    assert chroma.from_st(n) == chroma.from_st(n % MS.tones)
