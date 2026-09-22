import numpy as np
import pytest
from conftest import chroma_keys, notes
from hypothesis import given
from utils.fixtures import ALL_CHROMAS, ALL_KEYS

from chromath.constants import MS
from chromath.core import chroma


@pytest.mark.law
@pytest.mark.slow
def test_a_vector_converts_to_its_own_index():
    """ALL_CHROMAS[k] is the view of key k, so from_vector recovers k."""
    assert np.array_equal(chroma.from_vector(ALL_CHROMAS), ALL_KEYS)


@pytest.mark.law
def test_a_single_note_is_a_single_bit():
    for n in range(MS.tones):
        assert chroma.from_st(n) == 2**n


@pytest.mark.law
@pytest.mark.slow
def test_every_key_converts_to_its_own_vector():
    assert np.array_equal(chroma.to_vector(ALL_KEYS), ALL_CHROMAS)


@pytest.mark.law
def test_a_key_above_the_twelve_bit_range_is_rejected():
    """Keys are 12-bit: 4096 is one past the last valid chroma."""
    with pytest.raises(ValueError):
        chroma.validate_chroma(np.uint16(4096))


@pytest.mark.law
def test_the_largest_valid_key_is_accepted():
    assert chroma.cardinality(chroma.validate_chroma(np.uint16(4095))) == MS.tones


@pytest.mark.law
@given(chroma_keys())
def test_a_key_survives_a_round_trip_through_its_vector(c):
    assert chroma.from_vector(chroma.to_vector(c)) == c


@pytest.mark.law
@given(chroma_keys())
def test_semitones_are_the_set_bits(c):
    assert np.array_equal(chroma.to_st(c), np.flatnonzero(chroma.to_vector(c)))


@pytest.mark.law
@given(notes())
def test_a_note_converts_to_one_semitone(note):
    assert np.array_equal(np.flatnonzero(chroma.to_vector(note)), chroma.to_st(note))
