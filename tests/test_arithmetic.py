import pytest
from conftest import chromas, shifts
from hypothesis import given
from utils.fixtures import ALL_KEYS, keys

from chromath.constants import MS
from chromath.core import chroma

chords = keys.chords


@pytest.mark.law
@given(chromas)
def test_invert_identity(c):
    assert chroma.invert(chroma.invert(c)) == c


@pytest.mark.law
@pytest.mark.slow
def test_invert_identity_global():
    """Inversion permutes the whole chroma space."""
    assert set(chroma.invert(ALL_KEYS).tolist()) == set(ALL_KEYS.tolist())


@pytest.mark.law
@given(chromas)
def test_transpose_loops(c):
    assert chroma.transpose(c, MS.tones) == c


@pytest.mark.law
@pytest.mark.slow
def test_transpose_loops_global():
    """Every transposition permutes the whole chroma space."""
    for n in range(MS.tones):
        assert set(chroma.transpose(ALL_KEYS, n).tolist()) == set(ALL_KEYS.tolist())


@pytest.mark.law
def test_transpose_examples():
    assert chroma.transpose(chords.C_maj, 5) == chords.F_maj
    assert chroma.transpose(chords.C_maj, -7) == chords.F_maj
    assert chroma.transpose(chords.C_min, 2) == chords.D_min
    assert chroma.transpose(chords.C_min, 2 + MS.tones) == chords.D_min


@pytest.mark.law
@pytest.mark.slow
def test_transpose_moves_every_pitch_class():
    """Oracle: the transposed set is the input set shifted, exhaustively."""
    for k in ALL_KEYS:
        src = set(chroma.to_st(k).tolist())
        for n in range(MS.tones):
            got = set(chroma.to_st(chroma.transpose(k, n)).tolist())
            assert got == {(i + n) % MS.tones for i in src}


@pytest.mark.law
@given(chromas, shifts, shifts)
def test_transposition_is_additive(c, a, b):
    assert chroma.transpose(chroma.transpose(c, a), b) == chroma.transpose(c, a + b)


@pytest.mark.law
@given(chromas, shifts)
def test_transposition_preserves_cardinality(c, n):
    assert chroma.cardinality(chroma.transpose(c, n)) == chroma.cardinality(c)


@pytest.mark.law
@given(chromas)
def test_inversion_preserves_cardinality(c):
    assert chroma.cardinality(chroma.invert(c)) == chroma.cardinality(c)
