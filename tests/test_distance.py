import pytest
from conftest import chromas
from hypothesis import given
from utils.fixtures import keys

from chromath.core import chroma

METRICS = [chroma.hamming, chroma.jaccard, chroma.tversky_symm]
ALL_METRICS = [*METRICS, chroma.tversky]


@pytest.mark.law
@pytest.mark.parametrize("d", ALL_METRICS)
@given(chromas)
def test_a_chroma_is_at_no_distance_from_itself(d, c):
    assert d(c, c) == 0


@pytest.mark.law
@pytest.mark.parametrize("d", METRICS)
@given(chromas, chromas)
def test_distance_is_symmetric(d, a, b):
    """Tversky is excluded: it measures inclusion, deliberately asymmetric."""
    assert d(a, b) == d(b, a)


@pytest.mark.law
@pytest.mark.parametrize("d", ALL_METRICS)
@given(chromas, chromas)
def test_distance_is_in_the_unit_interval(d, a, b):
    assert 0.0 <= d(a, b) <= 1.0


@pytest.mark.law
@pytest.mark.parametrize("d", METRICS)
@given(chromas, chromas, chromas)
def test_distance_obeys_the_triangle_inequality(d, a, b, c):
    assert d(a, c) <= d(a, b) + d(b, c) + 1e-12


@pytest.mark.law
@pytest.mark.parametrize("d", ALL_METRICS)
@given(chromas, chromas)
def test_distance_is_invariant_under_transposition(d, a, b):
    """Transposing both chromas together cannot change how far apart they are."""
    for n in range(12):
        assert d(chroma.transpose(a, n), chroma.transpose(b, n)) == d(a, b)


@pytest.mark.law
@pytest.mark.parametrize("d", ALL_METRICS)
def test_two_empty_chromas_are_identical_not_undefined(d):
    """0/0 is defined as full overlap, so the distance is 0 rather than nan."""
    assert d(keys.edge_cases.empty, keys.edge_cases.empty) == 0
