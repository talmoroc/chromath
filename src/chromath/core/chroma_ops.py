import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import overload, Literal

from ..types import DT, ChromaArray, ScaleChromaArray
from ..constants import DefaultMusicSystem as MS
from ..core.validation import validate_chroma_array
from ..core.conversion import chroma_to_semitones

# GENERATION


def from_vector(v: ArrayLike) -> ChromaArray:
    v = np.array(v, dtype=DT.Chr)
    return validate_chroma_array(v)


def from_index(v: ArrayLike) -> ChromaArray:
    v = np.array(v)
    chroma = np.zeros(MS.tones, dtype=DT.Chr)
    chroma[v] = 1
    return validate_chroma_array(chroma)


def to_bits(v: ChromaArray) -> NDArray[np.uint8]:
    return np.packbits(v)


def from_bits(v: NDArray[np.uint8]) -> ChromaArray:
    return validate_chroma_array(np.unpackbits(v, count=12).astype(DT.Chr))


# CORE VECTOR FUNCTIONS


@overload
def invert(s: ScaleChromaArray, pivot: int = 0) -> ScaleChromaArray: ...


@overload
def invert(c: ChromaArray, pivot: int = 0) -> ChromaArray: ...


def invert(c: ChromaArray, pivot: int = 0) -> ChromaArray:
    """Musical inversion around a pivot (default 0)"""
    new_mask = np.zeros(MS.tones)
    indices = (pivot - chroma_to_semitones(c)) % MS.tones
    new_mask[indices] = 1
    return new_mask.astype(DT.Chr)


def tonalities_matrix(s: ScaleChromaArray) -> ScaleChromaArray:
    idx = np.arange(MS.tones)
    shifted_index = (idx - idx[:, np.newaxis]) % 12
    return s[shifted_index]


# UTILITIES


def isin(c: ChromaArray, container: ChromaArray) -> NDArray[DT.Chr]:
    if c.ndim > 1 and container.ndim > 1:
        c = c[..., np.newaxis, :]
    return np.all(container & c == c, axis=-1)


def common_tones(c1: ChromaArray, c2: ChromaArray) -> NDArray[DT.St]:
    if c1.ndim > 1 and c2.ndim > 1:
        c1 = c1[..., np.newaxis, :]
    return np.sum(c2 & c1, axis=-1, dtype=DT.St)


# Hamming distance : good for comparing similar objects. Difference in tones relative to the 12 tones
def hamming_dist(c1: ChromaArray, c2: ChromaArray) -> NDArray:
    """ Measures how many tones are different between two chromas. Relative to the total number of tones in the music system."""
    ct = common_tones(c1, c2)
    return (c1.sum(axis=-1) + c2.sum(axis=-1) - 2 * ct) / MS.tones


# Jaccard distance : good for comparing similar objects. Difference in tones within the space of c1 union c2
def jaccard_dist(c1: ChromaArray, c2: ChromaArray) -> NDArray:
    """ Measures how many tones are shared between two chromas. Relative to the total number of tones present in either chroma."""
    ct = common_tones(c1, c2)
    return 1 - ct / (c1.sum(axis=-1) + c2.sum(axis=-1) - ct)


# Assymetric Tversky : inclusion of an object within another one
def tversky_dist(from_: ChromaArray, to_: ChromaArray) -> NDArray:
    """ Measures how much 'from_' is included in 'to_'. """
    ct = common_tones(from_, to_)
    return 1 - ct / from_.sum(axis=-1)[..., np.newaxis]


def closest(from_: ChromaArray, to_: ChromaArray, n: int = 1, dist: Literal['hamming', 'jaccard', 'tversky'] = 'tversky') -> NDArray[DT.St]:
    """ Returns the indices of the closest chromas in 'to_' for each chroma in 'from_' based on the specified distance metric. """
    if from_.ndim > 1 and to_.ndim > 1:
        from_ = from_[..., np.newaxis, :]
    closest = np.argsort(-common_tones(from_, to_), stable=True, axis=-1).astype(DT.St)
    return closest[..., :n]
