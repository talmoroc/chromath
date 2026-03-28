import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import overload

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


# TODO: inversion for degrees ?


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


def dist(c1: ChromaArray, c2: ChromaArray) -> NDArray:
    ct = common_tones(c1, c2)
    return (ct.max() - ct) / ct.max() # TODO: improve to include min(1s in c1, 1s in c2)


def closest(c: ChromaArray, container: ChromaArray, n: int = 1) -> NDArray[DT.St]:
    if c.ndim > 1 and container.ndim > 1:
        c = c[..., np.newaxis, :]
    closest = np.argsort(-common_tones(c, container), stable=True, axis=-1).astype(DT.St)
    return closest[:, :n]


# Full matrices
