import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import overload

from ..types import DT, ChromaArray, ScaleChromaArray
from ..constants import DefaultMusicSystem as MS
from ..core.validation import validate_chroma_array
from ..core.conversion import chroma_to_degree

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
    indices = (pivot - chroma_to_degree(c)) % MS.tones
    new_mask[indices] = 1
    return new_mask.astype(DT.Chr)


# TODO: inversion for degrees ?


def tonalities_matrix(s: ScaleChromaArray) -> ScaleChromaArray:
    all_tonalities = np.array([np.roll(s, -i) for i in range(MS.tones)], dtype=DT.Chr)
    return all_tonalities


# UTILITIES


def isin(c: ChromaArray, container: ChromaArray) -> NDArray[DT.Chr]:
    return np.all(container & c == c, axis=container.ndim - 1)


def common_tones(c: ChromaArray, with_: ChromaArray) -> NDArray[DT.St]:
    return np.sum(with_ & c, axis=with_.ndim - 1, dtype=DT.St)


def dist(c: ChromaArray, with_: ChromaArray) -> NDArray:
    ct = common_tones(c, with_)
    return ct / ct.max()


def closest(c: ChromaArray, with_: ChromaArray, n: int = 1) -> NDArray[DT.St]:
    closest = np.argsort(-common_tones(c, with_), stable=True).astype(DT.St)
    return closest[:n]


# Full matrices
