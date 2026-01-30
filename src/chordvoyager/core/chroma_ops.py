import numpy as np

from src.chordvoyager.types import (
    NDArrayBool,
    NDArrayInt8,
    ChromaVec,
)
from src.chordvoyager.constants import DefaultMusicSystem as MS
from src.chordvoyager.core.validation import validate_chroma
from src.chordvoyager.core.conversion import chroma_to_degree


def generate(v: NDArrayBool | NDArrayInt8 | int) -> ChromaVec:
    if isinstance(v, int):
        if v >= MS.max_int_repr:
            raise ValueError(
                f"Bitwise representation must be less than {MS.max_int_repr}, got {v}"
            )
        vector = np.array([(v >> i) & 1 for i in range(MS.tones)])
        return validate_chroma(vector)
    elif isinstance(v, np.ndarray) and v.dtype == np.int8:
        chroma = np.zeros(MS.tones)
        chroma[v] = 1
        return validate_chroma(v)
    elif isinstance(v, np.ndarray) and v.dtype == np.bool_:
        return validate_chroma(v)
    else:
        raise TypeError(f"Unsupported input type: {type(v)}")

def to_int(v: ChromaVec) -> int:
    return int(np.dot(v, MS.powers))

def shift(v: ChromaVec, n: int) -> ChromaVec:
    return np.roll(v, n) # TODO: check if bitwise op is faster


def invert(v, pivot: int = 0) -> ChromaVec:
    """Musical inversion around a pivot (default 0)"""
    new_mask = np.zeros(MS.tones)
    indices = (pivot - chroma_to_degree(v)) % MS.tones
    new_mask[indices] = 1
    return new_mask


# UTILITIES

def isin(v: ChromaVec, e: ChromaVec) -> bool:
    return np.sum(v) == np.sum(v * e)  # TODO
# TODO: check if this works depending on dimensions
# TODO: check if this works for a matrix product



