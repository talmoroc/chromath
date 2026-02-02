import numpy as np

from ..types import DT, NDArray, Chroma, ScaleChroma, Degrees
from ..constants import DefaultMusicSystem as MS
from ..core.validation import validate_chroma
from ..core.conversion import chroma_to_degree


# GENERATION


def from_vector(v: NDArray) -> Chroma:
    return validate_chroma(v)


def from_index(v: NDArray) -> Chroma:
    chroma = np.zeros(MS.tones)
    chroma[v] = 1
    return validate_chroma(v)


def from_int(bitwise_repr: int) -> Chroma:
    if bitwise_repr >= MS.max_int_repr:
        raise ValueError(
            f"Bitwise representation must be less than {MS.max_int_repr}, got {bitwise_repr}"
        )
    vector = np.array([(bitwise_repr >> i) & 1 for i in range(MS.tones)])
    return validate_chroma(vector)


# CORE VECTOR FUNCTIONS


def to_int(v: Chroma) -> int:
    return int(np.dot(v, MS.powers))


def invert(v, pivot: int = 0) -> Chroma:
    """Musical inversion around a pivot (default 0)"""
    new_mask = np.zeros(MS.tones)
    indices = (pivot - chroma_to_degree(v)) % MS.tones
    new_mask[indices] = 1
    return new_mask.astype(DT.Chr)

# def generate_scale_chroma_matrix(s: ScaleChroma) -> ScaleChromaMatrix:
    


# UTILITIES


def isin(v: Chroma, e: Chroma) -> bool:
    return np.sum(v) == np.sum(v * e)  # TODO


# TODO: check if this works depending on dimensions
# TODO: check if this works for a matrix product


###### TODO: Scales and operations on scales
