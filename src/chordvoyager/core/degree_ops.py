import numpy as np

from ..types import IntervalArray
from ..constants import DefaultMusicSystem as MS


def to_int(v: IntervalArray) -> int:
    return int(np.dot(v, MS.powers))  # TODO


def from_bits(bitwise_repr: int) -> IntervalArray:
    if bitwise_repr >= MS.max_int_repr:
        raise ValueError(f"Bits should be < 2**{MS.tones}, got {bitwise_repr}")
    _temp_mask = np.array(((bitwise_repr >> i) & 1 for i in range(MS.tones)))
    return _temp_mask  # TODO


def isin(v: IntervalArray, e: IntervalArray) -> bool:
    return np.sum(v) == np.sum(v * e)  # TODO


def shift(v: IntervalArray, n: int) -> IntervalArray:
    return v  # TODO
