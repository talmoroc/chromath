import numpy as np
from itertools import product as cartesian_product

from ..types import DT, IntervalArray, ScaleIntervalArray, InterpretedIntervalArray, ScaleLookupArray, ScaleLookupCounts
from ..constants import DefaultMusicSystem as MS
from . import validation as val

def to_int(v: IntervalArray) -> int:
    """
    Converts an IntervalArray to its bitwise integer representation based on semitones.
    """
    semitones = np.atleast_2d(v)[..., 1]
    chroma = np.zeros(MS.tones, dtype=int)
    chroma[semitones % MS.tones] = 1
    return int(np.dot(chroma, MS.powers))

def from_bits(bitwise_repr: int) -> IntervalArray:
    """
    Reconstructs an IntervalArray from a bitwise integer representation.
    Note: Degrees are set to 0 as they cannot be inferred from bits alone.
    """
    if bitwise_repr >= MS.max_int_repr:
        raise ValueError(f"Bits should be < 2**{MS.tones}, got {bitwise_repr}")

    semitones = np.where((bitwise_repr >> np.arange(MS.tones)) & 1)[0]
    res = np.zeros((len(semitones), 2), dtype=DT.St)
    res[:, 1] = semitones
    return res


def isin(v: IntervalArray, e: IntervalArray) -> bool:
    """
    Checks if a specific interval [degree, semitone] exists within a collection of intervals.
    """
    return bool(np.any(np.all(e == v, axis=-1)))


def shift(v: IntervalArray, n: int) -> IntervalArray:
    """
    Transposes the interval(s) by n semitones.
    """
    res = np.array(v, copy=True)
    res[..., 1] = (res[..., 1] + n) % MS.tones
    return res


def from_scale(scale_semitones: list[int] | np.ndarray) -> ScaleIntervalArray:
    """
    Converts a scale (list of semitones) to a ScaleIntervalArray.
    Each interval is represented as (degree, semitone).
    Degrees are 1-indexed (1 through 7 for heptatonic scales).
    """
    if len(scale_semitones) != MS.degrees:
        raise ValueError(f"Scale must have {MS.degrees} degrees, got {len(scale_semitones)}")

    degrees = np.arange(1, MS.degrees + 1, dtype=DT.St)
    semitones = np.array(scale_semitones, dtype=DT.St)
    res = np.column_stack((degrees, semitones))
    return val.validate_scale_interval_array(res)


def from_chord(chord_semitones: list[int] | np.ndarray) -> IntervalArray:
    """
    Converts a chord (list of semitones relative to root) to an IntervalArray.
    Degrees are inferred from sorted position in the chord.
    """
    semitones = np.array(chord_semitones, dtype=DT.St)
    semitones = np.sort(semitones)
    degrees = np.arange(len(semitones), dtype=DT.St)
    res = np.column_stack((degrees, semitones))
    return val.validate_interval_array(res)


def semitone_distance(interval1: IntervalArray, interval2: IntervalArray) -> int:
    """
    Computes the semitone distance between two intervals.
    Returns the absolute difference in semitones.
    """
    st1 = np.atleast_1d(interval1)[..., 1]
    st2 = np.atleast_1d(interval2)[..., 1]
    return int(np.abs(st2 - st1) % MS.tones)


def scale_distance(scale1_semitones: list[int] | np.ndarray,
                   scale2_semitones: list[int] | np.ndarray) -> int:
    """
    Computes the total semitone distance between two scales.
    Returns the sum of absolute differences for each degree.
    """
    scale1 = np.array(scale1_semitones, dtype=DT.St)
    scale2 = np.array(scale2_semitones, dtype=DT.St)

    if len(scale1) != len(scale2):
        raise ValueError(f"Scales must have same length, got {len(scale1)} and {len(scale2)}")

    return int(np.sum(np.abs((scale2 - scale1) % MS.tones)))


def common_intervals(interval_array1: IntervalArray, interval_array2: IntervalArray) -> IntervalArray:
    """
    Finds intervals that appear in both arrays (by semitone value).
    """
    st1 = interval_array1[..., 1]
    st2 = interval_array2[..., 1]

    common_st = np.intersect1d(st1, st2)
    degrees = np.arange(len(common_st), dtype=DT.St)
    res = np.column_stack((degrees, common_st))
    return val.validate_interval_array(res)


def invert(v: IntervalArray, pivot: int = 0) -> IntervalArray:
    """
    Musical inversion of intervals around a pivot (default 0 semitones).
    """
    res = np.array(v, copy=True)
    res[..., 1] = (pivot - res[..., 1]) % MS.tones
    return res

