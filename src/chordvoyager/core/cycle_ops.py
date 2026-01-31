import numpy as np
import math

from ..types import (
    DTYPE,
    NDArray,
    NDArrayInt8,
    ChromaVec,
    CycleMatrix,
    ToneToPositionMatrix,
    PositionToToneMatrix,
    # CycleDirection,
)

from ..constants import DefaultMusicSystem as MS


def generate(step: int) -> tuple[PositionToToneMatrix, ToneToPositionMatrix, ChromaVec]:
    """
    Generator for a Cycle. The Cycle can be viewed as a cyclic mapping
    from integers (positive or negative) to tones (0-MusicSystem.tones).

    - It is also useful to have the reverse : which are the 12 tones positions in the cycle

    - Since we want vectors and the Cycle can go both ways, the output are matrices
    with 2 lines : the first one for the cycle increasing the second one for the cycle
    in negative direction.

    - Given that most cycles are partial in 12-tone temperament the reverse mapping
    is often partial, so in these cases a mask (ChromaVec) is returned to tell which
    tones belong to the cycle.

    Args:
        step (int): the step of the Cycle, in semitones. Must be between 1 and the number of tones in the current Music System.

    Returns:
        tuple[CycleMatrix, CycleMatrix, ChromaVec | None]:
            1) CycleMatrix Position in the Cycle of the given tone : index = tone, value = its rank in the cycle
            2) CycleMatrix Tone of the given position in the Cycle : index = position, value = tone
            3) Chromavec 1 = a tone is in the cycle, 0 = it is not. If the cycle is complete, it's full of 1.
                Useful to check before using the first CycleMatrix (Position in the cycle of a given tone)
    """
    if not 1 <= step < MS.tones:
        raise ValueError(f"Step must be between 1 and {MS.tones}: got {step}")
    pi = periodicity(step)
    cycle_rank_st = np.arange(0, step * pi, step, dtype=DTYPE.Cycle) % MS.tones
    cycle_st_rank = np.zeros(MS.tones)
    cycle_st_rank[cycle_rank_st] = np.arange(pi)
    tone_to_pos_mat = np.array([cycle_st_rank, reverse_ndarray(cycle_st_rank)], dtype=DTYPE.Cycle)
    pos_to_tone_mat = np.array(
        [cycle_rank_st, reverse_ndarray(cycle_rank_st)], dtype=DTYPE.Cycle
    )  # TODO: does this work if cycle(0) != 0 ?
    # TODO: refactor in smaller functions to debug it easier
    if not is_complete(pi):
        mask = np.zeros(MS.tones)
        mask[cycle_rank_st] = 1
    else:
        mask = np.full(MS.tones, 1, dtype=DTYPE.Chr)
    return tone_to_pos_mat, pos_to_tone_mat, mask


def periodicity(step: int) -> int:
    return MS.tones // math.gcd(step, MS.tones)


def is_complete(periodicity: int) -> bool:
    return periodicity == MS.tones


# TODO: Clarify why two outputs are necessary. Also, mask = None or


# CORE VECTOR OPERATIONS


def reverse_ndarray(arr: NDArray) -> NDArray:
    pivot = np.array(arr[0])
    arr_len = arr.shape[0]
    return np.array([pivot, arr[arr_len:0, -1]], dtype=arr.dtype)


# Should the formula for these two be different between the two CycleMatrices types ?
def shift(c: CycleMatrix, n: int) -> CycleMatrix:
    return c  # TODO recursive rotation, nice formula


def centered_on(c: CycleMatrix, tone: int) -> CycleMatrix:
    return c  # TODO the matrix centered around a given tone.


# UTILITIES


def get_closest(tone: int, n: int, c: CycleMatrix) -> NDArrayInt8:
    return c  # TODO.


# I need : to check where the tone is in the cycle so Tone To Position
# Then I need to get the slice of size n around this position so Position to Tone
# Or maybe I can vectorize and just do a translation of Position to Tone with the tone information
# I just need to check how it behaves with the negative part of the cycle
# But if this is the case the function is just a wrapper for rotate + slice
# Need to find also how to handle when n >= cycle periodicity while staying in vector land
# Basically this would be rotate(c, tone)[:,:n+1]


def dist(t1: int, t2: int, c: ToneToPositionMatrix) -> NDArray:
    return np.array(t1)  # TODO shape = (1,2)
