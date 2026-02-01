import numpy as np
import math

from ..types import DT, NDArray, ChromaVec, CycleVec, CycleMatrix, RankMatrix

from ..constants import DefaultMusicSystem as MS


def periodicity(step: int) -> int:
    return MS.tones // math.gcd(step, MS.tones)


def is_complete(periodicity: int) -> bool:
    return periodicity == MS.tones


def _generate_cycle_tones(step: int) -> CycleVec:
    pi = periodicity(step)
    return np.arange(0, step * pi, step, dtype=DT.Cycle) % MS.tones


def _generate_cycle_vector(step: int) -> tuple[CycleVec, ChromaVec]:
    """
    Generator for a Cycle.

    A Cycle can be viewed as a cyclic mapping
    from integers (positive or negative) to tones (0-MusicSystem.tones).

    - It is also useful to have the reverse : which are the 12 tones positions in the cycle.
    Identical for the circle of fifth.

    - Given that most cycles are partial in 12-tone temperament the reverse mapping
    is often partial, so in these cases a mask (ChromaVec) is returned to tell which
    tones belong to the cycle.

    Args:
        step (int): the step of the Cycle, in semitones. Must be between 1 and the number of tones in the current Music System.

    Returns:
        tuple[CycleVec, ChromaVec]:
            1. cycle_vec : index = tone, value = its rank in the cycle, len = tones
            2. mask : 1 = a tone is in the cycle, 0 = it is not. If the cycle is complete, it's full of 1.
                Useful to check before using the first CycleVec (Position in the cycle of a given tone)
    """
    if not 1 <= step < MS.tones:
        raise ValueError(f"Step must be between 1 and {MS.tones}: got {step}")
    cycle_tones = _generate_cycle_tones(step)
    cycle_vec = np.full(MS.tones, -1)
    cycle_vec[cycle_tones] = np.arange(periodicity(step))
    mask = np.where(cycle_vec == -1, 0, 1).astype(DT.Chr)
    return cycle_vec, mask

# Compute the cycle rank of the tones relative to each tone
def _matrix_over_tones(c: CycleVec):
    return np.array([(np.roll(c, i)) for i in range(MS.tones)], dtype=DT.Cycle)


def generate_cycle_matrix(step: int) -> CycleMatrix:
    positive_vector, mask = _generate_cycle_vector(step)
    positive_matrix = _matrix_over_tones(positive_vector)
    negative_matrix = positive_matrix.transpose()  # possible because square matrix
    return np.array([positive_matrix, negative_matrix], dtype=DT.Cycle)

# Compute the cycle semitones series starting on each tone
def _matrix_over_ranks(c: CycleVec):
    return np.array([(c + i) % MS.tones for i in range(MS.tones)], dtype=DT.Cycle)


def generate_rank_matrix(step: int) -> RankMatrix:
    pos_rank = _generate_cycle_tones(step)
    neg_rank = _generate_cycle_tones(-step)
    return np.array([_matrix_over_ranks(pos_rank), _matrix_over_ranks(neg_rank)], dtype=DT.Cycle)


# UTILITIES
def get_tone_from_rank(r: RankMatrix, rank: int, relative_to: int = 0) -> int:
    return r[rank < 0, relative_to, rank]


def get_rank_from_tone(
    c: CycleMatrix, tone: int, relative_to: int = 0
) -> np.ndarray[tuple[int], np.dtype[DT.Cycle]]:
    return c[:, relative_to, tone]


def dist(c: CycleMatrix, tone1: int, tone2: int, signed=False) -> NDArray:
    relative_rank = c[:, tone1, tone2]
    dist = relative_rank.min()
    if signed and relative_rank.argmin() == 1:
        return -dist
    return dist


def get_n_closest_tones(r: RankMatrix, tone: int, n: int = 1):
    return r[:, tone, 1 : n + 1]
