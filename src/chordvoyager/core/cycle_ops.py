import numpy as np
import math
from .validation import (
    validate_chroma_array,
    validate_cycle_array,
    validate_sym_cycle_array,
    validate_sym_rank_array,
)

from ..types import (
    DT,
    ChromaArray,
    CycleArray,
    RankArray,
    SymCycleArray,
    SymRankArray,
)

from ..constants import DefaultMusicSystem as MS


def periodicity(step: int) -> int:
    return MS.tones // math.gcd(step, MS.tones)


def is_complete(periodicity: int) -> bool:
    return periodicity == MS.tones


def _generate_cycle_tones(step: int) -> CycleArray:
    pi = periodicity(step)
    cycle_tones = np.arange(0, step * pi, step, dtype=DT.St) % MS.tones
    return validate_cycle_array(cycle_tones)


def _generate_cycle_vector(step: int) -> tuple[CycleArray, ChromaArray]:
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
    cycle_vec = np.full(MS.tones, -1, dtype=DT.St)
    cycle_vec[cycle_tones] = np.arange(periodicity(step))
    mask = np.where(cycle_vec == -1, 0, 1).astype(DT.Chr)
    return validate_cycle_array(cycle_vec), validate_chroma_array(mask)


# Compute the cycle rank of the tones relative to each tone
def _matrix_over_tones(c: CycleArray) -> CycleArray:
    ranks_relative_to_each_tone = [(np.roll(c, i)) for i in range(MS.tones)]
    return validate_cycle_array(np.array(ranks_relative_to_each_tone, dtype=DT.St))


def generate_sym_cycle_matrix(step: int) -> SymCycleArray:
    pos_cycle_vec, mask = _generate_cycle_vector(step)
    pos_cycle_mat = _matrix_over_tones(pos_cycle_vec)
    neg_cycle_mat = pos_cycle_mat.transpose()  # possible because square matrix
    cycle_matrix = np.array([pos_cycle_mat, neg_cycle_mat], dtype=DT.St)
    breakpoint()
    return validate_sym_cycle_array(cycle_matrix)


# Compute the cycle semitones series starting on each tone
def _matrix_over_ranks(c: CycleArray) -> RankArray:
    semitones_series = [(c + i) % MS.tones for i in range(MS.tones)]
    return np.array(semitones_series, dtype=DT.St)


def generate_sym_rank_matrix(step: int) -> SymRankArray:
    pos_rank = _generate_cycle_tones(step)
    neg_rank = _generate_cycle_tones(-step)
    pos_rank_mat = _matrix_over_ranks(pos_rank)
    neg_rank_mat = _matrix_over_ranks(neg_rank)
    rank_matrix = np.array([pos_rank_mat, neg_rank_mat], dtype=DT.St)
    return validate_sym_rank_array(rank_matrix, periodicity(step))


# UTILITIES
def get_tone_from_rank(r: SymRankArray, rank: int, relative_to: int = 0) -> int:
    return r[rank < 0, relative_to, rank]


def get_rank_from_tone(c: SymCycleArray, tone: int, relative_to: int = 0) -> SymRankArray:
    return c[:, relative_to, tone]


def dist(c: SymCycleArray, tone1: int, tone2: int, signed=False) -> int:
    relative_rank = c[:, tone1, tone2]
    dist = relative_rank.min()
    if signed and relative_rank.argmin() == 1:
        return -dist
    return dist


def get_closest_tones(r: SymRankArray, tone: int, n: int = 1) -> SymCycleArray:
    return r[:, tone, :]
