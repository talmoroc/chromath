import numpy as np
from numpy.typing import NDArray
import math
from .validation import (
    validate_chroma_array,
    validate_cycle_array,
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


def _generate_cycle_tones(step: int) -> NDArray:
    pi = periodicity(step)
    cycle_tones = np.arange(0, step * pi, step, dtype=DT.St) % MS.tones
    return cycle_tones


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
        step (int): the step of the Cycle, in semitones. Can be negative

    Returns:
        tuple[CycleVec, ChromaVec]:
            1. cycle_vec : index = tone, value = its rank in the cycle, len = tones
            2. mask : 1 = a tone is in the cycle, 0 = it is not. If the cycle is complete, it's full of 1.
                Useful to check before using the first CycleVec (Position in the cycle of a given tone)
    """
    step = step % MS.tones
    cycle_tones = _generate_cycle_tones(step)
    cycle_vec = np.full(MS.tones, -1, dtype=DT.St)
    cycle_vec[cycle_tones] = np.arange(periodicity(step))
    mask = np.where(cycle_vec == -1, 0, 1).astype(DT.Chr)
    return validate_cycle_array(cycle_vec), validate_chroma_array(mask)


# Compute the cycle rank of the tones relative to each tone
def generate_sym_cycle_matrix(step: int) -> SymCycleArray:
    pos_vec, _ = _generate_cycle_vector(step)
    neg_vec, _ = _generate_cycle_vector(-step)
    sym_vec = np.array([pos_vec, neg_vec])
    idx = np.arange(MS.tones)
    shift_indices = (idx - idx[:, np.newaxis]) % MS.tones
    cycle_array = sym_vec[:, shift_indices].transpose(1, 0, 2)  # (tone, direction, tone) -> rank
    return validate_cycle_array(cycle_array)


# Compute the cycle semitones series starting on each tone
def _matrix_over_ranks(c: CycleArray) -> RankArray:
    semitones_series = [(c + i) % MS.tones for i in range(MS.tones)]
    return np.array(semitones_series, dtype=DT.St)


def generate_sym_rank_array(step: int) -> SymRankArray:
    pos_rank = _generate_cycle_tones(step)
    neg_rank = _generate_cycle_tones(-step)
    sym_rank = np.array([pos_rank, neg_rank])
    rank_array = np.array([(sym_rank + i) % 12 for i in range(MS.tones)], dtype=DT.St)
    return validate_sym_rank_array(rank_array, periodicity(step))


# UTILITIES
# TODO: rendre ça compatible avec des matrices
def get_tone_from_rank(r: SymRankArray, rank: int, relative_to: int = 0) -> int:
    return r[relative_to, int(rank < 0), rank]


def get_rank_from_tone(c: SymCycleArray, tone: int, relative_to: int = 0) -> SymRankArray:
    return c[relative_to, :, tone]


def dist(c: SymCycleArray, tone1: int, tone2: int, signed=False) -> int:
    relative_rank = c[tone1, :, tone2]
    dist = relative_rank.min()
    if signed and relative_rank.argmin() == 1:
        return -dist
    return dist


def get_closest_tones(r: SymRankArray, tone: int, n: int = 1) -> SymCycleArray:
    return r[tone, :, :]
