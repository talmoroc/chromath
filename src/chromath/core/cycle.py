import math
from typing import Final, Literal, get_args

import numpy as np

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaKeyArray, ChromaMembersArray, IntArray
from . import chroma

type Direction = Literal["forward", "backward", "min"]
DIRECTIONS: Final = get_args(Direction.__value__)


def period(step: int, start: ChromaKeyArray) -> int:
    """How many steps before the start returns to itself"""
    returns = np.flatnonzero(np.asarray(chroma.transpositions(start))[1:] == start)
    symmetry = returns[0] + 1 if returns.size else MS.tones
    return int(symmetry // math.gcd(step, symmetry))


def cycle(step: int, start: ChromaKeyArray, stop: int | None = None) -> ChromaKeyArray:
    """The chromas of one turn, in walking order."""
    if stop is None:
        stop = period(step, start)
    ranks = np.arange(0, stop)
    return chroma.transpose(start, step * ranks)


def semitones(cycle: ChromaKeyArray, padded: bool = True) -> ChromaMembersArray:
    """The notes of each member, one row per rank."""
    if padded:
        return chroma.to_members(cycle)
    return chroma.to_members(cycle, chroma.cardinality(cycle)[0])


def mask(cycle: ChromaKeyArray) -> ChromaKeyArray:
    """Twelve-bit key of every tone the cycle touches."""
    return chroma.union(cycle)


def is_complete(keys: ChromaKeyArray) -> bool:
    """The Cycle covers all tones"""
    return bool(mask(keys) == chroma.MASK)


def rank(cycle: ChromaKeyArray, chroma: ChromaKeyArray, direction: Direction = "min") -> IntArray:
    """Rank of each chroma in the cycle. Raises if one is outside it."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    period: int = cycle.shape[-1]
    hits = chroma[..., np.newaxis] == cycle  # members are distinct within a turn
    found = hits.any(axis=-1)
    if not found.all():
        raise IndexError(f"Chromas {chroma} not in cycle {cycle}")

    forward = hits.argmax(axis=-1)
    backward = np.where(forward == 0, 0, forward - period)
    match direction:
        case "forward":
            return np.asarray(forward, dtype=DT.Int)
        case "backward":
            return np.asarray(backward, dtype=DT.Int)
        case "min":
            return np.asarray(np.where(forward < -backward, forward, backward), dtype=DT.Int)


def dist(cycle: ChromaKeyArray, a: ChromaKeyArray, b: ChromaKeyArray) -> IntArray:
    """Steps between two members the short way round."""
    period: int = cycle.shape[-1]
    d = (rank(cycle, b, "forward") - rank(cycle, a, "forward")) % period
    return np.asarray(np.minimum(d, period - d), dtype=DT.Int)
