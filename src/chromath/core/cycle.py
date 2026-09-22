import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import NDArray

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaArray, SemitonesArray
from . import chroma

type Direction = Literal["forward", "backward", "min"]


@dataclass(frozen=True)
class Cycle:
    """An cycle of tones given by repeated transpositions by a fixed step.

    - step: the number of semitones between an element of the cycle and the next.
    - start: the starting semitone of the cycle.
    - periodicity: the number of tones before the cycle repeats itself. between 1-12
    """

    step: int
    start: int = 0

    def __post_init__(self):
        if not 0 <= self.step <= MS.tones - 1:
            raise ValueError(f"step must be in 0..{MS.tones - 1}, got {self.step}")
        if not 0 <= self.start < MS.tones:
            raise ValueError(f"start must be in 0..{MS.tones - 1}, got {self.start}")

    def _tone(self, rank):
        """The semitone reached after *rank* steps. Negative ranks go backward."""
        return (self.start + self.step * rank) % MS.tones

    def _ranks(self, key: slice, stop: int) -> NDArray[np.int64]:
        """The ranks a slice names, unclamped and signed."""
        return np.arange(
            0 if key.start is None else key.start,
            stop if key.stop is None else key.stop,
            1 if key.step is None else key.step,
        )

    def __len__(self) -> int:
        """The length of one full turn."""
        return self.period

    def __iter__(self) -> Iterator[np.int8]:
        """One full turn."""
        yield from self.semitones

    def __getitem__(self, key: int | slice) -> SemitonesArray:
        """The semitone at a rank, or the semitones a range of ranks names."""
        if isinstance(key, slice):
            return np.astype(self._tone(self._ranks(key, self.period)), DT.St)
        return np.asarray(self._tone(key), dtype=DT.St)

    @property
    def sym(self) -> "_SymmetricCycle":
        """Index this instead to alternate sides: ``c.sym[5]``."""
        return _SymmetricCycle(self)

    @property
    def period(self) -> int:
        """How many steps before the cycle closes."""
        return MS.tones // math.gcd(self.step, MS.tones)

    @property
    def is_complete(self) -> bool:
        """A complete cycle cycles through all tones"""
        return self.period == MS.tones

    @property
    def semitones(self) -> SemitonesArray:
        """The tones of this cycle in order, starting from start."""
        return self[:]

    @property
    def cycle_chroma(self) -> ChromaArray:
        """An array of size period with each value giving the chroma (uint16) of the tone."""
        return np.stack([chroma.from_st(st) for st in self.semitones], axis=-1)

    @property
    def mask(self) -> ChromaArray:
        """Twelve-bit key of which tones belong to this cycle."""
        return chroma.from_st(self.semitones)

    def rank(self, semitone: int, direction: Direction = "min") -> int:
        """Position of *semitone* within this cycle, or raise if outside it."""
        chroma_st = chroma.from_st(semitone)
        if not self.mask & chroma_st:
            raise IndexError(f"Semitone {semitone} not in Cycle(step={self.step}, start={self.start}) = {self.semitones}")

        forward_rank = int(np.flatnonzero(self.cycle_chroma & chroma_st)[0])
        backward_rank = -(self.period - forward_rank)
        match direction:
            case "forward":
                return forward_rank
            case "backward":
                return forward_rank if forward_rank == 0 else backward_rank
            case "min":
                return forward_rank if forward_rank < abs(backward_rank) else backward_rank
            case _:
                raise ValueError(f"direction must be one of {get_args(Direction)}")


@dataclass(frozen=True)
class _SymmetricCycle:
    """The tones of a cycle paired by distance: entry *d* is ranks +d and -d.

    Indexing by a distance gives one pair, by a range of distances a row per
    distance, so the result is always ``(..., 2)``.
    """

    cycle: Cycle

    def __len__(self) -> int:
        """The number of distinct distances, counting zero."""
        return len(self.cycle) // 2 + 1

    def __iter__(self) -> Iterator[SemitonesArray]:
        """One pair per distance, nearest first."""
        yield from self[:]

    def __getitem__(self, key: int | slice) -> SemitonesArray:
        """The pair at a distance, or a row per distance a range names."""
        d = np.asarray(self.cycle._ranks(key, len(self)) if isinstance(key, slice) else key)
        return np.astype(self.cycle._tone(np.stack([d, -d], axis=-1)), DT.St)
