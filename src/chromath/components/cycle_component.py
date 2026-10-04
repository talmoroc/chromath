from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..core import chroma, cycle
from ..core.cycle import Direction
from ..types import DT, ChromaKey, ChromaKeyArray, ChromaMembersArray, IntArray

C: Final = ChromaKey(DT.Key(1))


def _ranks(key: int | slice, stop: int) -> NDArray[np.integer]:
    """The ranks an index names: one rank, or a slice read unclamped and signed."""
    if not isinstance(key, slice):
        return np.asarray(key)
    return np.arange(key.start or 0, stop if key.stop is None else key.stop, key.step or 1)


@dataclass(frozen=True, init=False)
class Cycle:
    """A cycle of chromas given by repeated transpositions of a starting chroma by a fixed step.

    - step: the number of semitones between an element of the cycle and the next.
    - start: the chroma key of the first element, a single C by default.

    This is the public face of `core.cycle`: it validates what it is given, then
    hands over to the trusted functions.
    """

    step: int
    start: ChromaKey

    def __init__(self, step: int, start: ArrayLike = C):
        # Frozen, so the parsed values are stored through object.__setattr__.
        object.__setattr__(self, "step", int(chroma.as_note_index(step)))
        object.__setattr__(self, "start", chroma.as_key(start))

    @property
    def members(self) -> ChromaKeyArray:
        """The chromas of one turn, in walking order."""
        return cycle.cycle(self.step, np.asarray(self.start))

    @property
    def period(self) -> int:
        """How many steps before the start returns to itself, between 1 and 12."""
        return len(self.members)

    @property
    def mask(self) -> ChromaKeyArray:
        """Twelve-bit key of every tone the cycle touches."""
        return cycle.mask(self.members)

    @property
    def is_complete(self) -> bool:
        """The cycle covers all tones."""
        return cycle.is_complete(self.members)

    def semitones(self, padded: bool = True) -> ChromaMembersArray:
        """The notes of each member, one row per rank."""
        return cycle.semitones(self.members, padded)

    def rank(self, c: ArrayLike, direction: Direction = "min") -> IntArray:
        """Position of each chroma of *c* within this cycle, or raise if one is outside it."""
        return cycle.rank(self.members, chroma.validate_chroma_keys(c), direction)

    def dist(self, a: ArrayLike, b: ArrayLike) -> IntArray:
        """Steps between two members the short way round, elementwise and broadcast."""
        return cycle.dist(self.members, chroma.validate_chroma_keys(a), chroma.validate_chroma_keys(b))

    def at(self, ranks: ArrayLike) -> ChromaKeyArray:
        """The chromas reached after *ranks* steps. Negative ranks go backward."""
        members = self.members
        return np.asarray(members[np.asarray(ranks) % len(members)])

    def __len__(self) -> int:
        return self.period

    def __iter__(self) -> Iterator[DT.Key]:
        return iter(self.members)

    def __getitem__(self, key: int | slice) -> ChromaKeyArray:
        """The chroma at a rank, or the chromas a range of ranks names."""
        return self.at(_ranks(key, len(self)))

    @property
    def sym(self) -> "_SymmetricCycle":
        """Index this instead to alternate sides: ``c.sym[5]``."""
        return _SymmetricCycle(self)


@dataclass(frozen=True)
class _SymmetricCycle:
    """The members of a cycle paired by distance: entry *d* is ranks +d and -d, so the result is always ``(..., 2)``."""

    cycle: Cycle

    def __len__(self) -> int:
        """The number of distinct distances, counting zero."""
        return len(self.cycle) // 2 + 1

    def __iter__(self) -> Iterator[ChromaKeyArray]:
        return iter(self[:])

    def __getitem__(self, key: int | slice) -> ChromaKeyArray:
        d = _ranks(key, len(self))
        return self.cycle.at(np.stack([d, -d], axis=-1))
