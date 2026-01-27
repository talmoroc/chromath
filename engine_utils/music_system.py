from typing import Any
from numpy.typing import NDArray, DTypeLike
from dataclasses import dataclass, field
from contextvars import ContextVar, Token
from types import TracebackType
import numpy as np


@dataclass(frozen=True)
class MusicSystem:
    tones: int = 12
    degrees: int = 7
    func_degrees: int = 13  # degrees meaningful for intervals
    powers: NDArray[np.int_] = field(init=False, repr=False)  # bitwise representation of tones
    max_int_repr: int = field(init=False, repr=False)

    def __post_init__(self):
        powers: NDArray = 1 << np.arange(self.tones, dtype=object)
        object.__setattr__(self, 'powers', powers)
        object.__setattr__(self, 'max_int_repr', int(powers.sum()))


_current_system = ContextVar('music_system', default=MusicSystem())


# DYNAMIC LOADING OF A MUSIC SYSTEM
class use_system:
    def __init__(self, tones: int = 12, degrees: int = 7):
        self.token: Token[MusicSystem] | None = None
        self.new_system = MusicSystem(tones, degrees)

    def __enter__(self) -> MusicSystem:
        self.token: Token[MusicSystem] = _current_system.set(self.new_system)
        return self.new_system

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None):
        if self.token is not None:
            _current_system.reset(self.token)


def get_current_music_system() -> MusicSystem:
    return _current_system.get()


__all__ = ['get_current_music_system', 'use_system']
