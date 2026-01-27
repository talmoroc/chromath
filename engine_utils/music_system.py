from dataclasses import dataclass, field
from contextvars import ContextVar
from types import TracebackType
import numpy as np


@dataclass(frozen=True)
class MusicSystem:
    tones: int = 12
    degrees: int = 7
    max_st: int = 2**7 - 1
    powers: np.ndarray = field(init=False, repr=False)  # bitwise representation of tones

    def __post_init__(self):
        powers = 1 << np.arange(self.tones, dtype=object)
        object.__setattr__(self, 'powers', powers)
        object.__setattr__(self, 'max_st', powers[-1])


_current_system = ContextVar('music_system', default=MusicSystem(12, 7))


class use_system:
    def __init__(self, tones: int = 12, degrees: int = 7):
        self.token = None
        self.new_system = MusicSystem(tones, degrees)

    def __enter__(self) -> MusicSystem:
        self.token = _current_system.set(self.new_system)
        return self.new_system

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None):
        if self.token is not None:
            _current_system.reset(self.token)


def get_current_music_system() -> MusicSystem:
    return _current_system.get()
