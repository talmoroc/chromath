from __future__ import annotations

import random

import numpy as np
from hypothesis import strategies as st

from chromath.constants import DefaultMusicSystem as MS
from chromath.core import chroma
from chromath.types import DT, ChromaArray

pitch_classes = st.integers(min_value=0, max_value=MS.tones - 1)
shifts = st.integers(min_value=-24, max_value=24)
#: Any chroma key. Twelve bits, so the largest is 4095.
chromas = st.integers(min_value=0, max_value=(1 << MS.tones) - 1)


# ── Samples ─────────────────────────────────────────────────────────────
# Parametrising over every cycle and every chroma is slower than it is
# informative. These are seeded, so a failure reproduces.

_sample = random.Random(20260922)

#: One cycle per step at a drawn start, plus the ones worth naming.
SAMPLE_CYCLES: list[tuple[int, int]] = sorted(
    {(step, _sample.randrange(MS.tones)) for step in range(MS.tones)} | {(0, 4), (1, 11), (3, 9), (4, 0), (6, 0), (7, 0)}
)
CYCLE_IDS: list[str] = [f"step{s}_from{t}" for s, t in SAMPLE_CYCLES]

#: Keys spanning the empty chroma, single notes, the full set and a draw.
SAMPLE_KEYS: ChromaArray = np.array(
    sorted({0, 1, (1 << MS.tones) - 1} | {1 << n for n in (0, 4, 7, 11)} | {_sample.randrange(1 << MS.tones) for _ in range(24)}), DT.Key
)


def tones_of(key) -> set[int]:
    """Pitch classes of a key, read bit by bit rather than via chromath."""
    return {i for i in range(MS.tones) if int(key) >> i & 1}


def orbit_of(step: int, start: int) -> set[int]:
    """Every tone reachable from *start* by adding *step*."""
    return {(start + k * step) % MS.tones for k in range(MS.tones)}


def chroma_key_arrays(min_size: int = 0, max_size: int = 8, shape2d: bool = False):
    """Arrays of chroma keys, for checking that operations stay elementwise."""
    if shape2d:
        return st.integers(0, 3).flatmap(
            lambda r: st.lists(chromas, min_size=r * 2, max_size=r * 2).map(lambda xs: np.array(xs, dtype=np.uint16).reshape(r, 2))
        )
    return st.lists(chromas, min_size=min_size, max_size=max_size).map(lambda xs: np.array(xs, dtype=np.uint16))


def chroma_keys(min_notes: int = 0, max_notes: int = MS.tones):
    """Chroma keys with cardinality in ``[min_notes, max_notes]``."""
    return st.sets(pitch_classes, min_size=min_notes, max_size=max_notes).map(lambda s: chroma.from_st(sorted(s)))


def notes():
    return chroma_keys(1, 1)


def intervals():
    """Two-note chromas. Simultaneous and sequential intervals are one type."""
    return chroma_keys(2, 2)


def chords(min_notes: int = 3, max_notes: int = 7):
    return chroma_keys(min_notes, max_notes)


def scales():
    return chroma_keys(MS.degrees, MS.degrees)
