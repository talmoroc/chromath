from __future__ import annotations

import math
import random

import numpy as np
from beartype import BeartypeConf
from beartype.claw import beartype_package
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import strategies as st

# Check every chromath call against its annotations. Must run before chromath is imported.
beartype_package("chromath", conf=BeartypeConf(is_pep484_tower=True))

from chromath.constants import DefaultMusicSystem as MS  # noqa: E402
from chromath.core import chroma, cycle  # noqa: E402
from chromath.types import DT, ChromaKey, ChromaKeyArray, NoteKeyArray  # noqa: E402

pitch_classes = st.integers(min_value=0, max_value=MS.tones - 1)
shifts = st.integers(min_value=-24, max_value=24)
#: Any chroma key as a raw int.
chroma_ints = st.integers(min_value=0, max_value=(1 << MS.tones) - 1)
#: Any chroma key as a 0-d key array.
chromas = chroma_ints.map(chroma.validate_chroma_keys)


#: Raised for a value outside a Literal, by the function or by beartype.
BAD_CHOICE = (ValueError, BeartypeCallHintParamViolation)


def each(arr) -> list:
    """The entries of *arr* as 0-d arrays."""
    return [arr[(*i, ...)] for i in np.ndindex(np.shape(arr))]


# ── Entry points ────────────────────────────────────────────────────────


def note_of(n) -> NoteKeyArray:
    """The note at pitch class *n*."""
    return chroma.from_st(chroma.validate_note_index_array(n))


def chroma_of(members) -> ChromaKeyArray:
    """The chroma holding *members*, one per row."""
    return chroma.from_members(chroma.validate_chroma_members_array(np.asarray(members, dtype=np.int_)))


# ── Samples ─────────────────────────────────────────────────────────────
# Seeded, so a failure reproduces.

_sample = random.Random(20260922)

#: One cycle per step at a drawn start, plus the ones worth naming.
SAMPLE_CYCLES: list[tuple[int, int]] = sorted(
    {(step, _sample.randrange(MS.tones)) for step in range(MS.tones)} | {(0, 4), (1, 11), (3, 9), (4, 0), (6, 0), (7, 0)}
)

#: Keys spanning the empty chroma, single notes, the full set and a draw.
SAMPLE_KEYS: ChromaKeyArray = np.array(
    sorted({0, 1, (1 << MS.tones) - 1} | {1 << n for n in (0, 4, 7, 11)} | {_sample.randrange(1 << MS.tones) for _ in range(24)}), DT.Key
)


#: Shapes to start a cycle on, as semitone offsets from C.
BASE_SHAPES: dict[str, tuple[int, ...]] = {
    "unison": (0,),
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "dom7": (0, 4, 7, 10),
    "whole_tone": (0, 2, 4, 6, 8, 10),
    "dim7": (0, 3, 6, 9),
    "aug": (0, 4, 8),
    "tritone": (0, 6),
}
BASES: dict[str, ChromaKey] = {name: chroma.as_key(chroma_of(st)) for name, st in BASE_SHAPES.items()}
BASE_IDS: list[str] = list(BASES)

#: (step, start, base name) of the sampled cycles.
SAMPLE_BASED: list[tuple[int, int, str]] = sorted(
    {(step, start, BASE_IDS[i % len(BASE_IDS)]) for i, (step, start) in enumerate(SAMPLE_CYCLES)}
    | {(0, 4, "minor"), (1, 0, "whole_tone"), (2, 0, "tritone"), (3, 0, "dim7"), (4, 0, "aug"), (7, 0, "major")}
)
BASED_IDS: list[str] = [f"step{s}_from{t}_{n}" for s, t, n in SAMPLE_BASED]


def tones_of(key) -> set[int]:
    """Pitch classes of a key, read bit by bit."""
    return {i for i in range(MS.tones) if int(key) >> i & 1}


def symmetry_of(key) -> int:
    """Smallest transposition mapping the chroma onto itself."""
    tones = tones_of(key)
    return next(n for n in range(1, MS.tones + 1) if {(t + n) % MS.tones for t in tones} == tones)


def period_of(step: int, base: int) -> int:
    """How many steps before a base returns to itself."""
    symmetry = symmetry_of(base)
    return symmetry // math.gcd(step, symmetry)


def members_of(step: int, start: int, base: int) -> list[int]:
    """The chromas of one turn, built by transposing the base directly."""
    key = chroma.validate_chroma_keys(base)
    return [int(chroma.transpose(key, start + step * r)) for r in range(period_of(step, base))]


def start_of(start: int = 0, name: str = "unison") -> ChromaKeyArray:
    """The named base moved up *start* semitones."""
    return chroma.transpose(np.asarray(BASES[name]), start)


def turn(step: int, start: int = 0, name: str = "unison") -> ChromaKeyArray:
    """One turn of a sampled cycle."""
    return cycle.cycle(step, start_of(start, name))


def foreign_to(base) -> ChromaKeyArray:
    """A chroma the cycle can never reach."""
    return chroma_of([0, 1]) if chroma.cardinality(np.asarray(base)) == 1 else note_of(0)


def chroma_key_arrays(min_size: int = 0, max_size: int = 8, shape2d: bool = False):
    """Arrays of chroma keys."""
    if shape2d:
        return st.integers(0, 3).flatmap(
            lambda r: st.lists(chroma_ints, min_size=r * 2, max_size=r * 2).map(lambda xs: np.array(xs, dtype=np.uint16).reshape(r, 2))
        )
    return st.lists(chroma_ints, min_size=min_size, max_size=max_size).map(lambda xs: np.array(xs, dtype=np.uint16))


def chroma_keys(min_notes: int = 0, max_notes: int = MS.tones):
    """Chroma keys with cardinality in ``[min_notes, max_notes]``."""
    return st.sets(pitch_classes, min_size=min_notes, max_size=max_notes).map(lambda s: chroma_of(sorted(s)))


def notes():
    return chroma_keys(1, 1)


def intervals():
    """Two-note chromas."""
    return chroma_keys(2, 2)


def chords(min_notes: int = 3, max_notes: int = 7):
    return chroma_keys(min_notes, max_notes)


def scales():
    return chroma_keys(MS.degrees, MS.degrees)
