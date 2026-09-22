"""Chroma: a set of pitch classes.

The canonical form is a 12-bit key (``uint16``), little-endian — bit *i* is
pitch class *i*, so index 0 is C and a C major chord is ``0b000010010001``.
``ChromaBoolArray`` is the twelve-lane view, materialised only where pitch
classes need to be addressed individually.

Keys are why the set operations are cheap: union, intersection and difference
are single instructions, and cardinality is ``bitwise_count``. Every function
here is elementwise, so it works on a scalar key or on any array of them.
"""

from collections.abc import Iterable
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaArray, ChromaBoolArray, SemitonesArray

MASK = np.uint16((1 << MS.tones) - 1)

# Bit-reversal of a 12-bit key, used by `invert`.
_REVERSED = np.array(
    [int(format(k, f"0{MS.tones}b")[::-1], 2) for k in range(1 << MS.tones)],
    dtype=DT.Key,
)


# VALIDATION


@overload
def validate_chroma(c: int) -> ChromaArray: ...
@overload
def validate_chroma(c: ArrayLike) -> ChromaArray: ...
def validate_chroma(c: ArrayLike | int) -> ChromaArray:
    """Reject keys outside the 12-bit range."""
    arr = np.asarray(c)
    if np.any(arr < 0) or np.any(arr >= (1 << MS.tones)):
        raise ValueError(f"Chromas are {MS.tones}-bits, got {c}")
    return arr.astype(DT.Key)


# GENERATION AND CONVERSION


def from_vector(v: ChromaBoolArray) -> ChromaArray:
    """Twelve-lane boolean view -> key."""
    arr = np.asarray(v, dtype=DT.Key)
    return (arr << np.arange(MS.tones, dtype=DT.Key)).sum(axis=-1, dtype=DT.Key)


@overload
def from_st(v: Iterable[int]) -> ChromaArray: ...
@overload
def from_st(v: SemitonesArray) -> ChromaArray: ...
@overload
def from_st(*v: int) -> ChromaArray: ...
def from_st(v: SemitonesArray | int | Iterable[int], *rest: int) -> ChromaArray:
    """Semitones -> key. Accepts a sequence or loose arguments; wraps mod 12."""
    idx = np.asarray((v, *rest) if rest else v, dtype=np.int64).ravel() % MS.tones
    if idx.size == 0:
        return np.array(0, dtype=DT.Key)
    return np.bitwise_or.reduce(np.left_shift(DT.Key(1), idx.astype(DT.Key)))


def to_vector(c: ChromaArray) -> ChromaBoolArray:
    """Key -> twelve-lane boolean view. Trailing axis is the pitch class."""
    arr = np.asarray(c, dtype=DT.Key)
    return ((arr[..., np.newaxis] >> np.arange(MS.tones, dtype=DT.Key)) & 1).astype(DT.Chroma)


def to_st(c: ChromaArray) -> SemitonesArray:
    """Key -> the semitones it contains, ascending."""
    arr = np.asarray(c, dtype=DT.Key)
    if arr.ndim:
        raise ValueError(f"to_st takes a single chroma, got shape {arr.shape}")
    return np.flatnonzero(to_vector(arr)).astype(DT.St)


# CORE OPERATIONS


def transpose(c: ChromaArray, shift: int) -> ChromaArray:
    """Move every pitch class up by *shift* semitones, wrapping at 12."""
    n = int(shift) % MS.tones
    arr = np.asarray(c, dtype=DT.Key)
    if n == 0:
        return arr
    return ((arr << DT.Key(n)) | (arr >> DT.Key(MS.tones - n))) & MASK


def invert(c: ChromaArray, pivot: int = 0) -> ChromaArray:
    """Musical inversion about *pivot*: pitch class j becomes pivot - j."""
    arr = np.asarray(c, dtype=DT.Key)
    return transpose(_REVERSED[arr], pivot + 1)


def transpositions(c: ChromaArray) -> ChromaArray:
    """All twelve transpositions of *c*, ascending by shift."""
    return np.stack([transpose(c, n) for n in range(MS.tones)])


# SET OPERATIONS AND MEASURES


def cardinality(c: ChromaArray) -> NDArray[np.uint8]:
    """How many pitch classes the chroma holds."""
    return np.bitwise_count(c)


def common_tones(c1: ChromaArray, c2: ChromaArray) -> NDArray[np.uint8]:
    """How many pitch classes the two chromas share."""
    return np.bitwise_count(c1 & c2)


def diverging_tones(c1: ChromaArray, c2: ChromaArray) -> NDArray[np.uint8]:
    """How many pitch classes the two chromas do not share."""
    return np.bitwise_count((c1 | c2) & ~(c1 & c2))


def isin(c: ChromaArray, container: ChromaArray) -> NDArray[np.bool_]:
    """Whether every pitch class of *c* is also in *container*."""
    arr = np.asarray(c, dtype=DT.Key)
    return (arr & np.asarray(container, dtype=DT.Key)) == arr


def _overlap(shared, total):
    """Fraction of *total* that is *shared*, defining 0/0 as 1.

    An empty chroma compared with an empty chroma shares everything there is
    to share, so the derived distance is 0 rather than nan.
    """
    return np.divide(shared, total, out=np.ones(np.broadcast(shared, total).shape), where=total != 0)


# Hamming: how many tones differ, relative to the twelve of the system.
def hamming(c1: ChromaArray, c2: ChromaArray) -> NDArray[np.float64]:
    """Tones present in one chroma but not the other, over MS.tones."""
    ct = common_tones(c1, c2)
    return (cardinality(c1) + cardinality(c2) - 2 * ct) / MS.tones


# Jaccard: how many tones differ, relative to the tones in either.
def jaccard(c1: ChromaArray, c2: ChromaArray) -> NDArray[np.float64]:
    """Symmetric difference over union. Two empty chromas are at distance 0."""
    ct = common_tones(c1, c2)
    return 1.0 - _overlap(ct, cardinality(c1) + cardinality(c2) - ct)


# Tversky: asymmetric — how much of `from_` is missing from `to_`.
def tversky(from_: ChromaArray, to_: ChromaArray) -> NDArray[np.float64]:
    """Measures how much 'from_' is included in 'to_'. Not symmetric."""
    return 1.0 - _overlap(common_tones(from_, to_), cardinality(from_))


# Symmetric Tversky: the reference is chosen as the minimum or maximum of difference to intersection, and weighted with beta and alpha
def tversky_symm(c1: ChromaArray, c2: ChromaArray, beta: float = 2.0, alpha: float = 0.8) -> NDArray[np.float64]:
    """
    *Symmetric Tversky* with beta >= 2.0 and 0 <= alpha <= 1
    - High alpha: more weight on the minimum difference (almost included = close, useful to compare similar chords with different extensions)
    - Low alpha: more weight on the maximum difference (many notes outside = far)
    - High beta: more weight on the difference parameter, the common tones weigh less.
    If beta = 2.0 and alpha = 0.5, we get jaccard.
    """
    intersection = common_tones(c1, c2)
    union = np.bitwise_count(c1 | c2, dtype=np.uint8)
    a = np.minimum(union - cardinality(c1), union - cardinality(c2))
    b = union - intersection - a
    return 1.0 - _overlap(intersection, (intersection + beta * (alpha * a + (1 - alpha) * b)))


DISTANCES = {"hamming": hamming, "jaccard": jaccard, "tversky_symm": tversky_symm, "tversky": tversky}


def closest(
    from_: ChromaArray,
    to_: ChromaArray,
    n: int = 1,
    dist: Literal["hamming", "jaccard", "tversky_symm", "tversky"] = "hamming",
) -> NDArray[np.intp]:
    """Indices into *to_* of the *n* chromas nearest each entry of *from_*."""
    if dist not in DISTANCES:
        raise ValueError(f"Unknown distance {dist!r}, choose from {sorted(DISTANCES)}")
    d = DISTANCES[dist](np.asarray(from_, dtype=DT.Key)[..., np.newaxis], np.asarray(to_, dtype=DT.Key))
    return np.argsort(d, axis=-1, stable=True)[..., :n]
