"""Chromas: sets of pitch classes, as little-endian MS.tones-bit keys."""

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..constants import DefaultMusicSystem as MS
from ..types import (
    DT,
    SENTINEL,
    ChromaKey,
    ChromaKeyArray,
    ChromaMembersArray,
    ChromaVec,
    ChromaVecArray,
    IntArray,
    NoteIndex,
    NoteIndexArray,
    NoteKeyArray,
    NoteVecArray,
    ScoreArray,
)

MASK = np.uint16((1 << MS.tones) - 1)

_REVERSED = np.array(  # Bit-reversal of a 12-bit key, used by `invert`.
    [int(format(k, f"0{MS.tones}b")[::-1], 2) for k in range(1 << MS.tones)],
    dtype=DT.Key,
)


# VALIDATION


def validate_chroma_keys(batch: ArrayLike) -> ChromaKeyArray:
    """Reject keys outside the MS.tones-bit range."""
    arr = np.asarray(batch)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Chromas must be integers, got dtype {arr.dtype}")
    bad = (arr < 0) | (arr > MASK)
    if bad.any():
        raise ValueError(f"Chromas are {MS.tones}-bits, got {arr[bad]}")
    return arr.astype(DT.Key, copy=False)


def validate_chroma_vecs(batch: ArrayLike) -> ChromaVecArray:
    """Reject shapes other than (..., MS.tones) and values other than 0 or 1."""
    arr = np.asarray(batch)
    if not arr.shape[-1] == 12:
        raise ValueError(f"ChromaVecs must be (..., {MS.tones}), got shape {arr.shape}")
    bad = (arr != 0) & (arr != 1)
    if bad.any():
        raise ValueError(f"ChromaVecs must contain only 0 or 1, got {arr[bad]}")
    return arr.astype(DT.Bool, copy=False)


def validate_note_index_array(note_idx: ArrayLike) -> NoteIndexArray:
    """Reject non-integers and indices outside 0..MS.tones-1."""
    arr = np.asarray(note_idx)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Note indices must be integers, got dtype {arr.dtype}")
    bad = (arr < 0) | (arr >= MS.tones)
    if bad.any():
        raise ValueError(f"Note indices must be in 0..{MS.tones - 1}, got {arr[bad]}")
    return arr.astype(DT.Note, copy=False)


def validate_note_keys(batch: ArrayLike) -> NoteKeyArray:
    """Reject keys that do not hold exactly one note."""
    arr = validate_chroma_keys(batch)
    bad = np.bitwise_count(arr) != 1
    if bad.any():
        raise ValueError(f"Note keys must have exactly one bit set, got {np.asarray(arr)[bad]}")
    return arr


def validate_note_vecs(batch: ArrayLike) -> NoteVecArray:
    arr = validate_chroma_vecs(batch)
    bad = arr.sum(axis=-1) != 1
    if bad.any():
        raise ValueError(f"Note vectors must be one-hot, got {arr[bad]}")
    return arr


def validate_chroma_members_array(batch: ArrayLike) -> ChromaMembersArray:
    """Reject 0-d input and values other than pitch classes or SENTINEL."""
    arr = np.asarray(batch)
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError(f"Chromas must be integers, got dtype {arr.dtype}")
    if arr.ndim < 1:
        raise ValueError(f"ChromaMembersArray must be at least 1-d, got shape {arr.shape}")
    bad = (arr != SENTINEL) & ((arr < 0) | (arr >= MS.tones))
    if bad.any():
        raise ValueError(f"ChromaMembersArray must contain only integers in 0..{MS.tones - 1} or SENTINEL, got {arr[bad]}")
    return arr.astype(DT.Note, copy=False)


# NARROWING


def as_key(arr: ArrayLike) -> ChromaKey:
    arr = validate_chroma_keys(arr)
    if arr.ndim != 0:
        raise ValueError(f"ChromaKey is a single key, got shape {arr.shape}")
    return ChromaKey(DT.Key(arr))


def as_vec(arr: ArrayLike) -> ChromaVec:
    arr = validate_chroma_vecs(arr)
    if not arr.ndim == 1:
        raise ValueError(f"A ChromaVec is a single vector, got {arr.shape}")
    return arr


def as_note_index(arr: ArrayLike) -> NoteIndex:
    arr = validate_note_index_array(arr)
    if np.ndim(arr) != 0:
        raise ValueError(f"Expected a single index, got shape {np.shape(arr)}")
    return NoteIndex(DT.Note(arr))


# GENERATION AND CONVERSION


def from_vector(v: ChromaVecArray) -> ChromaKeyArray:
    """Twelve-lane boolean view -> key."""
    chroma_vecs = validate_chroma_vecs(v)
    chroma_keys = (chroma_vecs << np.arange(MS.tones)).sum(axis=-1)
    return np.asarray(chroma_keys, DT.Key)


def from_members(members: ChromaMembersArray) -> ChromaKeyArray:
    """Member indices -> key, over the last axis. SENTINEL slots are skipped."""
    present = members != SENTINEL
    bits = DT.Key(1) << np.where(present, members, 0).astype(DT.Key)
    return union(np.where(present, bits, DT.Key(0)))


def from_st(st: NoteIndexArray) -> NoteKeyArray:
    """Note indices -> note keys."""
    return np.asarray(DT.Key(1) << np.asarray(st, dtype=DT.Note), dtype=DT.Key)


def to_vector(c: ChromaKeyArray) -> ChromaVecArray:
    """Key -> twelve-lane boolean view. Trailing axis is the pitch class."""
    chroma_vecs = (c[..., np.newaxis] >> np.arange(MS.tones)) & 1
    chroma_vecs = np.astype(chroma_vecs, DT.Bool)
    return chroma_vecs


def to_members(c: ChromaKeyArray, width=MS.max_chroma_members) -> ChromaMembersArray:
    """Key -> its pitch classes, ascending, padded with SENTINEL to *width*."""
    max_cardinality = int(np.max(cardinality(c), initial=0))
    if width < max_cardinality:
        raise ValueError(f"width {width} is smaller than the largest chroma ({max_cardinality} notes)")
    vec = to_vector(c)
    order = np.argsort(~vec, axis=-1, kind="stable").astype(DT.Note)  # members first, ascending
    padded = np.where(np.take_along_axis(vec, order, axis=-1), order, SENTINEL)
    return padded[..., :width]


def to_st(c: NoteKeyArray) -> NoteIndexArray:
    """Note keys -> note indices."""
    return np.asarray(np.bitwise_count(c - 1, dtype=np.uint8), dtype=DT.Note)


# CORE OPERATIONS


def union(c: ChromaKeyArray) -> ChromaKeyArray:
    """Pitch classes held by any chroma along the last axis: (..., n) -> (...)."""
    return np.asarray(np.bitwise_or.reduce(c, axis=-1), dtype=DT.Key)


def transpositions(c: ChromaKeyArray) -> ChromaKeyArray:
    """All twelve transpositions of *c*, ascending by shift."""
    return np.stack([((c << DT.Key(n)) | (c >> DT.Key(MS.tones - n))) & MASK for n in range(MS.tones)])


def transpose(c: ChromaKeyArray, shift: ArrayLike) -> ChromaKeyArray:
    """Move every pitch class up by *shift* semitones, wrapping at 12."""
    shift = np.asarray(shift) % MS.tones
    return np.asarray(transpositions(c)[shift])


def invert(c: ChromaKeyArray, pivot: int = 0) -> ChromaKeyArray:
    """Musical inversion about *pivot*: pitch class j becomes pivot - j."""
    return transpose(np.asarray(_REVERSED[c]), pivot + 1)


# SET OPERATIONS AND MEASURES


def cardinality(c: ChromaKeyArray) -> IntArray:
    """How many pitch classes the chroma holds."""
    return np.asarray(np.bitwise_count(c), dtype=DT.Int)


def common_tones(c1: ChromaKeyArray, c2: ChromaKeyArray) -> IntArray:
    """How many pitch classes the two chromas share."""
    return np.asarray(np.bitwise_count(c1 & c2), dtype=DT.Int)


def diverging_tones(c1: ChromaKeyArray, c2: ChromaKeyArray) -> IntArray:
    """How many pitch classes the two chromas do not share."""
    return np.asarray(np.bitwise_count((c1 | c2) & ~(c1 & c2)), dtype=DT.Int)


def isin(c: ChromaKeyArray, container: ChromaKeyArray) -> NDArray[DT.Bool]:
    """Whether every pitch class of *c* is also in *container*."""
    arr = np.asarray(c, dtype=DT.Key)
    return np.asarray((arr & np.asarray(container, dtype=DT.Key)) == arr, dtype=DT.Bool)


def _overlap(shared, total):
    """Fraction of *total* that is *shared*, with 0/0 defined as 1."""
    return np.divide(shared, total, out=np.ones(np.broadcast(shared, total).shape), where=total != 0)


def hamming(c1: ChromaKeyArray, c2: ChromaKeyArray) -> ScoreArray:
    """Tones present in one chroma but not the other, over MS.tones."""
    ct = common_tones(c1, c2)
    return np.asarray((cardinality(c1) + cardinality(c2) - 2 * ct) / MS.tones, DT.Score)


def jaccard(c1: ChromaKeyArray, c2: ChromaKeyArray) -> ScoreArray:
    """Symmetric difference over union. Two empty chromas are at distance 0."""
    ct = common_tones(c1, c2)
    return np.asarray(1.0 - _overlap(ct, cardinality(c1) + cardinality(c2) - ct), DT.Score)


def tversky(from_: ChromaKeyArray, to_: ChromaKeyArray) -> ScoreArray:
    """How much of *from_* is missing from *to_*. Not symmetric."""
    return np.asarray(1.0 - _overlap(common_tones(from_, to_), cardinality(from_)), DT.Score)


def tversky_symm(c1: ChromaKeyArray, c2: ChromaKeyArray, beta: float = 2.0, alpha: float = 0.8) -> ScoreArray:
    """Symmetric Tversky: *alpha* weighs the smaller difference, *beta* the differences against the common tones. beta=2, alpha=0.5 is Jaccard."""
    intersection = common_tones(c1, c2)
    union = np.bitwise_count(c1 | c2, dtype=np.uint8)
    a = np.minimum(union - cardinality(c1), union - cardinality(c2))
    b = union - intersection - a
    return np.asarray(1.0 - _overlap(intersection, (intersection + beta * (alpha * a + (1 - alpha) * b))), DT.Score)


DISTANCES = {"hamming": hamming, "jaccard": jaccard, "tversky_symm": tversky_symm, "tversky": tversky}
METRICS = [hamming, jaccard]
SYMMETRIC = [*METRICS, tversky_symm]
ALL_DISTANCES = [*SYMMETRIC, tversky]
DISTANCE_IDS = [d.__name__ for d in ALL_DISTANCES]


def closest(
    from_: ChromaKeyArray,
    to_: ChromaKeyArray,
    n: int = 1,
    dist: Literal["hamming", "jaccard", "tversky_symm", "tversky"] = "hamming",
) -> IntArray:
    """Indices into *to_* of the *n* chromas nearest each entry of *from_*."""
    if dist not in DISTANCES:
        raise ValueError(f"Unknown distance {dist!r}, choose from {sorted(DISTANCES)}")
    d = DISTANCES[dist](np.asarray(from_, dtype=DT.Key)[..., np.newaxis], np.asarray(to_, dtype=DT.Key))
    return np.asarray(np.argsort(d, axis=-1, stable=True)[..., :n], DT.Int)
