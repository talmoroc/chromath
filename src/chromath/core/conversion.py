import numpy as np

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaBoolArray, InterpretedIntervalArray, ScaleLookupArray
from . import chroma


def chroma_to_interpreted(v: ChromaBoolArray, lookup: ScaleLookupArray) -> InterpretedIntervalArray:
    """
    Interpret a chroma vector through a scale lookup (single-lookup fast path).
    """
    from .interval import _interpret_single_lookup

    st = chroma.to_st(v)
    return _interpret_single_lookup(st, lookup)


def interpreted_to_chroma(v: InterpretedIntervalArray) -> ChromaBoolArray:
    """
    Reconstruct a chroma vector from an InterpretedIntervalArray using
    the semitone column.
    """
    st = v[..., 1].ravel()
    valid = st >= 0
    chroma = np.zeros(MS.tones, dtype=DT.Chroma)
    chroma[st[valid]] = True
    return chroma
