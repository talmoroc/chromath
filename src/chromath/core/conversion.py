import numpy as np
from numpy.typing import NDArray

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaArray, InterpretedIntervalArray, IntervalArray, ScaleChromaArray, ScaleIntervalArray, ScaleLookupArray
from . import validation as val


def chroma_to_degree(v: ChromaArray) -> IntervalArray:
    deg = np.flatnonzero(v).astype(DT.St)
    return val.validate_interval_array(deg)


def chroma_to_semitones(v: ChromaArray) -> NDArray[DT.St]:
    return np.flatnonzero(v).astype(DT.St)


def degree_to_chroma(v: IntervalArray) -> ChromaArray:
    chroma = np.zeros(MS.tones)
    chroma[v] = 1
    chroma = chroma.astype(DT.Chr)
    return val.validate_chroma_array(chroma)


def chroma_to_scale_chroma(v: ChromaArray) -> ScaleChromaArray:
    return val.validate_scale_chroma_array(v)


def scale_chroma_to_degree(v: ScaleChromaArray) -> ScaleIntervalArray:
    deg = chroma_to_degree(v)
    return val.validate_scale_interval_array(deg)


def chroma_to_interpreted(v: ChromaArray, lookup: ScaleLookupArray) -> InterpretedIntervalArray:
    """
    Interpret a chroma vector through a scale lookup (single-lookup fast path).
    """
    from .interval_ops import _interpret_single_lookup

    st = chroma_to_semitones(v)
    return _interpret_single_lookup(st, lookup)


def interpreted_to_chroma(v: InterpretedIntervalArray) -> ChromaArray:
    """
    Reconstruct a chroma vector from an InterpretedIntervalArray using
    the semitone column.
    """
    st = v[..., 1].ravel()
    valid = st >= 0
    chroma = np.zeros(MS.tones, dtype=DT.Chr)
    chroma[st[valid]] = True
    return val.validate_chroma_array(chroma)
