import numpy as np
from numpy.typing import ArrayLike

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaKeyArray, InterpretedDegreeArray, ScaleLookupArray
from . import chroma


def chroma_to_interpreted(v: ArrayLike, lookup: ScaleLookupArray) -> InterpretedDegreeArray:
    """
    Interpret a chroma vector through a scale lookup (single-lookup fast path).
    """
    from .interval import _interpret_single_lookup

    st = chroma.to_members(v)  # type: ignore
    return _interpret_single_lookup(st, lookup)


def interpreted_to_chroma(v: InterpretedDegreeArray) -> ChromaKeyArray:
    """
    Reconstruct a chroma vector from an InterpretedIntervalArray using
    the semitone column.
    """
    st = v[..., 1].ravel()
    valid = st >= 0
    chroma_vec = np.zeros(MS.tones, dtype=DT.Bool)
    chroma_vec[st[valid]] = True
    return chroma.from_vector(chroma_vec)
