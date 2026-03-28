import numpy as np
from numpy.typing import NDArray
from ..types import DT, ChromaArray, IntervalArray, ScaleChromaArray, ScaleIntervalArray
from ..constants import DefaultMusicSystem as MS
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
