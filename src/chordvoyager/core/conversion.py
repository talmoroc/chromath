import numpy as np
from ..types import DT, Chroma, Degrees, ScaleChroma, ScaleDegrees
from ..constants import DefaultMusicSystem as MS
from . import validation as val


def chroma_to_degree(v: Chroma) -> Degrees:
    deg = np.flatnonzero(v).astype(DT.Deg)
    return val.validate_degree(deg)


def degree_to_chroma(v: Degrees) -> Chroma:
    chroma = np.zeros(MS.tones)
    chroma[v] = 1
    chroma = chroma.astype(DT.Chr)
    return val.validate_chroma(chroma)


def chroma_to_scale_chroma(v: Chroma) -> ScaleChroma:
    return val.validate_scale_chroma(v)


def scale_chroma_to_degree(v: ScaleChroma) -> ScaleDegrees:
    deg = chroma_to_degree(v)
    return val.validate_scale_degree_vector(deg)
