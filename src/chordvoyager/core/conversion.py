import numpy as np
from src.chordvoyager.types import ChromaVec, DegreeVec, ScaleChromaVec
from src.chordvoyager.constants import DefaultMusicSystem as MS
import src.chordvoyager.core.validation as val


def chroma_to_degree(v: ChromaVec) -> DegreeVec:
    deg = np.flatnonzero(v)
    return val.validate_degree(deg)


def degree_to_chroma(v: DegreeVec) -> ChromaVec:
    chr = np.zeros(MS.tones)
    chr[v] = 1
    return val.validate_chroma(chr)


def chroma_to_scale(v: ChromaVec) -> ScaleChromaVec:
    return val.validate_scale_chroma(v)
