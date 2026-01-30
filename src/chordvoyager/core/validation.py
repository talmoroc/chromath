from src.chordvoyager.constants import DefaultMusicSystem as MS
from src.chordvoyager.types import (
    DTYPE,
    NDArray,
    ChromaVec,
    DegreeVec,
    ScaleChromaVec,
)


def validate_degree(v: NDArray) -> DegreeVec:
    if len(v) != MS.degrees:
        raise ValueError(f"Degree vectors have length {MS.degrees}, got {len(v)}")
    if not set(v) <= MS.tones_set:
        raise ValueError(f"Degree vectors have tones in [0-{MS.tones - 1}]")
    return v.astype(DTYPE.Deg)


def validate_chroma(v: NDArray) -> ChromaVec:
    if len(v) != MS.tones:
        raise ValueError(f"Chroma vectors have length {MS.tones}, got {len(v)}")
    if not set(v) <= {0, 1}:
        raise ValueError(f"Chroma vectors are binary (0 or 1), got {set(v)}")
    return v.astype(DTYPE.Chr)


def validate_scale_chroma(v: NDArray) -> ScaleChromaVec:
    v = validate_chroma(v)
    if not sum(v) == 7:
        raise ValueError(f"Scale chroma vectors have exactly 7 tones, got {sum(v)}")
    return v
