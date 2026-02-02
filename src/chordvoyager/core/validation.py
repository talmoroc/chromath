from ..types import (
    DT,
    NDArray,
    ChromaVec,
    DegreeVec,
    ScaleChromaVec,
    CycleVec,
    CycleMatrix,
    RankMatrix,
)
from ..constants import DefaultMusicSystem as MS


def validate_degree(v: NDArray) -> DegreeVec:
    if len(v) != MS.degrees:
        raise ValueError(f"Degree vectors have length {MS.degrees}, got {len(v)}")
    if not set(v) <= MS.tones_set:
        raise ValueError(f"Degree vectors have tones in [0-{MS.tones - 1}]")
    if v.dtype != DT.Deg:
        raise ValueError(f"Degree vectors must have dtype {DT.Deg}, got {v.dtype}")
    return v


def validate_chroma(v: NDArray) -> ChromaVec:
    if len(v) != MS.tones:
        raise ValueError(f"Chroma vectors have length {MS.tones}, got {len(v)}")
    if not set(v) <= {0, 1}:
        raise ValueError(f"Chroma vectors are binary (0 or 1), got {set(v)}")
    if v.dtype != DT.Chr:
        raise ValueError(f"Chroma vectors must have dtype {DT.Chr}, got {v.dtype}")
    return v


def validate_scale_chroma(v: NDArray) -> ScaleChromaVec:
    v = validate_chroma(v)
    if not sum(v) == 7:
        raise ValueError(f"Scale chroma vectors have exactly 7 tones, got {sum(v)}")
    return v


def validate_cycle_vector(v: NDArray) -> CycleVec:
    if v.shape[0] != 2:
        raise ValueError(f"Cycle vectors have shape (2, n), got {v.shape}")
    if v.dtype != DT.Cycle:
        raise ValueError(f"Cycle matrices must have dtype {DT.Cycle}, got {v.dtype}")
    return v


def validate_rank_matrix(v: NDArray) -> RankMatrix:
    if v.shape[0] != 2 or v.shape[1] != MS.tones:
        raise ValueError(f"Cycle matrices have shape (2, {MS.tones}, m), got {v.shape}")
    if v.dtype != DT.Cycle:
        raise ValueError(f"Cycle matrices must have dtype {DT.Cycle}, got {v.dtype}")
    return v


def validate_cycle_matrix(v: NDArray) -> CycleMatrix:
    if v.shape[0] != 2 or v.shape[1] != MS.tones or v.shape[2] != MS.tones:
        raise ValueError(f"Cycle matrices have shape (2, {MS.tones}, {MS.tones}), got {v.shape}")
    if v.dtype != DT.Cycle:
        raise ValueError(f"Cycle matrices must have dtype {DT.Cycle}, got {v.dtype}")
    return v
