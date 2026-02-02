from numpy.typing import NDArray, DTypeLike
from ..types import (
    DT,
    Chroma,
    ScaleChroma,
    Degrees,
    ScaleDegrees,
    CycleVec,
    SymCycleVec,
    CycleMatrix,
    SymCycleMatrix,
    RankVec,
    SymRankVec,
    RankMatrix,
    SymRankMatrix,
)
from ..constants import DefaultMusicSystem as MS


def _validate_dim_number(arr: NDArray, dims_number: int, class_name: str):
    if len(arr.shape) != dims_number:
        raise ValueError(f"{class_name} must have {dims_number} dimensions, got {len(arr.shape)}")


def _validate_shape(arr: NDArray, dim: int, length: int, class_name: str):
    if arr.shape[dim] != length:
        raise ValueError(f"{class_name} dimension {dim} must have length {length}, got {arr.shape[dim]}")


def _validate_shape_max(arr: NDArray, dim: int, length: int, class_name: str):
    if arr.shape[dim] > length:
        raise ValueError(f"{class_name} dimension {dim} must have length {length}, got {arr.shape[dim]}")


def _validate_values(arr: NDArray, set_values: set, class_name: str):
    if not set(arr) <= set_values:
        raise ValueError(f"{class_name} values must be in {set_values}, got {set(arr)}")


def _validate_dtype(arr: NDArray, dtype: DTypeLike, class_name: str):
    if arr.dtype != dtype:
        raise ValueError(f"{class_name} must have dtype {dtype}, got {arr.dtype}")


## Cycle Object Validation


def validate_cycle_vector(v: NDArray) -> CycleVec:
    _validate_shape_max(v, 0, MS.tones, "CycleVec")
    _validate_values(v, MS.tones_set.union({-1}), "CycleVec")
    _validate_dtype(v, DT.Cycle, "CycleVec")
    return v


def validate_sym_cycle_vector(v: NDArray) -> SymCycleVec:
    _validate_dim_number(v, 2, "SymCycleVec")
    _validate_shape(v, 0, 2, "SymCycleVec")
    validate_cycle_vector(v[0])
    validate_cycle_vector(v[1])
    return v


def validate_cycle_matrix(v: NDArray) -> CycleMatrix:
    _validate_dim_number(v, 2, "CycleMatrix")
    _validate_shape(v, 0, MS.tones, "CycleMatrix")
    _validate_shape(v, 1, MS.tones, "CycleMatrix")
    _validate_dtype(v, DT.Cycle, "CycleMatrix")
    return v


def validate_sym_cycle_matrix(v: NDArray) -> SymCycleMatrix:
    _validate_dim_number(v, 3, "SymCycleMatrix")
    _validate_shape(v, 0, 2, "SymCycleMatrix")
    validate_cycle_matrix(v[0])
    validate_cycle_matrix(v[1])
    return v


def validate_rank_vector(v: NDArray) -> RankVec:
    _validate_shape_max(v, 0, MS.tones, "RankVec")
    _validate_values(v, MS.tones_set, "RankVec")
    _validate_dtype(v, DT.Cycle, "RankVec")
    return v


def validate_sym_rank_vector(v: NDArray) -> SymRankVec:
    _validate_dim_number(v, 2, "SymRankVec")
    _validate_shape(v, 0, 2, "SymRankVec")
    validate_rank_vector(v[0])
    validate_rank_vector(v[1])
    return v


def validate_rank_matrix(v: NDArray) -> RankMatrix:
    _validate_dim_number(v, 2, "RankMatrix")
    _validate_shape(v, 0, MS.tones, "RankMatrix")
    _validate_shape_max(v, 1, MS.tones, "RankMatrix")
    _validate_dtype(v, DT.Cycle, "RankMatrix")
    return v


def validate_sym_rank_matrix(v: NDArray) -> SymRankMatrix:
    _validate_dim_number(v, 3, "SymRankMatrix")
    _validate_shape(v, 0, 2, "SymRankMatrix")
    validate_rank_matrix(v[0])
    validate_rank_matrix(v[1])
    return v


## Chroma Object Validation


def validate_chroma(v: NDArray) -> Chroma:
    _validate_shape(v, 0, MS.tones, "Chroma")
    _validate_values(v, {0, 1}, "Chroma")
    _validate_dtype(v, DT.Chr, "Chroma")
    return v


def validate_scale_chroma(v: NDArray) -> ScaleChroma:
    validate_chroma(v)
    if not sum(v) == 7:
        raise ValueError(f"Scale chroma vectors have exactly 7 tones, got {sum(v)}")
    return v


# Degree Object Validations


def validate_degree(v: NDArray) -> Degrees:
    if len(v) != MS.degrees:
        raise ValueError(f"Degree vectors have length {MS.degrees}, got {len(v)}")
    if not set(v) <= MS.tones_set:
        raise ValueError(f"Degree vectors have tones in [0-{MS.tones - 1}]")
    if v.dtype != DT.Deg:
        raise ValueError(f"Degree vectors must have dtype {DT.Deg}, got {v.dtype}")
    return v


def validate_scale_degree_vector(v: NDArray) -> ScaleDegrees:
    v = validate_degree(v)
    if not len(v) == 7:
        raise ValueError(f"Scale chroma vectors have exactly 7 tones, got {sum(v)}")
    return v
