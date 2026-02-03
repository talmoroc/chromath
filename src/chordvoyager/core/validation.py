import numpy as np
from typing import TypeGuard
from numpy.typing import NDArray, DTypeLike, ArrayLike
from ..types import (
    DT,
    ChromaArray,
    ScaleChromaArray,
    IntervalArray,
    ScaleIntervalArray,
    CycleArray,
    SymCycleArray,
    SymRankArray,
)
from ..constants import DefaultMusicSystem as MS


def validate_array(
    arr: ArrayLike, expected_tail: tuple[int, ...], dtype: DTypeLike, name: str
) -> NDArray:
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr, dtype=dtype)
    if arr.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {arr.dtype}")
    tail_length = len(expected_tail)
    if arr.shape[-tail_length:] != expected_tail:
        raise ValueError(f"{name} must end with shape {expected_tail}, got {arr.shape}")
    return arr


# Chroma Objects Validation


def validate_chroma_array(arr: ArrayLike) -> ChromaArray:
    return validate_array(arr, (MS.tones,), DT.Chr, "ChromaArray")


def is_scale_chroma(c: ChromaArray) -> TypeGuard[ScaleChromaArray]:
    return bool(np.all(np.sum(c, axis=-1) == MS.degrees))


def validate_scale_chroma_array(v: ArrayLike) -> ScaleChromaArray:
    v_chr = validate_chroma_array(v)
    if not is_scale_chroma(v_chr):
        raise ValueError(f"Expected {MS.degrees} tones")
    return v_chr


## Cycle Object Validation


def validate_cycle_array(arr: ArrayLike) -> CycleArray:
    return validate_array(arr, (MS.tones,), DT.St, "CycleArray")


def validate_sym_cycle_array(arr: ArrayLike) -> SymCycleArray:
    return validate_array(arr, (2, MS.tones), DT.St, "SymCycleArray")


def validate_sym_rank_array(arr: ArrayLike, periodicity: int) -> SymRankArray:
    return validate_array(arr, (2, periodicity), DT.St, "SymRankArray")


# Degree Object Validations


def validate_interval_array(arr: ArrayLike) -> IntervalArray:
    return validate_array(arr, (2,), DT.St, "IntervalArray")


def validate_scale_interval_array(arr: ArrayLike) -> ScaleIntervalArray:
    return validate_array(arr, (MS.degrees, 2), DT.St, "IntervalArray")
