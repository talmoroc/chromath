import numpy as np
from typing import Annotated, Final
from numpy.typing import NDArray


## np.dtypes used to limit memory usage and accelerate int operations
## (modulo, integer division, bitwise operations)
## Maybe use only int8 everywhere ?
class DT:
    Chr: Final = np.bool_  # Chroma
    St: Final = np.int8  # Semitones or rank


## Chroma Types

type ChromaArray = Annotated[NDArray[DT.Chr], "Shape: (..., MS.tones)"]
type ScaleChromaArray = Annotated[ChromaArray, "Shape: (..., MS.tones) with MS.degrees ones"]

## Degree Types
type IntervalArray = Annotated[NDArray[DT.St], "Shape: (..., 2) (degree, tone) representation"]
type ScaleIntervalArray = Annotated[IntervalArray, "Shape: (..., MS.degrees, 2)"]

## Cycle Types
type CycleArray = Annotated[NDArray[DT.St], "Shape: (..., MS.tones) tone -> cycle_rank map"]
type RankArray = Annotated[
    NDArray[DT.St], "Shape: (..., <= cycle periodicity) cycle_rank -> tone map"
]

type SymCycleArray = Annotated[NDArray[DT.St], "Shape: (..., tone, 2) with dim for cycle direction"]
type SymRankArray = Annotated[NDArray[DT.St], "Shape: (..., 2, rank) with dim for cycle direction"]
