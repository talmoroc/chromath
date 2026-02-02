import numpy as np
from numpy.typing import NDArray


from typing import Annotated, Final


## np.dtypes used to limit memory usage and accelerate int operations
## (modulo, integer division, bitwise operations)
## Maybe use only int8 everywhere ?
class DT:
    Chr: Final = np.bool_
    Deg: Final = np.int8
    Cycle: Final = np.int8


## Generic Types
type Vec[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]
type Mat[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type Cube[T: np.generic] = np.ndarray[tuple[int, int, int], np.dtype[T]]


type NDArrayBool = NDArray[np.bool_]
type NDArrayInt8 = NDArray[np.int8]
type NDArrayInt64 = NDArray[np.int64]


## Chroma Types
type Chroma = Annotated[Vec[DT.Chr], "(Tones,) bool mask"]
type ScaleChroma = Annotated[Chroma, "(Tones,) bool mask with a fixed number of True values"]
type ChromaMatrix = Annotated[Mat[DT.Chr], "Square matrix of chroma"]
type ScaleChromaMatrix = Annotated[ChromaMatrix, "Square matrix of ScaleChroma"]

## Degree Types
type Degrees = Annotated[Vec[DT.Deg], "(N,) degree->tone map"]
type ScaleDegrees = Annotated[Degrees, "(Degrees,) degree -> tone map"]
type Interval = Annotated[Mat[DT.Deg], ""]


## Cycle Types
type CycleVec = Annotated[Vec[DT.Cycle], "(Tones,) tone->cycle_rank map (-1 if tone not in cycle)"]
type RankVec = Annotated[Vec[DT.Cycle], "(Periodicity,) cycle_rank->tone map"]
type CycleMatrix = Annotated[Mat[DT.Cycle], "Matrix of CycleVecs: (starting_tone, tone) -> rank"]
type RankMatrix = Annotated[Mat[DT.Cycle], "Matrix of RankVecs: (starting_tone, rank) -> tone"]

type SymCycleVec = Annotated[Mat[DT.Cycle], "(dir, tone)->cycle_rank"]
type SymRankVec = Annotated[Mat[DT.Cycle], "(dir, cycle_rank)->tone"]
type SymCycleMatrix = Annotated[Cube[DT.Cycle], "(direction, starting_tone, tone) -> rank"]
type SymRankMatrix = Annotated[Cube[DT.Cycle], "(direction, starting_tone, rank) -> tone"]
