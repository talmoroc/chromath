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
type Chroma = Annotated[Vec[DT.Chr], "(MS.tones,) bool array representing a mask over the tones"]

type ScaleChroma = Annotated[Chroma, "Chroma with a fixed number of True values"]

type ChromaMatrix = Annotated[Mat[DT.Chr], "Squatre matrix of chroma"]

## Degree Types
type Degrees = Annotated[Vec[DT.Deg], "Array for which Index = degree, Value = tone"]

type ScaleDegrees = Annotated[Degrees, "DegreeVec with MS.degrees values"]

type Interval = Annotated[Mat[DT.Deg], ""]


## Cycle Types
type CycleVec = Annotated[
    Vec[DT.Cycle],
    "Vector representing a cycle from a starting point. vec[tone] = corresponding rank in the cycle or -1 if the tone is missing from the cycle",
]

type RankVec = Annotated[Vec[DT.Cycle], "Inverse of a CycleVec: vec[rank] = corresponding tone"]

type CycleMatrix = Annotated[Mat[DT.Cycle], "Matrix of CycleVecs: (starting_tone, tone) -> rank"]

type RankMatrix = Annotated[Mat[DT.Cycle], "Matrix of RankVecs: (starting_tone, rank) -> tone"]

type SymCycleVec = Annotated[
    Mat[DT.Cycle], "CycleVec with an added dimension for positive/negative direction of cycle"
]

type SymRankVec = Annotated[
    Mat[DT.Cycle], "CycleVec with an added dimension for positive/negative direction of cycle"
]


type SymCycleMatrix = Annotated[
    Cube[DT.Cycle], "CycleMatrix with an added dimension for positive/negative direction of cycle"
]

type SymRankMatrix = Annotated[
    Cube[DT.Cycle], "RankMatrix with an added dimension for positive/negative direction of cycle"
]
