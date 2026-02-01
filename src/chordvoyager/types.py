import numpy as np
from numpy.typing import NDArray


from typing import Annotated, NewType, Final


class DT:
    Chr: Final = np.bool_
    Deg: Final = np.int8
    Cycle: Final = np.int8

type NDArrayBool = NDArray[np.bool_]
type NDArrayInt8 = NDArray[np.int8]
type NDArrayInt64 = NDArray[np.int64]

type ChromaVec = Annotated[
    np.ndarray[tuple[int], np.dtype[DT.Chr]],
    "(MusicSystem.tones,) bool array representing a mask over the tones",
]

type ScaleChromaVec = Annotated[
    NewType["ScaleChromaVec", ChromaVec], "ChromaVec with a fixed number of True values"
]

type DegreeVec = Annotated[
    np.ndarray[tuple[int], np.dtype[DT.Deg]],
    "(MusicSystem.degrees,) int8 array. Index = degree, Value = tone",
]

type IntervalVec = Annotated[np.ndarray[tuple[int, int], np.dtype[DT.Deg]], ""]

type CycleVec = Annotated[
    np.ndarray[tuple[int], np.dtype[DT.Cycle]],
    "(MusicSystem.tones,) int8 array. Index = tone, Value = Position in the cycle, -1 if the tone is not in the cycle",
]

type CycleMatrix = Annotated[
    np.ndarray[tuple[int, int, int], np.dtype[DT.Cycle]],
    "Matrix representing a cycle",
]
type RankMatrix = CycleMatrix
