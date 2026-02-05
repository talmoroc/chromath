from enum import IntEnum
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DTYPE:
    Chr = np.bool_
    Deg = np.int8
    Cycle = np.int8


type NDArrayBool = NDArray[np.bool_]
type NDArrayInt8 = NDArray[np.int8]
type NDArrayInt64 = NDArray[np.int64]

type ChromaVec = NDArray[DTYPE.Chr]
type ScaleChromaVec = NDArray[DTYPE.Chr]
type DegreeVec = NDArray[DTYPE.Deg]

type CycleMatrix = NDArray[DTYPE.Cycle]
type PositionToToneMatrix = CycleMatrix
type ToneToPositionMatrix = CycleMatrix
type CycleVec = NDArray[DTYPE.Cycle]


class CycleDirection(IntEnum):
    POSITION = 0
    NEGATIVE = 1
