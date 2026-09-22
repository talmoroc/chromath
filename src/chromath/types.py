from typing import Final

import numpy as np
from jaxtyping import Bool, Int8, UInt16
from numpy.typing import NDArray

# NOTE: np.dtypes used to limit memory usage (bool + int8)
# If vectorization becomes a bottleneck, use int32/int64 everywhere


class DT:
    Key: Final = np.uint16  # Chroma as a 12-bit key -> UInt16[...]
    Chroma: Final = np.bool_  # Chroma as a 12-lane view -> Bool[...]
    St: Final = np.int8  # Semitones or rank -> Int8[...]
    Score: Final = np.float32  # -> Float32[...]


# Chroma Types

ChromaBoolArray = Bool[NDArray, "*batch 12"]
ChromaArray = UInt16[np.ndarray, "*batch"]


# Degree Types
SemitonesArray = Int8[np.ndarray, "*batch 2 intervals"]

# Interpreted Interval Types (degree + semitone + alteration)
InterpretedIntervalArray = Int8[np.ndarray, "*batch 3 intervals"]

# Scale Lookup Types (maps each semitone to possible degree/alteration pairs)
ScaleLookupArray = Int8[np.ndarray, "*batch 12 2 2"]
ScaleLookupCounts = Int8[np.ndarray, "*batch 12"]
