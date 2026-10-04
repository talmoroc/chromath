from typing import Final, NewType

import numpy as np
from numpy.typing import NDArray


class DT:
    Key = np.uint16
    Bool = np.bool_
    Note = np.int8
    Score = np.float32
    Int = np.int_


SENTINEL: Final = DT.Note(np.iinfo(DT.Note).min)  # -128: fills unused slots of padded DT.Note arrays


# Identifier of a chroma
ChromaKey = NewType("ChromaKey", DT.Key)  # hashable representation, held by classes
ChromaVec = NDArray[DT.Bool]  # (MS.tones,)  # Encoding as a binary vector, useful for vectorized/visible manipulation

# Special case: chroma with one and only one note. Can be a key, a vector or a single index.
# These subtypes are used whenever the input of a function needs to be a note or list of notes and not complex chromas.
NoteKey = NewType("NoteKey", DT.Key)  # power of 2
NoteVec = NDArray[DT.Bool]  # (MS.tones,)  # one-hot vector

# Batches
ChromaKeyArray = NDArray[DT.Key]  # (...)
NoteKeyArray = NDArray[DT.Key]  # (...)
ChromaVecArray = NDArray[DT.Bool]  # (..., MS.tones)
NoteVecArray = NDArray[DT.Bool]  # (..., MS.tones)

# Then we can define the Chroma as an array of NoteIndex:
NoteIndex = NewType("NoteIndex", DT.Note)  # Semitones representation of singleton
NoteIndexArray = NDArray[DT.Note]  # (...) shifts, steps, ranks, does not represent the members of a chroma.
ChromaMembersArray = NDArray[DT.Note]  # (..., width) Batched members, ascending, then SENTINEL. Fixed given width of MS.max_chroma_members

# Sequences: a batch whose last axis is time or ordering. I will need this later.
# ChromaSeq = NDArray[DT.Key]  # (..., order)
# NoteSeq = NDArray[DT.Note]  # (..., order)

# Scores. Can be int (ranks in cycles, ordering in general) or float (DT.Score)
ScoreArray = NDArray[DT.Score]  # (...)
IntArray = NDArray[DT.Int]  # (...)

# TODO: clean these types
# Degree Types (degree + semitone)
DegreeArray = NDArray[DT.Note]  # (..., 2, intervals)
# Interpreted Interval Types (degree + semitone + alteration)
InterpretedDegreeArray = NDArray[DT.Note]  # (..., 3, intervals)

# Scale Lookup Types (maps each semitone to possible degree/alteration pairs)
ScaleLookupArray = NDArray[np.integer] | list[int]  # (..., 12, 2, 2)
ScaleLookupCounts = NDArray[np.integer] | list[int]  # (..., 12)
