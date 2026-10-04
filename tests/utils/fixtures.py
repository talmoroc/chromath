"""Named chroma vectors and the full chroma domain.

Data lives inline rather than in JSON so ruff covers it and the column
alignment is protected explicitly. Keep the ``# fmt: off`` block hand-aligned:
the vectors are meant to be scanned as a grid, not read one at a time.
"""

from types import SimpleNamespace

import numpy as np

# fmt: off
SCALES = {
    "C_major":            [1,0,1,0,1,1,0,1,0,1,0,1],
    "C_natural_minor":    [1,0,1,1,0,1,0,1,1,0,1,0],
    "C_harmonic_minor":   [1,0,1,1,0,1,0,1,1,0,0,1],
    "C_melodic_minor":    [1,0,1,1,0,1,0,1,0,1,0,1],
    "C_dorian":           [1,0,1,1,0,1,0,1,0,1,1,0],
    "C_phrygian":         [1,1,0,1,0,1,0,1,1,0,1,0],
    "C_lydian":           [1,0,1,0,1,0,1,1,0,1,0,1],
    "C_mixolydian":       [1,0,1,0,1,1,0,1,0,1,1,0],
    "C_locrian":          [1,1,0,1,0,1,1,0,1,0,1,0],
    "C_whole_tone":       [1,0,1,0,1,0,1,0,1,0,1,0],
    "C_chromatic":        [1,1,1,1,1,1,1,1,1,1,1,1],
    "C_pentatonic_major": [1,0,1,0,1,0,0,1,0,1,0,0],
    "C_pentatonic_minor": [1,0,0,1,0,1,0,1,0,0,1,0],
    "D_major":            [0,1,1,0,1,0,1,1,0,1,0,1],
    "E_major":            [0,1,0,1,1,0,1,0,1,1,0,1],
    "F_major":            [1,0,1,0,1,1,0,1,0,1,1,0],
    "G_major":            [1,0,1,0,1,0,1,1,0,1,0,1],
    "A_major":            [0,1,1,0,1,0,1,0,1,1,0,1],
    "B_major":            [0,1,0,1,1,0,1,0,1,0,1,1],
    "Csharp_major":       [1,1,0,1,0,1,1,0,1,0,1,0],
    "Dsharp_major":       [1,0,1,1,0,1,0,1,1,0,1,0],
    "Fsharp_major":       [0,1,0,1,0,1,1,0,1,0,1,1],
    "Gsharp_major":       [1,1,0,1,0,1,0,1,1,0,1,0],
    "Asharp_major":       [1,0,1,1,0,1,0,1,0,1,1,0],
}

CHORDS = {
    "C_maj":       [1,0,0,0,1,0,0,1,0,0,0,0],
    "C_min":       [1,0,0,1,0,0,0,1,0,0,0,0],
    "C_dim":       [1,0,0,1,0,0,1,0,0,0,0,0],
    "C_aug":       [1,0,0,0,1,0,0,0,1,0,0,0],
    "C_sus2":      [1,0,1,0,0,0,0,1,0,0,0,0],
    "C_sus4":      [1,0,0,0,0,1,0,1,0,0,0,0],
    "C_dom7":      [1,0,0,0,1,0,0,1,0,0,1,0],
    "C_maj7":      [1,0,0,0,1,0,0,1,0,0,0,1],
    "C_min7":      [1,0,0,1,0,0,0,1,0,0,1,0],
    "C_dim7":      [1,0,0,1,0,0,1,0,0,1,0,0],
    "C_half_dim7": [1,0,0,1,0,0,1,0,0,0,1,0],
    "C_min_maj7":  [1,0,0,1,0,0,0,1,0,0,0,1],
    "C_dom9":      [1,0,1,0,1,0,0,1,0,0,1,0],
    "C_add9":      [1,0,1,0,1,0,0,1,0,0,0,0],
    "C_6":         [1,0,0,0,1,0,0,1,0,1,0,0],
    "D_maj":       [0,0,1,0,0,0,1,0,0,1,0,0],
    "D_min":       [0,0,1,0,0,1,0,0,0,1,0,0],
    "E_maj":       [0,0,0,0,1,0,0,0,1,0,0,1],
    "E_min":       [0,0,0,0,1,0,0,1,0,0,0,1],
    "F_maj":       [1,0,0,0,0,1,0,0,0,1,0,0],
    "G_maj":       [0,0,1,0,0,0,0,1,0,0,0,1],
    "A_maj":       [0,1,0,0,1,0,0,0,0,1,0,0],
    "A_min":       [1,0,0,0,1,0,0,0,0,1,0,0],
    "B_maj":       [0,0,0,1,0,0,1,0,0,0,0,1],
    "Csharp_maj":  [0,1,0,0,0,1,0,0,1,0,0,0],
    "Csharp_min":  [0,1,0,0,1,0,0,0,1,0,0,0],
    "Dsharp_maj":  [0,0,0,1,0,0,0,1,0,0,1,0],
    "Dsharp_min":  [0,0,0,1,0,0,1,0,0,0,1,0],
    "Fsharp_maj":  [0,1,0,0,0,0,1,0,0,0,1,0],
    "Fsharp_min":  [0,1,0,0,0,0,1,0,0,1,0,0],
    "Gsharp_maj":  [1,0,0,1,0,0,0,0,1,0,0,0],
    "Gsharp_min":  [0,0,0,1,0,0,0,0,1,0,0,1],
    "Asharp_maj":  [0,0,1,0,0,1,0,0,0,0,1,0],
    "Asharp_min":  [0,1,0,0,0,1,0,0,0,0,1,0],
    "G_dom7":      [0,0,1,0,0,1,0,1,0,0,0,1],
    "D_dom7":      [1,0,1,0,0,0,1,0,0,1,0,0],
}

INTERVALS = {
    "unison":      [1,0,0,0,0,0,0,0,0,0,0,0],
    "minor_2nd":   [1,1,0,0,0,0,0,0,0,0,0,0],
    "major_2nd":   [1,0,1,0,0,0,0,0,0,0,0,0],
    "minor_3rd":   [1,0,0,1,0,0,0,0,0,0,0,0],
    "major_3rd":   [1,0,0,0,1,0,0,0,0,0,0,0],
    "perfect_4th": [1,0,0,0,0,1,0,0,0,0,0,0],
    "tritone":     [1,0,0,0,0,0,1,0,0,0,0,0],
    "perfect_5th": [1,0,0,0,0,0,0,1,0,0,0,0],
    "minor_6th":   [1,0,0,0,0,0,0,0,1,0,0,0],
    "major_6th":   [1,0,0,0,0,0,0,0,0,1,0,0],
    "minor_7th":   [1,0,0,0,0,0,0,0,0,0,1,0],
    "major_7th":   [1,0,0,0,0,0,0,0,0,0,0,1],
}

EDGE_CASES = {
    "empty":         [0,0,0,0,0,0,0,0,0,0,0,0],
    "single_C":      [1,0,0,0,0,0,0,0,0,0,0,0],
    "single_Fsharp": [0,0,0,0,0,0,1,0,0,0,0,0],
    "chromatic":     [1,1,1,1,1,1,1,1,1,1,1,1],
}
# fmt: on


_CATEGORIES = (
    ("scales", SCALES),
    ("chords", CHORDS),
    ("intervals", INTERVALS),
    ("edge_cases", EDGE_CASES),
)


def _to_key(vector: list[int]) -> np.ndarray:
    """Little-endian 12-bit key, built here rather than via chromath.

    An independent oracle: the suite must not check the library against
    itself.
    """
    return np.array(sum(bit << i for i, bit in enumerate(vector)), dtype=np.uint16)


def _namespace(convert) -> SimpleNamespace:
    ns = SimpleNamespace()
    for category, items in _CATEGORIES:
        setattr(ns, category, SimpleNamespace(**{k: convert(v) for k, v in items.items()}))
    return ns


#: Named chromas as twelve-lane boolean views.
fixtures = _namespace(lambda v: np.array(v, dtype=np.bool_))

#: The same chromas as 12-bit keys — the canonical form.
keys = _namespace(_to_key)

# The whole finite domain: 12 pitch classes means 4096 chromas, small enough
# that a law can be checked on every one rather than sampled. The row index of
# ALL_CHROMAS is its own key, so ALL_CHROMAS[k] is the view of key k.
ALL_CHROMAS: np.ndarray = np.array([[(i >> b) & 1 for b in range(12)] for i in range(1 << 12)], dtype=np.bool_)
NONEMPTY_CHROMAS: np.ndarray = ALL_CHROMAS[1:]

ALL_KEYS: np.ndarray = np.arange(1 << 12, dtype=np.uint16)
NONEMPTY_KEYS: np.ndarray = ALL_KEYS[1:]
