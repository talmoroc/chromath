from __future__ import annotations

from collections.abc import Callable
from itertools import product as cartesian_product

import numpy as np

from ..constants import DefaultMusicSystem as MS
from ..types import (
    DT,
    InterpretedIntervalArray,
    IntervalArray,
    ScaleIntervalArray,
    ScaleLookupArray,
    ScaleLookupCounts,
)
from . import validation as val


def to_int(v: IntervalArray) -> int:
    """
    Converts an IntervalArray to its bitwise integer representation based on semitones.
    """
    semitones = np.atleast_2d(v)[..., 1]
    chroma = np.zeros(MS.tones, dtype=int)
    chroma[semitones % MS.tones] = 1
    return int(np.dot(chroma, MS.powers))


def from_bits(bitwise_repr: int) -> IntervalArray:
    """
    Reconstructs an IntervalArray from a bitwise integer representation.
    Note: Degrees are set to 0 as they cannot be inferred from bits alone.
    """
    if bitwise_repr >= MS.max_int_repr:
        raise ValueError(f"Bits should be < 2**{MS.tones}, got {bitwise_repr}")

    semitones = np.where((bitwise_repr >> np.arange(MS.tones)) & 1)[0]
    res = np.zeros((len(semitones), 2), dtype=DT.St)
    res[:, 1] = semitones
    return res


def isin(v: IntervalArray, e: IntervalArray) -> bool:
    """
    Checks if a specific interval [degree, semitone] exists within a collection of intervals.
    """
    return bool(np.any(np.all(e == v, axis=-1)))


def shift(v: IntervalArray, n: int) -> IntervalArray:
    """
    Transposes the interval(s) by n semitones.
    """
    res = np.array(v, copy=True)
    res[..., 1] = (res[..., 1] + n) % MS.tones
    return res


def from_scale(scale_semitones: list[int] | np.ndarray) -> ScaleIntervalArray:
    """
    Converts a scale (list of semitones) to a ScaleIntervalArray.
    Each interval is represented as (degree, semitone).
    Degrees are 1-indexed (1 through 7 for heptatonic scales).
    """
    if len(scale_semitones) != MS.degrees:
        raise ValueError(f"Scale must have {MS.degrees} degrees, got {len(scale_semitones)}")

    degrees = np.arange(1, MS.degrees + 1, dtype=DT.St)
    semitones = np.array(scale_semitones, dtype=DT.St)
    res = np.column_stack((degrees, semitones))
    return val.validate_scale_interval_array(res)


def from_chord(chord_semitones: list[int] | np.ndarray) -> IntervalArray:
    """
    Converts a chord (list of semitones relative to root) to an IntervalArray.
    Degrees are inferred from sorted position in the chord.
    """
    semitones = np.array(chord_semitones, dtype=DT.St)
    semitones = np.sort(semitones)
    degrees = np.arange(len(semitones), dtype=DT.St)
    res = np.column_stack((degrees, semitones))
    return val.validate_interval_array(res)


def semitone_distance(interval1: IntervalArray, interval2: IntervalArray) -> int:
    """
    Computes the semitone distance between two intervals.
    Returns the absolute difference in semitones.
    """
    st1 = np.atleast_1d(interval1)[..., 1]
    st2 = np.atleast_1d(interval2)[..., 1]
    return int(np.abs(st2 - st1) % MS.tones)


def scale_distance(scale1_semitones: list[int] | np.ndarray, scale2_semitones: list[int] | np.ndarray) -> int:
    """
    Computes the total semitone distance between two scales.
    Returns the sum of absolute differences for each degree.
    """
    scale1 = np.array(scale1_semitones, dtype=DT.St)
    scale2 = np.array(scale2_semitones, dtype=DT.St)

    if len(scale1) != len(scale2):
        raise ValueError(f"Scales must have same length, got {len(scale1)} and {len(scale2)}")

    return int(np.sum(np.abs((scale2 - scale1) % MS.tones)))


def common_intervals(interval_array1: IntervalArray, interval_array2: IntervalArray) -> IntervalArray:
    """
    Finds intervals that appear in both arrays (by semitone value).
    """
    st1 = interval_array1[..., 1]
    st2 = interval_array2[..., 1]

    common_st = np.intersect1d(st1, st2)
    degrees = np.arange(len(common_st), dtype=DT.St)
    res = np.column_stack((degrees, common_st))
    return val.validate_interval_array(res)


def invert(v: IntervalArray, pivot: int = 0) -> IntervalArray:
    """
    Musical inversion of intervals around a pivot (default 0 semitones).
    """
    res = np.array(v, copy=True)
    res[..., 1] = (pivot - res[..., 1]) % MS.tones
    return res


# ── Scale Lookup & Interpretation ──────────────────────────────────────


_SENTINEL = np.int8(np.iinfo(np.int8).min)  # -128, must not collide with valid alt values


def build_scale_lookup(
    scale_semitones: list[int] | np.ndarray,
) -> tuple[ScaleLookupArray, ScaleLookupCounts]:
    """
    Build a fixed-shape lookup table mapping every semitone (0-11) to its
    possible (degree_0idx, alteration) interpretations within a scale.

    Args:
        scale_semitones: sorted semitone positions of the scale (length = MS.degrees).

    Returns:
        lookup : int8 array, shape (12, 2, 2).
            lookup[st, k] = (degree, alteration) for the k-th interpretation.
            Unused slots are filled with _SENTINEL.
        counts : int8 array, shape (12,).
            Number of valid interpretations per semitone (1 or 2).
    """
    sc = np.sort(np.asarray(scale_semitones, dtype=DT.St))
    n_deg = len(sc)

    lookup = np.full((MS.tones, 2, 2), _SENTINEL, dtype=DT.St)
    counts = np.zeros(MS.tones, dtype=DT.St)

    sc_set = set(int(s) for s in sc)

    for st in range(MS.tones):
        k = 0
        if st in sc_set:
            # Diatonic note: exact match, alteration = 0
            deg = int(np.searchsorted(sc, st))
            lookup[st, 0] = (deg, 0)
            k = 1
        else:
            # Chromatic note: try sharp of degree below, flat of degree above
            pos = int(np.searchsorted(sc, st))  # sc[pos-1] < st < sc[pos % n]

            # Sharp of degree below: degree = pos-1, alt = st - sc[pos-1]
            deg_below = (pos - 1) % n_deg
            alt_sharp = (st - int(sc[deg_below])) % MS.tones
            if alt_sharp > MS.tones // 2:
                alt_sharp -= MS.tones
            if alt_sharp == 1:
                lookup[st, k] = (deg_below, 1)
                k += 1

            # Flat of degree above: degree = pos % n, alt = st - sc[pos % n]
            deg_above = pos % n_deg
            alt_flat = (st - int(sc[deg_above])) % MS.tones
            if alt_flat > MS.tones // 2:
                alt_flat -= MS.tones
            if alt_flat == -1:
                lookup[st, k] = (deg_above, -1)
                k += 1

        counts[st] = k

    return val.validate_scale_lookup(lookup), val.validate_scale_lookup_counts(counts)


def build_all_scale_lookups(
    scale_semitones: list[int] | np.ndarray,
) -> tuple[ScaleLookupArray, ScaleLookupCounts]:
    """
    Build lookup tables for all 12 transpositions of a scale.

    Returns:
        lookups : int8 array, shape (12, 12, 2, 2)
        counts  : int8 array, shape (12, 12)
    """
    sc = np.asarray(scale_semitones, dtype=DT.St)
    all_lookups = np.empty((MS.tones, MS.tones, 2, 2), dtype=DT.St)
    all_counts = np.empty((MS.tones, MS.tones), dtype=DT.St)
    for root in range(MS.tones):
        transposed = (sc + root) % MS.tones
        lk, ct = build_scale_lookup(np.sort(transposed))
        all_lookups[root] = lk
        all_counts[root] = ct
    return all_lookups, all_counts


def _cof_distance(a: int, b: int) -> int:
    """Minimum number of fifths (±7 semitones) separating two pitch classes."""
    d = (a - b) * 7 % MS.tones  # map semitone diff to CoF steps
    return min(d, MS.tones - d)


def _interpret_single_lookup(
    chord_semitones: np.ndarray,
    lookup: ScaleLookupArray,
) -> InterpretedIntervalArray:
    """Fast-path: take slot-0 interpretation from a single lookup.

    Used internally by the solver for per-scale scoring.
    Sentinel values are passed through unchanged.
    """
    st = np.asarray(chord_semitones, dtype=DT.St)
    original_shape = st.shape

    mask = st >= 0
    safe_st = np.where(mask, st, 0)

    deg_alt = lookup[safe_st, 0]

    result = np.full((*original_shape, 3), _SENTINEL, dtype=DT.St)
    result[..., 0] = np.where(mask, deg_alt[..., 0], _SENTINEL)
    result[..., 1] = np.where(mask, st, _SENTINEL)
    result[..., 2] = np.where(mask, deg_alt[..., 1], _SENTINEL)

    return result


def interpret_canonical(
    chord_semitones: np.ndarray,
    scale_semitones_list: list[np.ndarray],
    reference_roots: int | list[int] = 0,
    distance_fn: Callable[[int, int], float] | None = None,
) -> InterpretedIntervalArray:
    """Tonality-aware canonical interpretation.

    For each chromatic note in the chord, all 12 transpositions of every
    supplied scale type cast a weighted vote for sharp vs flat spelling.
    The weight is ``1 / (1 + distance)`` where *distance* is the
    **minimum** circle-of-fifths distance between the tonality root and
    any of the *reference_roots* (configurable via *distance_fn*).

    In a progression context (start → ? → end), pass both the start and
    end chord roots so that the middle chord's spelling is influenced by
    both the backward-facing (where we came from) and forward-facing
    (where we are going) tonal contexts.

    Diatonic notes (alteration = 0) are unambiguous and always resolved
    identically regardless of tonality.

    Args:
        chord_semitones: int8 array, shape (n_notes,). Values 0-11.
        scale_semitones_list: list of scale definitions (e.g.
            ``[IONIAN_SEMITONES]``).  All 12 transpositions are built for
            each.
        reference_roots: one or more root pitch-classes (0-11) from which
            CoF distance is measured.  Can be a single int or a list.
            When multiple roots are given the minimum distance to any of
            them is used, giving equal influence to backward and forward
            tonal context.
        distance_fn: ``(root, reference_root) -> numeric`` used for
            weighting.  Defaults to :func:`_cof_distance`.

    Returns:
        InterpretedIntervalArray, shape (n_notes, 3):
        ``(degree, semitone, alteration)``.
    """
    if distance_fn is None:
        distance_fn = _cof_distance

    # Normalise to list
    ref_list = [int(reference_roots)] if isinstance(reference_roots, (int, np.integer)) else [int(r) for r in reference_roots]

    st = np.asarray(chord_semitones, dtype=DT.St).ravel()
    n_notes = len(st)

    # Build all lookups
    all_lookups: list[tuple[ScaleLookupArray, ScaleLookupCounts]] = []
    for sc in scale_semitones_list:
        all_lookups.append(build_all_scale_lookups(sc))

    # Strategy: vote at the CHORD level, not note by note.
    # A tonality (scale transposition) votes only if the WHOLE chord is diatonic
    # in it (every note has alteration == 0).  The tonality's position on the
    # circle of fifths implies the conventional enharmonic spelling:
    #   CoF position = (root * 7) % 12
    #   1-6  (G D A E B F#)  → sharp-side → votes for sharps
    #   7-11 (Db Ab Eb Bb F) → flat-side  → votes for flats
    #   0    (C)              → neutral, no vote
    # Weight = 1 / (1 + dist) where dist = min CoF distance to any reference root.
    # The chord-level vote decides the single enharmonic spelling applied to all
    # notes; degrees are always read from the original (root=0) lookup.

    sharp_weight: float = 0.0
    flat_weight: float = 0.0
    voting_keys: list[tuple[int, str, float]] = []  # (root, side, weight) for debug

    # Original (root=0) lookup for degree extraction
    original_lookups, original_counts = all_lookups[0]
    original_lk = original_lookups[0]  # (12, 2, 2)
    original_ct = original_counts[0]  # (12,)

    valid_st = [int(s) for s in st if s >= 0]

    for lookups, counts in all_lookups:
        for root in range(MS.tones):
            lk = lookups[root]  # (12, 2, 2)
            ct = counts[root]  # (12,)

            # Check if the whole chord is diatonic in this transposition
            chord_is_diatonic = all(any(int(lk[s, k, 1]) == 0 for k in range(int(ct[s]))) for s in valid_st)
            if not chord_is_diatonic:
                continue

            dist = float(min(distance_fn(root, r) for r in ref_list))
            w = 1.0 / (1.0 + dist)
            cof_pos = (root * 7) % MS.tones

            if 0 < cof_pos <= 6:
                sharp_weight += w
                voting_keys.append((root, "sharp", w))
            elif cof_pos > 6:
                flat_weight += w
                voting_keys.append((root, "flat", w))
            # cof_pos == 0 (C major): neutral, no vote

    # Resolve ambiguity — only warn when at least one note has two possible
    # spellings in the original lookup (i.e. is chromatic / non-diatonic).
    has_chromatic = any(int(original_ct[s]) > 1 for s in valid_st)
    if has_chromatic and sharp_weight == flat_weight:
        note_names_dbg = ["C", "C#/Db", "D", "D#/Eb", "E", "F", "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B"]
        chord_str = " ".join(note_names_dbg[s] for s in valid_st)
        ref_str = " ".join(note_names_dbg[r] for r in ref_list)
        print(f"[interpret_canonical] AMBIGUOUS spelling for chord [{chord_str}] ref={ref_str}  sharp_w={sharp_weight:.3f} flat_w={flat_weight:.3f}")
        for root, side, w in sorted(voting_keys, key=lambda x: -x[2]):
            note_names_dbg2 = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
            print(f"  root={note_names_dbg2[root]:<3} side={side:<5} w={w:.3f}")
        # Default to flat on a tie (conventional)

    use_sharp = sharp_weight > flat_weight

    # Build result using original (root=0) degrees and the chosen spelling
    result = np.full((n_notes, 3), _SENTINEL, dtype=DT.St)
    for ni in range(n_notes):
        s = int(st[ni])
        if s < 0:
            continue
        nc = int(original_ct[s])
        if nc == 0:
            continue

        chosen_deg: int | None = None
        chosen_alt: int | None = None

        if nc == 1:
            # Diatonic in original scale — unambiguous
            chosen_deg = int(original_lk[s, 0, 0])
            chosen_alt = int(original_lk[s, 0, 1])
        else:
            # Two interpretations: find (sharp, alt>0) and (flat, alt<0)
            for k in range(nc):
                alt = int(original_lk[s, k, 1])
                if (use_sharp and alt > 0) or (not use_sharp and alt < 0):
                    chosen_deg = int(original_lk[s, k, 0])
                    chosen_alt = alt
                    break
            # Fallback: take first interpretation
            if chosen_deg is None:
                chosen_deg = int(original_lk[s, 0, 0])
                chosen_alt = int(original_lk[s, 0, 1])

        result[ni] = (chosen_deg, s, int(chosen_alt) if chosen_alt is not None else 0)

    return result


def interpret_all(
    chord_semitones: np.ndarray,
    lookup: ScaleLookupArray,
    counts: ScaleLookupCounts,
) -> list[InterpretedIntervalArray]:
    """
    Rich-path interpretation: enumerate all valid interpretation combinations for
    a chord. Only used for analysis of specific candidates.

    Args:
        chord_semitones: 1-D int8 array of semitone values.
        lookup: shape (12, 2, 2).
        counts: shape (12,).

    Returns:
        List of InterpretedIntervalArray, each shape (n_notes, 3).
    """
    st = np.asarray(chord_semitones, dtype=DT.St).ravel()
    valid = st >= 0
    active_st = st[valid]

    # Build per-note list of possible (degree, alteration) tuples
    per_note_options: list[list[tuple[int, int]]] = []
    for s in active_st:
        n = int(counts[s])
        options = [(int(lookup[s, k, 0]), int(lookup[s, k, 1])) for k in range(n)]
        per_note_options.append(options)

    results: list[InterpretedIntervalArray] = []
    for combo in cartesian_product(*per_note_options):
        interp = np.full((len(st), 3), _SENTINEL, dtype=DT.St)
        for i, (idx, s) in enumerate(zip(np.where(valid)[0], active_st, strict=True)):
            deg, alt = combo[i]
            interp[idx] = (deg, s, alt)
        results.append(interp)

    return results


def alteration_cost(interpreted: InterpretedIntervalArray) -> int:
    """
    Sum of |alteration| across all notes. Sentinel values are ignored.
    """
    alt = interpreted[..., 2]
    mask = alt != _SENTINEL
    return int(np.sum(np.abs(alt[mask])))


def resolution_score(
    interpreted_from: InterpretedIntervalArray,
    interpreted_to: InterpretedIntervalArray,
) -> float:
    """
    Score how well alterations in `interpreted_from` resolve into `interpreted_to`.

    Sharp notes (+1) that move up by a semitone and flat notes (-1) that move
    down by a semitone are considered good resolutions. Returns a score in [0, 1]
    where 1 = all alterations resolve correctly.

    Only considers notes with non-zero alteration in the source chord.
    """
    from_alt = interpreted_from[..., 2].ravel()
    from_st = interpreted_from[..., 1].ravel()
    to_st = interpreted_to[..., 1].ravel()

    mask = (from_alt != _SENTINEL) & (from_alt != 0)
    if not np.any(mask):
        return 1.0  # no alterations to resolve

    alts = from_alt[mask]
    src = from_st[mask]

    # For each altered note, find the closest semitone in the target chord
    to_valid = to_st[to_st >= 0]
    if len(to_valid) == 0:
        return 0.0

    # Compute signed distances (mod 12, centered)
    # src[:, None] vs to_valid[None, :]
    diffs = (to_valid[None, :].astype(np.int16) - src[:, None].astype(np.int16)) % MS.tones
    diffs = np.where(diffs > MS.tones // 2, diffs - MS.tones, diffs)

    # For sharps (alt=+1): ideal resolution is +1 (move up by 1 semitone)
    # For flats (alt=-1): ideal resolution is -1 (move down by 1 semitone)
    # Find the closest target note for each altered note
    abs_diffs = np.abs(diffs)
    closest_idx = np.argmin(abs_diffs, axis=1)
    closest_diff = diffs[np.arange(len(src)), closest_idx]

    # Score: +1 if resolution direction matches alteration sign, 0 otherwise
    resolves = (np.sign(closest_diff) == np.sign(alts)) & (np.abs(closest_diff) <= 2)
    return float(np.mean(resolves))
