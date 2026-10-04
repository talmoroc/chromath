"""Score functions for the chord solver; tuning formula, ratios and dissonance."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp

from ..constants import DefaultMusicSystem as MS
from ..types import DT, SENTINEL, ChromaVec
from . import scale

# ---------------------------------------------------------------------------
# Tuning formula, frequency ratios and dissonance
# ---------------------------------------------------------------------------


def midi_freq(midi_value: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440 * (2 ** (float(midi_value - 69) / 12))


# https://arxiv.org/pdf/1306.6458#subsection.3.2
def approximate_frequency_ratio(f1: float, f2: float, d: float = 0.01, divide_by: int = 1) -> tuple[int, int, float]:
    r_init = f2 / f1
    r_min, r_max = (r_init * (1 - d), r_init * (1 + d))
    a_low, b_low = math.floor(r_init), 1
    a_high, b_high = math.ceil(r_init), 1
    a, b = round(r_init), 1
    r = a / b

    while r < r_min or r > r_max:
        r0 = 2 * r_init - r
        if r_init < r:
            a_high, b_high = a, b
            k = math.floor((r0 * b_low - a_low) / (a_high - r0 * b_high))
            a_low, b_low = (a_low + k * a_high, b_low + k * b_high)
        else:
            a_low, b_low = a, b
            k = math.floor((a_high - r0 * b_high) / (r0 * b_low - a_low))
            a_high, b_high = (a_high + k * a_low, b_high + k * b_low)
        a, b = a_low + a_high, b_low + b_high
        r = a / b
    b = b * divide_by
    r = r / divide_by
    return (a, b, r)


def relative_periodicity(semitones: list[int], reference_index: int = 0, d: float = 0.011, sort: bool = True) -> int:
    reference_freq = 2 ** (semitones[reference_index] / 12)
    ratios = [
        approximate_frequency_ratio(
            reference_freq,
            2 ** ((s + 12 * (i < reference_index)) / 12),
            d,
            divide_by=(i < reference_index) + 1,
        )
        for i, s in enumerate(semitones)
    ]
    denominators = [r[1] for r in ratios]
    return int(ratios[0][2] * math.lcm(*denominators))


def smoothed_relative_periodicity(
    semitones: list[int],
    d: float = 0.011,
    sort: bool = True,
    log: bool = True,
    verbose: bool = True,
) -> float:
    current_semitones = sorted(list(set(semitones))) if sort else list(set(semitones))
    n = len(current_semitones)
    relative_periodicities: list[int] = []
    for i in range(len(current_semitones)):
        current_semitones = [s - current_semitones[i] for s in current_semitones]
        relative_periodicities.append(relative_periodicity(current_semitones, reference_index=i, d=d))
    smoothed_periodicity = sum([math.log2(rp) for rp in relative_periodicities]) / n if log else sum(relative_periodicities) / n
    if verbose:
        print(f"Chord {semitones}, dissonance: {smoothed_periodicity}")

    return smoothed_periodicity


def midi_interval_ratio(midi1: int, midi2: int, d: float = 0.011) -> tuple[int, int, float]:
    f1 = midi_freq(midi1)
    f2 = midi_freq(midi2)
    return approximate_frequency_ratio(f1, f2, d)


def relative_dissonance(semitones: list[int], d: float = 0.011, reference_index: int = 0) -> float:
    """Dissonance from the relative periodicity against one reference note, not smoothed."""
    unique = sorted(set(semitones))
    if len(unique) < 1:
        return 0.0
    # Normalize semitones relative to the reference
    normalized = [s - unique[reference_index] for s in unique]
    rp = relative_periodicity(normalized, reference_index=0, d=d)
    return math.log2(rp)


def pairwise_dissonance(semitones: list[int], d: float = 0.011) -> float:
    """Sum of log2(relative periodicity) over every pair of notes."""
    unique = sorted(set(semitones))
    if len(unique) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            rp = relative_periodicity([unique[i], unique[j]], reference_index=0, d=d)
            total += math.log2(rp)
            count += 1
    return total / count if count else 0.0


def max_pairwise_dissonance(semitones: list[int], d: float = 0.011) -> float:
    """Largest pairwise dissonance among all note pairs."""
    unique = sorted(set(semitones))
    if len(unique) < 2:
        return 0.0
    worst = 0.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            rp = relative_periodicity([unique[i], unique[j]], reference_index=0, d=d)
            worst = max(worst, math.log2(rp))
    return worst


# Registry of dissonance methods
DISSONANCE_METHODS: dict[str, type[None]] = {}  # populated below


def dissonance(
    semitones: list[int],
    method: str = "smoothed",
) -> float:
    """Dissonance of a chord by *method*: smoothed, smoothed_raw, relative, raw, pairwise or max_pairwise."""
    if method == "smoothed":
        return smoothed_relative_periodicity(semitones, verbose=False, log=True)
    elif method == "smoothed_raw":
        return smoothed_relative_periodicity(semitones, verbose=False, log=False)
    elif method == "relative":
        return relative_dissonance(semitones)
    elif method == "raw":
        return relative_periodicity(semitones)
    elif method == "pairwise":
        return pairwise_dissonance(semitones)
    elif method == "max_pairwise":
        return max_pairwise_dissonance(semitones)
    else:
        raise ValueError(f"Unknown dissonance method {method!r}. Choose from: 'smoothed', 'smoothed_raw', 'relative', 'pairwise', 'max_pairwise'.")

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ScoringContext:
    """All shared data available to scoring functions."""

    candidate_interps: np.ndarray  # (n_scales, n_cand, max_notes, 3)
    start_interp: np.ndarray  # (n_scales, 1, max_notes, 3)
    end_interp: np.ndarray  # (n_scales, 1, max_notes, 3)
    candidate_semitones: np.ndarray  # (n_cand, max_notes)
    start_semitones: np.ndarray  # (1, max_notes)
    end_semitones: np.ndarray  # (1, max_notes)
    candidate_lengths: np.ndarray  # (n_cand,)
    candidate_dissonance: np.ndarray  # (n_cand,)
    candidate_roots: np.ndarray  # (n_cand,)
    candidate_tertian: np.ndarray  # (n_cand,)
    start_root: int
    end_root: int
    start_length: int
    end_length: int
    max_notes: int


@dataclass
class ScoreFn:
    """A pluggable score: a name, fn(ctx) -> (raw, normalized in [0, 1], 0 = best), and a default weight."""

    name: str
    fn: Callable[[ScoringContext], tuple[np.ndarray, np.ndarray]]
    default_weight: float = 0.0


# ---------------------------------------------------------------------------
# Batch utility functions
# ---------------------------------------------------------------------------


def batch_alteration_cost(interpretations: np.ndarray) -> np.ndarray:
    """Sum of |alteration| per chord: (..., max_notes, 3) -> (...)."""
    alt = interpretations[..., 2]
    mask = alt != SENTINEL
    return np.sum(np.abs(alt) * mask, axis=-1)


def batch_resolution_score(
    interp_from: np.ndarray,
    interp_to: np.ndarray,
) -> np.ndarray:
    """Share of the alterations of *interp_from* that resolve into *interp_to*, in [0, 1], shape (...)."""
    from_alt = interp_from[..., 2]
    from_st = interp_from[..., 1].astype(np.int16)
    to_st = interp_to[..., 1].astype(np.int16)

    has_alt = (from_alt != SENTINEL) & (from_alt != 0)
    n_altered = has_alt.sum(axis=-1).astype(np.float64)

    to_valid_mask = interp_to[..., 1] != SENTINEL

    diff = to_st[..., np.newaxis, :] - from_st[..., :, np.newaxis]
    diff = diff % MS.tones
    diff = np.where(diff > MS.tones // 2, diff - MS.tones, diff)

    abs_diff = np.abs(diff)
    abs_diff = np.where(to_valid_mask[..., np.newaxis, :], abs_diff, 999)

    closest_idx = np.argmin(abs_diff, axis=-1)
    closest_diff = np.take_along_axis(diff, closest_idx[..., np.newaxis], axis=-1).squeeze(-1)

    resolves = (np.sign(closest_diff) == np.sign(from_alt)) & (np.abs(closest_diff) <= 2)
    resolves = resolves & has_alt

    good = resolves.sum(axis=-1).astype(np.float64)
    n_safe = np.maximum(n_altered, 1.0)
    return np.where(n_altered > 0, good / n_safe, 1.0)


def batch_voice_leading_cost(
    from_st: np.ndarray,
    to_st: np.ndarray,
) -> np.ndarray:
    """Sum of nearest-note semitone movements between two chords, shape (...)."""
    f = from_st.astype(np.int16)
    t = to_st.astype(np.int16)

    f_valid = from_st != SENTINEL
    t_valid = to_st != SENTINEL

    diff = t[..., np.newaxis, :] - f[..., :, np.newaxis]
    diff = diff % MS.tones
    diff = np.where(diff > MS.tones // 2, MS.tones - diff, diff)
    diff = np.where(t_valid[..., np.newaxis, :], diff, 999)

    min_dist = np.min(diff, axis=-1)
    min_dist = np.where(f_valid, min_dist, 0)
    return min_dist.sum(axis=-1).astype(np.float64)


def _semitone_distance(a: int, b: int) -> int:
    """Minimum semitone distance between two pitch-classes (0-6)."""
    d = abs(a - b) % MS.tones
    return min(d, MS.tones - d)


def batch_root_distance(
    candidate_roots: np.ndarray,
    reference_root: int,
) -> np.ndarray:
    """Circle-of-fifths distance plus half the semitone distance from candidate roots to a reference."""
    n = len(candidate_roots)
    cof = np.empty(n, dtype=np.float64)
    semi = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = int(candidate_roots[i])
        cof[i] = scale._cof_distance(r, reference_root)
        semi[i] = _semitone_distance(r, reference_root)
    return cof + 0.5 * semi


_TRIAD_PATTERNS = np.array(
    [
        [0, 4, 7],  # major
        [0, 3, 7],  # minor
        [0, 3, 6],  # diminished
        [0, 4, 8],  # augmented
    ],
    dtype=np.int16,
)


def batch_tertian_score(chromas: ChromaVec) -> np.ndarray:
    """1.0 where the chord contains a standard triad, else 0.0, shape (n,)."""
    n = chromas.shape[0]
    result = np.zeros(n, dtype=np.float64)
    for i in range(n):
        notes = np.flatnonzero(chromas[i])
        for root in notes:
            for pattern in _TRIAD_PATTERNS:
                triad = (root + pattern) % 12
                if np.all(chromas[i, triad]):
                    result[i] = 1.0
                    break
            if result[i] > 0:
                break
    return result


def _tertian_chain(candidate: int, notes: set[int], max_gaps: int = 1) -> tuple[int, int]:
    """Walk a tertian chain from *candidate*, allowing gaps."""
    covered = {candidate}
    visited = {candidate}
    current = candidate
    gaps_used = 0

    for _ in range(6):
        major_third = (current + 4) % 12
        minor_third = (current + 3) % 12

        if major_third in notes and major_third not in visited:
            visited.add(major_third)
            covered.add(major_third)
            current = major_third
        elif minor_third in notes and minor_third not in visited:
            visited.add(minor_third)
            covered.add(minor_third)
            current = minor_third
        elif gaps_used < max_gaps:
            found = False
            for skip in [major_third, minor_third]:
                for next_note in [(skip + 4) % 12, (skip + 3) % 12]:
                    if next_note in notes and next_note not in visited:
                        visited.add(skip)
                        visited.add(next_note)
                        covered.add(next_note)
                        current = next_note
                        gaps_used += 1
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        else:
            break

    return len(covered), len(visited)


def estimate_roots(chromas: ChromaVec) -> np.ndarray:
    """Estimate the most likely root pitch-class for each chord."""
    n = chromas.shape[0]
    roots = np.zeros(n, dtype=DT.Note)

    for i in range(n):
        notes = set(np.flatnonzero(chromas[i]).tolist())
        if not notes:
            continue

        best_score = -1.0
        best_root = min(notes)

        for candidate in notes:
            notes_covered, _ = _tertian_chain(candidate, notes)

            p5 = 3.0 * ((candidate + 7) % 12 in notes)
            m3 = 2.0 * ((candidate + 4) % 12 in notes)
            mi3 = 1.0 * ((candidate + 3) % 12 in notes)
            mi7 = 0.5 * ((candidate + 10) % 12 in notes)
            ma7 = 0.5 * ((candidate + 11) % 12 in notes)

            interval_score = p5 + m3 + mi3 + mi7 + ma7
            score = notes_covered * 10.0 + interval_score
            score -= candidate * 0.01

            if score > best_score:
                best_score = score
                best_root = candidate

        roots[i] = best_root

    return roots


def root_position_semitones(semitones: list[int], root: int) -> list[int]:
    """Reorder semitones so *root* is 0 and others above in ascending order."""
    return sorted((s - root) % 12 for s in semitones)


def batch_dissonance(
    chromas: ChromaVec,
    roots: np.ndarray | None = None,
    method: str = "smoothed",
) -> np.ndarray:
    """Dissonance of each chord, shape (n,). Higher is more dissonant."""
    n = chromas.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        st = [int(x) for x in np.flatnonzero(chromas[i])]
        if roots is not None:
            st = root_position_semitones(st, int(roots[i]))
        result[i] = dissonance(st, method=method)
    return result


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns 0 if range is zero."""
    lo = arr.min()
    hi = arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Individual score function implementations
# ---------------------------------------------------------------------------


def _compute_alt_cost(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Alteration cost: sum of |alteration| per chord. Lower = more diatonic."""
    raw = batch_alteration_cost(ctx.candidate_interps).astype(np.float64)
    norm = np.clip(raw / max(ctx.max_notes, 1), 0, 1)
    return raw, norm


def _compute_res_in(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Resolution: start → candidate. Higher raw = better resolution."""
    start_bc = np.broadcast_to(ctx.start_interp, ctx.candidate_interps.shape)
    raw = batch_resolution_score(start_bc, ctx.candidate_interps)
    norm = 1.0 - raw
    return raw, norm


def _compute_res_out(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Resolution: candidate → end. Higher raw = better resolution."""
    end_bc = np.broadcast_to(ctx.end_interp, ctx.candidate_interps.shape)
    raw = batch_resolution_score(ctx.candidate_interps, end_bc)
    norm = 1.0 - raw
    return raw, norm


def _make_vl_penalty(
    min_vl: float = 1.0,
    max_vl: float = 6.0,
) -> ScoreFn:
    """Factory for voice-leading penalty with configurable range."""

    def compute(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
        vl_in = batch_voice_leading_cost(ctx.start_semitones, ctx.candidate_semitones)
        vl_out = batch_voice_leading_cost(ctx.candidate_semitones, ctx.end_semitones)
        vl_in_pen = np.where(
            vl_in < min_vl,
            min_vl - vl_in,
            np.where(vl_in > max_vl, vl_in - max_vl, 0.0),
        )
        vl_out_pen = np.where(
            vl_out < min_vl,
            min_vl - vl_out,
            np.where(vl_out > max_vl, vl_out - max_vl, 0.0),
        )
        raw = vl_in_pen + vl_out_pen
        norm = raw / (1.0 + raw)
        return raw, norm

    return ScoreFn("vl_penalty", compute)


def _compute_card_penalty(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Cardinality penalty: prefer lengths near start/end chord size."""
    lo = float(min(ctx.start_length, ctx.end_length))
    hi = float(max(ctx.start_length, ctx.end_length) + 1)
    cand_f = ctx.candidate_lengths.astype(np.float64)
    raw = np.where(cand_f < lo, lo - cand_f, np.where(cand_f > hi, cand_f - hi, 0.0))
    norm = np.clip(raw / max(ctx.max_notes, 1), 0, 1)
    return raw, norm


def _compute_dissonance(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Dissonance: uses pre-computed dissonance values."""
    raw = ctx.candidate_dissonance.copy()
    norm = _normalize_minmax(raw)
    return raw, norm


def _compute_root_total(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Root distance: sum of CoF+semitone distances to start and end roots."""
    root_from = batch_root_distance(ctx.candidate_roots, ctx.start_root)
    root_to = batch_root_distance(ctx.candidate_roots, ctx.end_root)
    raw = root_from + root_to
    norm = _normalize_minmax(raw)
    return raw, norm


def _compute_tertian(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Tertian quality: 1.0 if chord contains a standard triad."""
    raw = ctx.candidate_tertian.copy()
    norm = 1.0 - raw
    return raw, norm


def _compute_same_root(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
    """Same root penalty: 1.0 if candidate shares root with start or end."""
    raw = ((ctx.candidate_roots == ctx.start_root) | (ctx.candidate_roots == ctx.end_root)).astype(np.float64)
    norm = raw.copy()
    return raw, norm


# ---------------------------------------------------------------------------
# Default score function set
# ---------------------------------------------------------------------------

ALTERATION_COST = ScoreFn("alt_cost", _compute_alt_cost, default_weight=1.0)
RESOLUTION_IN = ScoreFn("res_in", _compute_res_in, default_weight=1.0)
RESOLUTION_OUT = ScoreFn("res_out", _compute_res_out, default_weight=1.0)
CARDINALITY_PENALTY = ScoreFn("card_penalty", _compute_card_penalty, default_weight=0.3)
DISSONANCE = ScoreFn("dissonance", _compute_dissonance, default_weight=0.2)
ROOT_DISTANCE = ScoreFn("root_total", _compute_root_total, default_weight=0.3)
TERTIAN = ScoreFn("tertian", _compute_tertian, default_weight=0.5)
SAME_ROOT_PENALTY = ScoreFn("same_root", _compute_same_root, default_weight=0.4)


def default_score_fns(
    min_voice_leading: float = 1.0,
    max_voice_leading: float = 6.0,
) -> list[ScoreFn]:
    """The default score functions, for a desired voice-leading range."""
    return [
        ALTERATION_COST,
        RESOLUTION_IN,
        RESOLUTION_OUT,
        _make_vl_penalty(min_voice_leading, max_voice_leading),
        CARDINALITY_PENALTY,
        DISSONANCE,
        ROOT_DISTANCE,
        TERTIAN,
        SAME_ROOT_PENALTY,
    ]


DEFAULT_SCORE_FNS = default_score_fns()


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------


def _aggregate_weighted_sum(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Simple weighted sum. Lower = better."""
    total = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    for name, arr in components.items():
        total += weights.get(name, 0.0) * arr
    return total


def _aggregate_logsumexp(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """LogSumExp aggregation — sensitive to worst individual score."""
    stacked = []
    for name, arr in components.items():
        w = weights.get(name, 0.0)
        if w > 0:
            stacked.append(w * arr)
    if not stacked:
        return np.zeros_like(next(iter(components.values())), dtype=np.float64)
    mat = np.stack(stacked, axis=0)
    return np.asarray(_scipy_logsumexp(mat, axis=0), dtype=DT.Score)


def _aggregate_weighted_product(
    components: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    """Weighted geometric mean (via log). All component values must be > 0."""
    total_weight = 0.0
    log_sum = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    for name, arr in components.items():
        w = weights.get(name, 0.0)
        if w > 0:
            log_sum += w * np.log(arr + 1e-8)
            total_weight += w
    if total_weight == 0:
        return np.zeros_like(log_sum)
    return np.exp(log_sum / total_weight)


AGGREGATION_METHODS = {
    "weighted_sum": _aggregate_weighted_sum,
    "logsumexp": _aggregate_logsumexp,
    "weighted_product": _aggregate_weighted_product,
}
