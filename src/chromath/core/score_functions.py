"""Pluggable scoring functions for the chord solver.

Each score function takes a :class:`ScoringContext` and returns a
``(raw, normalized)`` tuple of arrays.  Normalized values are in [0, 1]
where 0 = best.

To add a custom score, write a function matching this protocol and wrap it
in a :class:`ScoreFn`::

    def my_score(ctx: ScoringContext) -> tuple[np.ndarray, np.ndarray]:
        raw = ...  # shape (n_scales, n_cand) or (n_cand,)
        norm = ... # normalize to [0, 1], 0=best
        return raw, norm

    MY_SCORE = ScoreFn("my_score", my_score, default_weight=1.0)

Then pass ``score_fns=default_score_fns() + [MY_SCORE]`` to
:func:`solver.solve`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp as _scipy_logsumexp

from ..constants import DefaultMusicSystem as MS
from ..types import DT, ChromaArray
from . import freq_ops
from . import interval_ops as interval

_SENTINEL = np.int8(np.iinfo(np.int8).min)  # -128


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class ScoringContext:
    """All shared data available to scoring functions.

    Attributes:
        candidate_interps: shape ``(n_scales, n_cand, max_notes, 3)``.
        start_interp: shape ``(n_scales, 1, max_notes, 3)``.
        end_interp: shape ``(n_scales, 1, max_notes, 3)``.
        candidate_semitones: shape ``(n_cand, max_notes)``.
        start_semitones: shape ``(1, max_notes)``.
        end_semitones: shape ``(1, max_notes)``.
        candidate_lengths: shape ``(n_cand,)``.
        candidate_dissonance: shape ``(n_cand,)``.
        candidate_roots: shape ``(n_cand,)``.
        candidate_tertian: shape ``(n_cand,)``.
        start_root: Root pitch-class of the start chord.
        end_root: Root pitch-class of the end chord.
        start_length: Number of notes in the start chord.
        end_length: Number of notes in the end chord.
        max_notes: Maximum chord cardinality (used for normalization).
    """

    candidate_interps: np.ndarray
    start_interp: np.ndarray
    end_interp: np.ndarray
    candidate_semitones: np.ndarray
    start_semitones: np.ndarray
    end_semitones: np.ndarray
    candidate_lengths: np.ndarray
    candidate_dissonance: np.ndarray
    candidate_roots: np.ndarray
    candidate_tertian: np.ndarray
    start_root: int
    end_root: int
    start_length: int
    end_length: int
    max_notes: int


@dataclass
class ScoreFn:
    """A pluggable scoring function.

    Attributes:
        name: Unique key used in weights dict and DataFrame columns.
        fn: ``Callable(ScoringContext) → (raw, normalized)``.
            Both arrays have shape ``(n_scales, n_cand)`` for
            scale-dependent scores or ``(n_cand,)`` for scale-independent.
            *normalized* is [0, 1] where 0 = best.
        default_weight: Weight used when the weights dict omits this key.
    """

    name: str
    fn: Callable[[ScoringContext], tuple[np.ndarray, np.ndarray]]
    default_weight: float = 0.0


# ---------------------------------------------------------------------------
# Batch utility functions
# ---------------------------------------------------------------------------

def batch_alteration_cost(interpretations: np.ndarray) -> np.ndarray:
    """Sum of |alteration| per chord.

    Args:
        interpretations: shape ``(..., max_notes, 3)``.

    Returns:
        int array, shape ``(...)``.
    """
    alt = interpretations[..., 2]
    mask = alt != _SENTINEL
    return np.sum(np.abs(alt) * mask, axis=-1)


def batch_resolution_score(
    interp_from: np.ndarray,
    interp_to: np.ndarray,
) -> np.ndarray:
    """Vectorized resolution scoring.

    For each chord in *interp_from*, score how well its alterations
    resolve into the corresponding chord in *interp_to*.

    Returns:
        float array, shape ``(...)``. Score in [0, 1].
    """
    from_alt = interp_from[..., 2]
    from_st = interp_from[..., 1].astype(np.int16)
    to_st = interp_to[..., 1].astype(np.int16)

    has_alt = (from_alt != _SENTINEL) & (from_alt != 0)
    n_altered = has_alt.sum(axis=-1).astype(np.float64)

    to_valid_mask = interp_to[..., 1] != _SENTINEL

    diff = to_st[..., np.newaxis, :] - from_st[..., :, np.newaxis]
    diff = diff % MS.tones
    diff = np.where(diff > MS.tones // 2, diff - MS.tones, diff)

    abs_diff = np.abs(diff)
    abs_diff = np.where(to_valid_mask[..., np.newaxis, :], abs_diff, 999)

    closest_idx = np.argmin(abs_diff, axis=-1)
    closest_diff = np.take_along_axis(
        diff, closest_idx[..., np.newaxis], axis=-1
    ).squeeze(-1)

    resolves = (
        (np.sign(closest_diff) == np.sign(from_alt))
        & (np.abs(closest_diff) <= 2)
    )
    resolves = resolves & has_alt

    good = resolves.sum(axis=-1).astype(np.float64)
    n_safe = np.maximum(n_altered, 1.0)
    return np.where(n_altered > 0, good / n_safe, 1.0)


def batch_voice_leading_cost(
    from_st: np.ndarray,
    to_st: np.ndarray,
) -> np.ndarray:
    """Minimum total semitone movement between two chords.

    Returns:
        float array, shape ``(...)``. Lower = smoother voice leading.
    """
    f = from_st.astype(np.int16)
    t = to_st.astype(np.int16)

    f_valid = from_st != _SENTINEL
    t_valid = to_st != _SENTINEL

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
    """Combined CoF + semitone distance from candidate roots to a reference.

    Returns ``cof_dist + 0.5 * semitone_dist``.
    """
    n = len(candidate_roots)
    cof = np.empty(n, dtype=np.float64)
    semi = np.empty(n, dtype=np.float64)
    for i in range(n):
        r = int(candidate_roots[i])
        cof[i] = interval._cof_distance(r, reference_root)
        semi[i] = _semitone_distance(r, reference_root)
    return cof + 0.5 * semi


_TRIAD_PATTERNS = np.array([
    [0, 4, 7],   # major
    [0, 3, 7],   # minor
    [0, 3, 6],   # diminished
    [0, 4, 8],   # augmented
], dtype=np.int16)


def batch_tertian_score(chromas: ChromaArray) -> np.ndarray:
    """Check whether each chord contains a standard triad.

    Returns:
        float64 array, shape ``(n,)``. 1.0 = contains a triad, 0.0 = not.
    """
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


def _tertian_chain(
    candidate: int, notes: set[int], max_gaps: int = 1
) -> tuple[int, int]:
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


def estimate_roots(chromas: ChromaArray) -> np.ndarray:
    """Estimate the most likely root pitch-class for each chord."""
    n = chromas.shape[0]
    roots = np.zeros(n, dtype=DT.St)

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
    chromas: ChromaArray,
    roots: np.ndarray | None = None,
    method: str = "smoothed",
) -> np.ndarray:
    """Compute dissonance for each chord.

    Returns:
        float64 array, shape ``(n,)``. Higher = more dissonant.
    """
    n = chromas.shape[0]
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        st = [int(x) for x in np.flatnonzero(chromas[i])]
        if roots is not None:
            st = root_position_semitones(st, int(roots[i]))
        result[i] = freq_ops.dissonance(st, method=method)
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
        vl_in = batch_voice_leading_cost(
            ctx.start_semitones, ctx.candidate_semitones
        )
        vl_out = batch_voice_leading_cost(
            ctx.candidate_semitones, ctx.end_semitones
        )
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
    raw = (
        (ctx.candidate_roots == ctx.start_root)
        | (ctx.candidate_roots == ctx.end_root)
    ).astype(np.float64)
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
    """Build the default set of scoring functions.

    Args:
        min_voice_leading: Minimum desired voice-leading distance.
        max_voice_leading: Maximum desired voice-leading distance.
    """
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
