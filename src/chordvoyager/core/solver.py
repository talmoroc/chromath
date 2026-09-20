import numpy as np
import pandas as pd
from itertools import combinations

from ..types import DT, ChromaArray, ScaleLookupArray#, ScaleLookupCounts
from ..constants import DefaultMusicSystem as MS
from . import interval_ops as interval
#  from . import conversion as conv

from .score_functions import (  # noqa: F401
    ScoringContext,
    ScoreFn,
    batch_alteration_cost,
    batch_resolution_score,
    batch_voice_leading_cost,
    batch_root_distance,
    batch_dissonance,
    batch_tertian_score,
    estimate_roots,
    root_position_semitones,
    default_score_fns,
    DEFAULT_SCORE_FNS,
    AGGREGATION_METHODS,
    _normalize_minmax,
    _TRIAD_PATTERNS,
    _semitone_distance,
)
from .score_functions import _SENTINEL


def compute_tonality_weights(
    reference_root: int | None = None,
    distance_fn=None,
) -> np.ndarray:
    """Compute weights for each of 12 tonalities based on circle-of-fifths distance.

    Weights reflect how likely each tonality (scale transposition) is to be the
    "current tonal context". A tonality closer to the reference root gets higher
    weight using a ``1 / (1 + distance)`` formula.

    Args:
        reference_root: Root pitch-class (0-11) to measure distance from.
            If None, returns uniform weights (all 1/12).
        distance_fn: Function ``(root_a, root_b) -> int`` for computing
            circle-of-fifths distance. Defaults to ``interval._cof_distance``.

    Returns:
        np.ndarray shape (12,) with weights summing to 1.0.
    """
    if reference_root is None:
        return np.ones(MS.tones, dtype=np.float64) / MS.tones

    if distance_fn is None:
        distance_fn = interval._cof_distance

    weights = np.zeros(MS.tones, dtype=np.float64)
    for tonality_root in range(MS.tones):
        dist = distance_fn(tonality_root, reference_root)
        weights[tonality_root] = 1.0 / (1.0 + dist)

    # Normalize to sum to 1.0
    weights /= weights.sum()
    return weights


def generate_all_chromas(min_notes: int = 2, max_notes: int = 7) -> ChromaArray:
    """
    Generate all binary subsets of {0..11} with cardinality in [min_notes, max_notes].

    Returns:
        ChromaArray, shape (n_candidates, 12) bool.
    """
    rows: list[np.ndarray] = []
    for k in range(min_notes, max_notes + 1):
        for combo in combinations(range(MS.tones), k):
            row = np.zeros(MS.tones, dtype=DT.Chr)
            row[list(combo)] = True
            rows.append(row)
    return np.array(rows, dtype=DT.Chr)


def chromas_to_padded_semitones(
    chromas: ChromaArray, max_notes: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a batch of chroma vectors to padded semitone arrays.

    Returns:
        semitones: int8 array, shape (n, max_notes). Padded with _SENTINEL.
        lengths:   int8 array, shape (n,). Actual number of notes.
    """
    n = chromas.shape[0]
    semitones = np.full((n, max_notes), _SENTINEL, dtype=DT.St)
    lengths = chromas.sum(axis=-1).astype(DT.St)

    for i in range(n):
        st = np.flatnonzero(chromas[i]).astype(DT.St)
        semitones[i, : len(st)] = st

    return semitones, lengths


def batch_interpret_canonical(
    candidate_semitones: np.ndarray,
    lookups: ScaleLookupArray,
) -> np.ndarray:
    """
    Vectorized canonical interpretation of all candidates against all scale lookups.

    Args:
        candidate_semitones: int8, shape (n_candidates, max_notes).
        lookups: int8, shape (n_scales, 12, 2, 2).

    Returns:
        int8 array, shape (n_scales, n_candidates, max_notes, 3): (degree, semitone, alteration).
    """
    n_scales = lookups.shape[0]
    n_cand, max_notes = candidate_semitones.shape

    mask = candidate_semitones != _SENTINEL  # (n_cand, max_notes)
    safe_st = np.where(mask, candidate_semitones, 0)  # safe for indexing

    # lookups[s, safe_st] → (n_scales, n_cand, max_notes, 2)  [degree, alteration]
    # We use advanced indexing: lookups[:, safe_st, 0] with broadcasting
    deg_alt = lookups[:, safe_st, 0]  # (n_scales, n_cand, max_notes, 2)

    result = np.full((n_scales, n_cand, max_notes, 3), _SENTINEL, dtype=DT.St)

    mask_bc = mask[np.newaxis, ...]  # (1, n_cand, max_notes)

    result[..., 0] = np.where(mask_bc, deg_alt[..., 0], _SENTINEL)  # degree
    result[..., 1] = np.where(
        mask_bc,
        candidate_semitones[np.newaxis, ...],
        _SENTINEL,
    )  # semitone
    result[..., 2] = np.where(mask_bc, deg_alt[..., 1], _SENTINEL)  # alteration

    return result


def score_candidates(
    ctx: ScoringContext,
    score_fns: list[ScoreFn] | None = None,
    weights: dict[str, float] | None = None,
    aggregation: str = "weighted_sum",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Score all candidates for the middle position in start → ? → end.

    Each :class:`ScoreFn` in *score_fns* is evaluated against *ctx*.
    Normalized scores ([0, 1], 0=best) are aggregated using the chosen method.

    Args:
        ctx: Scoring context with all candidate and chord data.
        score_fns: List of scoring functions.  ``None`` → :data:`DEFAULT_SCORE_FNS`.
        weights: dict mapping score name → weight.  Missing keys fall back
            to each :attr:`ScoreFn.default_weight`.
        aggregation: ``"weighted_sum"``, ``"logsumexp"``, or ``"weighted_product"``.

    Returns:
        ``(scores, raw_subscores, norm_subscores)``:
        - *scores*: float array, shape ``(n_scales, n_cand)``. Lower = better.
        - *raw_subscores*: dict of un-normalized arrays.
        - *norm_subscores*: dict of [0,1]-normalized arrays.
    """
    if score_fns is None:
        score_fns = DEFAULT_SCORE_FNS
    if weights is None:
        weights = {}

    raw_subscores: dict[str, np.ndarray] = {}
    norm_subscores: dict[str, np.ndarray] = {}

    for sf in score_fns:
        raw, norm = sf.fn(ctx)
        raw_subscores[sf.name] = raw
        norm_subscores[sf.name] = norm

    # Broadcast scale-independent scores to (n_scales, n_cand)
    n_scales = ctx.candidate_interps.shape[0]
    components: dict[str, np.ndarray] = {}
    for name, arr in norm_subscores.items():
        if arr.ndim == 1:
            components[name] = np.broadcast_to(
                arr[np.newaxis, :], (n_scales, arr.shape[0])
            )
        else:
            components[name] = arr

    # Resolve weights: explicit dict overrides, then ScoreFn defaults
    resolved_weights: dict[str, float] = {}
    for sf in score_fns:
        resolved_weights[sf.name] = weights.get(sf.name, sf.default_weight)

    from .score_functions import AGGREGATION_METHODS as _agg_methods
    agg_fn = _agg_methods.get(aggregation, _agg_methods["weighted_sum"])
    scores = agg_fn(components, resolved_weights)

    return scores, raw_subscores, norm_subscores


def score_candidate_breakdown(
    candidate_idx: int,
    start_interp: np.ndarray,
    candidate_interps: np.ndarray,
    end_interp: np.ndarray,
    candidate_semitones: np.ndarray,
    start_semitones: np.ndarray,
    end_semitones: np.ndarray,
    candidate_length: int,
    candidate_dissonance: float,
    candidate_root: int,
    candidate_tertian: float,
    start_root: int,
    end_root: int,
    start_length: int,
    end_length: int,
    weights: dict[str, float] | None = None,
    scale_idx: int = 0,
    max_notes: int = 7,
    score_fns: list[ScoreFn] | None = None,
) -> dict:
    """Break down the score for a single candidate at a specific scale.

    Runs each :class:`ScoreFn` on a mini context for the single candidate,
    returning raw, normalized, and weighted values.

    Returns:
        ``{"raw": {name: float}, "normalized": {name: float},
        "weighted": {name: float}, "score": float}``
    """
    if score_fns is None:
        score_fns = DEFAULT_SCORE_FNS
    if weights is None:
        weights = {}

    ctx = ScoringContext(
        candidate_interps=candidate_interps[
            scale_idx : scale_idx + 1, candidate_idx : candidate_idx + 1
        ],
        start_interp=start_interp[scale_idx : scale_idx + 1],
        end_interp=end_interp[scale_idx : scale_idx + 1],
        candidate_semitones=candidate_semitones[
            candidate_idx : candidate_idx + 1
        ],
        start_semitones=start_semitones,
        end_semitones=end_semitones,
        candidate_lengths=np.array([candidate_length], dtype=DT.St),
        candidate_dissonance=np.array([candidate_dissonance], dtype=np.float64),
        candidate_roots=np.array([candidate_root], dtype=DT.St),
        candidate_tertian=np.array([candidate_tertian], dtype=np.float64),
        start_root=start_root,
        end_root=end_root,
        start_length=start_length,
        end_length=end_length,
        max_notes=max_notes,
    )

    raw_dict: dict[str, float] = {}
    norm_dict: dict[str, float] = {}
    weighted_dict: dict[str, float] = {}
    total = 0.0

    for sf in score_fns:
        r, n = sf.fn(ctx)
        raw_val = float(r.flat[0])
        norm_val = float(n.flat[0])
        w = weights.get(sf.name, sf.default_weight)
        w_val = norm_val * w

        raw_dict[sf.name] = raw_val
        norm_dict[sf.name] = norm_val
        weighted_dict[sf.name] = w_val
        total += w_val

    return {
        "raw": raw_dict,
        "normalized": norm_dict,
        "weighted": weighted_dict,
        "score": total,
    }


def results_to_dataframe(result: dict, top_n: int | None = None) -> pd.DataFrame:
    """Build a pandas DataFrame from ``solve()`` output.

    Columns include: rank, root, n_notes, mean_score, plus raw_* and norm_*
    for each sub-score, and the chroma vector.

    Args:
        result: dict returned by ``solve()``.
        top_n: limit to top N candidates (default: all best_indices).

    Returns:
        DataFrame sorted by mean score (lower=better).
    """
    import pandas as pd

    indices = result["best_indices"]
    if top_n is not None:
        indices = indices[:top_n]

    mean_scores = (result["scores"].T @ result["_tonality_weights"])
    raw = result.get("raw_subscores", {})
    norm = result.get("norm_subscores", {})

    rows = []
    for rank, idx in enumerate(indices, 1):
        row: dict = {
            "rank": rank,
            "candidate_idx": int(idx),
            "root": int(result["roots"][idx]),
            "n_notes": int(result["lengths"][idx]),
            "mean_score": float(mean_scores[idx]),
        }
        # Raw sub-scores (mean across scales for 2D arrays)
        for name, arr in raw.items():
            if arr.ndim == 2:
                row[f"raw_{name}"] = float(arr[:, idx].T @ result["_tonality_weights"])
            else:
                row[f"raw_{name}"] = float(arr[idx])
        # Normalized sub-scores (mean across scales for 2D arrays)
        for name, arr in norm.items():
            if arr.ndim == 2:
                row[f"norm_{name}"] = float(arr[:, idx].T @ result["_tonality_weights"])
            else:
                row[f"norm_{name}"] = float(arr[idx])

        rows.append(row)

    return pd.DataFrame(rows)


def solve(
    start_chroma: ChromaArray,
    end_chroma: ChromaArray,
    scale_semitones: list[np.ndarray],
    min_notes: int = 2,
    max_notes: int = 7,
    top_n: int = 20,
    weights: dict[str, float] | None = None,
    score_fns: list[ScoreFn] | None = None,
    exclude_same_root: bool = False,
    exclude_subsets: bool = False,
    dissonance_method: str = "smoothed",
    aggregation: str = "weighted_sum",
    tonality_weights: np.ndarray | None = None,
    preceding_root: int | None = None
) -> dict:
    """Find the best candidate chords for start → ? → end across multiple scales.

    Args:
        start_chroma: shape (12,) bool.
        end_chroma: shape (12,) bool.
        scale_semitones: list of scale definitions (each a list of semitones).
        min_notes, max_notes: cardinality range for candidates.
        top_n: number of top candidates to return.
        weights: scoring weights dict.  Keys match :attr:`ScoreFn.name`.
        score_fns: list of :class:`ScoreFn`.  ``None`` builds the default
            set, extracting ``min_voice_leading`` / ``max_voice_leading``
            from *weights* when present.
        exclude_same_root: if True, filter out candidates sharing root with
            start/end.
        exclude_subsets: if True, filter out candidates whose pitch classes
            are a strict subset of (or equal to) start or end.
        dissonance_method: ``"smoothed"`` (default), ``"pairwise"``,
            ``"max_pairwise"``.
        aggregation: ``"weighted_sum"``, ``"logsumexp"``, ``"weighted_product"``.
        preceding_root: optional root pitch-class (0-11) of the preceding chord.
            If provided, scores are averaged across tonalities using weights based
            on circle-of-fifths distance. If None, uses uniform averaging.

    Returns:
        dict with keys ``candidates``, ``scores``, ``best_indices``,
        ``best_scores``, ``lookups``, ``counts``, ``roots``, ``dissonance``,
        ``start_root``, ``end_root``, ``lengths``, ``raw_subscores``,
        ``norm_subscores``, plus internal arrays prefixed with ``_``.
    """
    # Resolve score_fns (backward compat: extract VL params from weights)
    if score_fns is None:
        min_vl = weights.get("min_voice_leading", 1.0) if weights else 1.0
        max_vl = weights.get("max_voice_leading", 6.0) if weights else 6.0
        score_fns = default_score_fns(min_vl, max_vl)

    # Build all scale lookups
    all_lookups = []
    all_counts = []
    for sc in scale_semitones:
        lk, ct = interval.build_all_scale_lookups(sc)
        all_lookups.append(lk)
        all_counts.append(ct)
    lookups = np.concatenate(all_lookups, axis=0)   # (n_scales, 12, 2, 2)
    counts = np.concatenate(all_counts, axis=0)     # (n_scales, 12)

    # Generate candidates
    candidates = generate_all_chromas(min_notes, max_notes)
    cand_st, cand_len = chromas_to_padded_semitones(candidates, max_notes)

    # Interpret start and end
    start_st = np.flatnonzero(start_chroma).astype(DT.St)
    end_st = np.flatnonzero(end_chroma).astype(DT.St)

    def _pad(st: np.ndarray) -> np.ndarray:
        padded = np.full(max_notes, _SENTINEL, dtype=DT.St)
        padded[: len(st)] = st
        return padded

    start_padded = _pad(start_st)[np.newaxis, :]
    end_padded = _pad(end_st)[np.newaxis, :]

    # Estimate roots
    cand_roots = estimate_roots(candidates)
    start_root = int(estimate_roots(start_chroma[np.newaxis, :])[0])
    end_root = int(estimate_roots(end_chroma[np.newaxis, :])[0])

    # Pre-compute dissonance and tertian quality
    cand_dissonance = batch_dissonance(candidates, roots=cand_roots, method=dissonance_method)
    cand_tertian = batch_tertian_score(candidates)

    # Batch interpret
    cand_interps = batch_interpret_canonical(cand_st, lookups)
    start_interps = batch_interpret_canonical(start_padded, lookups)
    end_interps = batch_interpret_canonical(end_padded, lookups)

    # Build scoring context
    ctx = ScoringContext(
        candidate_interps=cand_interps,
        start_interp=start_interps,
        end_interp=end_interps,
        candidate_semitones=cand_st,
        start_semitones=start_padded,
        end_semitones=end_padded,
        candidate_lengths=cand_len,
        candidate_dissonance=cand_dissonance,
        candidate_roots=cand_roots,
        candidate_tertian=cand_tertian,
        start_root=start_root,
        end_root=end_root,
        start_length=int(start_chroma.sum()),
        end_length=int(end_chroma.sum()),
        max_notes=max_notes,
    )

    # Score
    scores, raw_subscores, norm_subscores = score_candidates(
        ctx,
        score_fns=score_fns,
        weights=weights,
        aggregation=aggregation,
    )

    # Compute tonality weights and apply weighted averaging
    if tonality_weights is None:
        tonality_weights = compute_tonality_weights(preceding_root)
    # Reshape weights to (n_scales, 1) for broadcasting: scores @ weights
    mean_scores = scores.T @ tonality_weights  # (n_cand,)
    
    eligible = np.ones(len(candidates), dtype=bool)
    if exclude_same_root:
        eligible &= (cand_roots != start_root) & (cand_roots != end_root)
    if exclude_subsets:
        # A candidate is a subset of X if every pitch in candidate is also in X
        subset_of_start = (candidates & start_chroma).sum(axis=1) == candidates.sum(axis=1)
        subset_of_end = (candidates & end_chroma).sum(axis=1) == candidates.sum(axis=1)
        eligible &= ~subset_of_start & ~subset_of_end
    if eligible.all():
        best_idx = np.argsort(mean_scores)[:top_n]
    else:
        eligible_idx = np.flatnonzero(eligible)
        ranked = eligible_idx[np.argsort(mean_scores[eligible_idx])]
        best_idx = ranked[:top_n]

    return {
        "candidates": candidates,
        "scores": scores,
        "best_indices": best_idx,
        "best_scores": mean_scores[best_idx],
        "lookups": lookups,
        "counts": counts,
        "roots": cand_roots,
        "dissonance": cand_dissonance,
        "start_root": start_root,
        "end_root": end_root,
        "lengths": cand_len,
        "raw_subscores": raw_subscores,
        "norm_subscores": norm_subscores,
        # Internal arrays for score_candidate_breakdown
        "_cand_interps": cand_interps,
        "_start_interps": start_interps,
        "_end_interps": end_interps,
        "_cand_st": cand_st,
        "_start_st": start_padded,
        "_end_st": end_padded,
        "_tonality_weights": tonality_weights,
    }
