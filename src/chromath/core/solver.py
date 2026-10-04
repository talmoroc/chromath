from itertools import combinations

import numpy as np
import pandas as pd

from ..constants import DefaultMusicSystem as MS
from ..types import DT, SENTINEL, ChromaVec, ScaleLookupArray
from . import chroma, scale
from .scoring import (  # noqa: F401
    _TRIAD_PATTERNS,
    AGGREGATION_METHODS,
    DEFAULT_SCORE_FNS,
    ScoreFn,
    ScoringContext,
    _normalize_minmax,
    _semitone_distance,
    batch_alteration_cost,
    batch_dissonance,
    batch_resolution_score,
    batch_root_distance,
    batch_tertian_score,
    batch_voice_leading_cost,
    default_score_fns,
    estimate_roots,
    root_position_semitones,
)


def compute_tonality_weights(
    reference_root: int | None = None,
    distance_fn=None,
) -> np.ndarray:
    """Weights (12,) summing to 1 for the 12 tonalities: 1 / (1 + distance to *reference_root*), uniform if it is None."""
    if reference_root is None:
        return np.ones(MS.tones, dtype=np.float64) / MS.tones

    if distance_fn is None:
        distance_fn = scale._cof_distance

    weights = np.zeros(MS.tones, dtype=np.float64)
    for tonality_root in range(MS.tones):
        dist = distance_fn(tonality_root, reference_root)
        weights[tonality_root] = 1.0 / (1.0 + dist)

    # Normalize to sum to 1.0
    weights /= weights.sum()
    return weights


def generate_all_chromas(min_notes: int = 2, max_notes: int = 7) -> ChromaVec:
    """All chroma vectors with cardinality in [min_notes, max_notes], shape (n_candidates, 12)."""
    rows: list[np.ndarray] = []
    for k in range(min_notes, max_notes + 1):
        for combo in combinations(range(MS.tones), k):
            row = np.zeros(MS.tones, dtype=DT.Bool)
            row[list(combo)] = True
            rows.append(row)
    return np.array(rows, dtype=DT.Bool)


def batch_interpret_canonical(
    candidate_semitones: np.ndarray,
    lookups: ScaleLookupArray,
) -> np.ndarray:
    """Candidates (n_cand, max_notes) through lookups (n_scales, 12, 2, 2): (n_scales, n_cand, max_notes, 3) of (degree, semitone, alteration)."""
    lookups = np.asarray(lookups)
    n_scales = lookups.shape[0]
    n_cand, max_notes = candidate_semitones.shape

    mask = candidate_semitones != SENTINEL  # (n_cand, max_notes)
    safe_st = np.where(mask, candidate_semitones, 0)  # safe for indexing

    # lookups[s, safe_st] → (n_scales, n_cand, max_notes, 2)  [degree, alteration]
    # We use advanced indexing: lookups[:, safe_st, 0] with broadcasting
    deg_alt = lookups[:, safe_st, 0]  # (n_scales, n_cand, max_notes, 2)

    result = np.full((n_scales, n_cand, max_notes, 3), SENTINEL, dtype=DT.Note)

    mask_bc = mask[np.newaxis, ...]  # (1, n_cand, max_notes)

    result[..., 0] = np.where(mask_bc, deg_alt[..., 0], SENTINEL)  # degree
    result[..., 1] = np.where(
        mask_bc,
        candidate_semitones[np.newaxis, ...],
        SENTINEL,
    )  # semitone
    result[..., 2] = np.where(mask_bc, deg_alt[..., 1], SENTINEL)  # alteration

    return result


def score_candidates(
    ctx: ScoringContext,
    score_fns: list[ScoreFn] | None = None,
    weights: dict[str, float] | None = None,
    aggregation: str = "weighted_sum",
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Score all candidates for the middle of start → ? → end: scores (n_scales, n_cand), raw sub-scores, normalized sub-scores."""
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
            components[name] = np.broadcast_to(arr[np.newaxis, :], (n_scales, arr.shape[0]))
        else:
            components[name] = arr

    # Resolve weights: explicit dict overrides, then ScoreFn defaults
    resolved_weights: dict[str, float] = {}
    for sf in score_fns:
        resolved_weights[sf.name] = weights.get(sf.name, sf.default_weight)

    from .scoring import AGGREGATION_METHODS as _agg_methods

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
    """Raw, normalized and weighted sub-scores of one candidate at one scale."""
    if score_fns is None:
        score_fns = DEFAULT_SCORE_FNS
    if weights is None:
        weights = {}

    ctx = ScoringContext(
        candidate_interps=candidate_interps[scale_idx : scale_idx + 1, candidate_idx : candidate_idx + 1],
        start_interp=start_interp[scale_idx : scale_idx + 1],
        end_interp=end_interp[scale_idx : scale_idx + 1],
        candidate_semitones=candidate_semitones[candidate_idx : candidate_idx + 1],
        start_semitones=start_semitones,
        end_semitones=end_semitones,
        candidate_lengths=np.array([candidate_length], dtype=DT.Note),
        candidate_dissonance=np.array([candidate_dissonance], dtype=np.float64),
        candidate_roots=np.array([candidate_root], dtype=DT.Note),
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
    """DataFrame of a solve() result, sorted by mean score."""
    import pandas as pd

    indices = result["best_indices"]
    if top_n is not None:
        indices = indices[:top_n]

    mean_scores = result["scores"].T @ result["_tonality_weights"]
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
    start_chroma: ChromaVec,
    end_chroma: ChromaVec,
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
    preceding_root: int | None = None,
) -> dict:
    """Find the best candidate chords for start → ? → end across multiple scales."""
    # Resolve score_fns (backward compat: extract VL params from weights)
    if score_fns is None:
        min_vl = weights.get("min_voice_leading", 1.0) if weights else 1.0
        max_vl = weights.get("max_voice_leading", 6.0) if weights else 6.0
        score_fns = default_score_fns(min_vl, max_vl)

    # Build all scale lookups
    all_lookups = []
    all_counts = []
    for sc in scale_semitones:
        lk, ct = scale.build_all_scale_lookups(sc)
        all_lookups.append(lk)
        all_counts.append(ct)
    lookups = np.concatenate(all_lookups, axis=0)  # (n_scales, 12, 2, 2)
    counts = np.concatenate(all_counts, axis=0)  # (n_scales, 12)

    # Generate candidates
    candidates = generate_all_chromas(min_notes, max_notes)
    cand_keys = chroma.from_vector(candidates)
    cand_st = chroma.to_members(cand_keys)
    cand_len = chroma.cardinality(cand_keys).astype(DT.Note)

    # Interpret start and end
    start_padded = chroma.to_members(chroma.from_vector(start_chroma))[np.newaxis, :]
    end_padded = chroma.to_members(chroma.from_vector(end_chroma))[np.newaxis, :]

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
