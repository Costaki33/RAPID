"""Pick and probability comparison between two backends (e.g. FP16 vs FP32).

Two flavors of comparison:

1. **Probability-trace drift** — element-wise statistics between two aligned
   ``(B, T, C)`` outputs:
     - mean absolute error
     - max absolute error
     - RMSE
     - Pearson correlation
   Works on any pair of backends that both produce aligned post-processed outputs.

2. **Pick-time drift** at a threshold (placeholder for the 100-event dataset):
     - detect picks via ``trigger_onset`` on each probability trace
     - match picks across backends by nearest in time
     - report time-delta distribution (mean, median, p95)

The pick-time function is intentionally minimal so we can plug in the user-
provided manual picks once we have them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TraceStats:
    mae: float
    max_abs_err: float
    rmse: float
    pearson: float
    n_samples: int


def compare_probabilities(a: np.ndarray, b: np.ndarray) -> TraceStats:
    """Compute element-wise drift stats between two aligned prediction arrays."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    af = a.astype(np.float64, copy=False)
    bf = b.astype(np.float64, copy=False)
    mask = np.isfinite(af) & np.isfinite(bf)
    if not mask.any():
        return TraceStats(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    da = af[mask]
    db = bf[mask]
    diff = da - db
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    mx = float(np.max(np.abs(diff)))
    if da.std() == 0 or db.std() == 0:
        p = float("nan")
    else:
        p = float(np.corrcoef(da, db)[0, 1])
    return TraceStats(mae=mae, max_abs_err=mx, rmse=rmse, pearson=p, n_samples=int(mask.sum()))


def as_dict(s: TraceStats) -> Dict[str, float]:
    d = asdict(s)
    d["n_samples"] = int(d["n_samples"])
    return d


# ---------------------------------------------------------------------------
# Pick extraction (simple onset detector on one channel)
# ---------------------------------------------------------------------------


def extract_picks_simple(
    prob_trace: np.ndarray,
    threshold: float = 0.3,
    min_separation: int = 50,
) -> np.ndarray:
    """Return sample indices where ``prob_trace`` crosses ``threshold`` upwards.

    Parameters
    ----------
    prob_trace : np.ndarray
        1-D probability vs time.
    threshold : float
    min_separation : int
        Minimum samples between successive picks.
    """
    prob_trace = np.asarray(prob_trace)
    above = prob_trace >= threshold
    # Rising edges:
    onsets = np.where(np.diff(above.astype(np.int8)) == 1)[0] + 1
    if onsets.size == 0:
        return onsets
    keep = [onsets[0]]
    for o in onsets[1:]:
        if o - keep[-1] >= min_separation:
            keep.append(o)
    return np.asarray(keep, dtype=np.int64)


def match_picks(
    picks_a: np.ndarray,
    picks_b: np.ndarray,
    tol_samples: int = 50,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy nearest-neighbor match between two pick arrays."""
    pairs: List[Tuple[int, int]] = []
    used_b = set()
    unmatched_a: List[int] = []
    for ai, a in enumerate(picks_a):
        if picks_b.size == 0:
            unmatched_a.append(ai)
            continue
        diffs = np.abs(picks_b - a)
        diffs[list(used_b)] = np.iinfo(np.int64).max
        j = int(np.argmin(diffs))
        if diffs[j] <= tol_samples:
            pairs.append((ai, j))
            used_b.add(j)
        else:
            unmatched_a.append(ai)
    unmatched_b = [j for j in range(picks_b.size) if j not in used_b]
    return pairs, unmatched_a, unmatched_b


def pick_time_drift_samples(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    channel: int = 1,
    threshold: float = 0.3,
) -> Dict[str, float]:
    """Summarize pick-time differences between two prediction tensors.

    Works on a ``(B, T, C)`` layout (SeisBench post-processed outputs).
    """
    if preds_a.shape != preds_b.shape:
        raise ValueError(f"Shape mismatch: {preds_a.shape} vs {preds_b.shape}")
    deltas = []
    n_missing_a = 0
    n_missing_b = 0
    n_pairs = 0
    for i in range(preds_a.shape[0]):
        pa = extract_picks_simple(preds_a[i, :, channel], threshold=threshold)
        pb = extract_picks_simple(preds_b[i, :, channel], threshold=threshold)
        pairs, miss_a, miss_b = match_picks(pa, pb)
        for ai, bj in pairs:
            deltas.append(int(pa[ai]) - int(pb[bj]))
        n_missing_a += len(miss_a)
        n_missing_b += len(miss_b)
        n_pairs += len(pairs)
    if not deltas:
        return dict(
            n_pairs=0,
            n_missing_fp32=n_missing_a,
            n_missing_fp16=n_missing_b,
            mean_delta_samples=float("nan"),
            median_delta_samples=float("nan"),
            p95_abs_delta_samples=float("nan"),
            max_abs_delta_samples=float("nan"),
        )
    a = np.asarray(deltas)
    return dict(
        n_pairs=n_pairs,
        n_missing_fp32=n_missing_a,
        n_missing_fp16=n_missing_b,
        mean_delta_samples=float(np.mean(a)),
        median_delta_samples=float(np.median(a)),
        p95_abs_delta_samples=float(np.percentile(np.abs(a), 95)),
        max_abs_delta_samples=float(np.max(np.abs(a))),
    )
