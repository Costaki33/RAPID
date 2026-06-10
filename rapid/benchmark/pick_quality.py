"""Pick-quality metrics (Camilo email): ΔT, P/R/F1, matched/missing/additional picks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from obspy import UTCDateTime

from rapid.quality import match_picks

SR = 100.0
TOL_SAMPLES = 50
TOLERANCE_BUCKETS = (1, 5, 10)


def load_manifest_catalog(manifest_path: Path) -> Tuple[UTCDateTime, Dict[str, Dict[str, Any]]]:
    manifest = json.loads(Path(manifest_path).read_text())
    t0 = UTCDateTime(manifest["meta"]["t0"])
    return t0, manifest["stations"]


def _phase_metrics(
    deltas: List[float],
    tp: int,
    fn: int,
    fp: int,
    dup: int,
    n_catalog: int,
    n_detected: int,
) -> Dict[str, Any]:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) and np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )
    out: Dict[str, Any] = dict(
        n_catalog=n_catalog,
        n_detected=n_detected,
        matched=tp,
        missing=fn,
        additional=fp,
        duplicated=dup,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
    if deltas:
        a = np.asarray(deltas, dtype=float)
        absa = np.abs(a)
        out.update(
            mean_dt=float(np.mean(a)),
            median_dt=float(np.median(a)),
            std_dt=float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            p50_abs_dt=float(np.percentile(absa, 50)),
            p95_abs_dt=float(np.percentile(absa, 95)),
            p99_abs_dt=float(np.percentile(absa, 99)),
        )
        for b in TOLERANCE_BUCKETS:
            out[f"frac_within_{b}"] = float(np.mean(absa <= b))
    else:
        out.update(
            mean_dt=float("nan"),
            median_dt=float("nan"),
            std_dt=float("nan"),
            p50_abs_dt=float("nan"),
            p95_abs_dt=float("nan"),
            p99_abs_dt=float("nan"),
        )
        for b in TOLERANCE_BUCKETS:
            out[f"frac_within_{b}"] = float("nan")
    return out


def _as_pick_array(value: Any) -> Optional[np.ndarray]:
    """Normalize manifest scalars or reference-pick lists to a 1-D float array."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=float)
        return arr if arr.size else None
    try:
        return np.asarray([float(value)], dtype=float)
    except (TypeError, ValueError):
        return None


def _catalog_phase_array(meta: Dict[str, Any], phase: str) -> Optional[np.ndarray]:
    phase_l = phase.lower()
    return _as_pick_array(meta.get(f"{phase_l}_sample")) or _as_pick_array(meta.get(phase_l))


def _is_duplicate(det_val: float, cat_arr: np.ndarray, tol_samples: int) -> bool:
    return bool(np.any(np.abs(cat_arr - det_val) <= tol_samples))


def compare_pick_sets(
    *,
    catalog_by_station: Dict[str, Dict[str, Any]],
    detected_by_station: Dict[str, Dict[str, List[float]]],
    label: str = "",
    reference_label: str = "catalog",
    tol_samples: int = TOL_SAMPLES,
) -> Dict[str, Any]:
    """Compare detected P/S sample indices per station against catalog or reference picks.

    ``detected_by_station[sta]`` = ``{"p": [samples...], "s": [samples...]}``.
    Catalog entries may be manifest-style (``p_sample`` / ``s_sample`` scalars) or
    reference-pick JSON (``p`` / ``s`` lists), same as detected format.
    """
    p_deltas: List[float] = []
    s_deltas: List[float] = []
    p_tp = p_fn = p_fp = p_dup = p_cat = p_det = 0
    s_tp = s_fn = s_fp = s_dup = s_cat = s_det = 0

    all_stations = set(catalog_by_station) | set(detected_by_station)
    for station in sorted(all_stations):
        meta = catalog_by_station.get(station, {})
        det = detected_by_station.get(station, {})
        cat_p = _catalog_phase_array(meta, "P")
        cat_s = _catalog_phase_array(meta, "S")
        det_p = np.asarray(det.get("p", []), dtype=float)
        det_s = np.asarray(det.get("s", []), dtype=float)

        if cat_p is not None:
            p_cat += 1
            p_det += int(det_p.size)
            pairs, miss_a, miss_b = match_picks(cat_p, det_p, tol_samples=tol_samples)
            for ai, bj in pairs:
                p_deltas.append(float(det_p[bj]) - float(cat_p[ai]))
            p_tp += len(pairs)
            p_fn += len(miss_a)
            for bj in miss_b:
                if _is_duplicate(float(det_p[bj]), cat_p, tol_samples):
                    p_dup += 1
                else:
                    p_fp += 1

        if cat_s is not None:
            s_cat += 1
            s_det += int(det_s.size)
            pairs, miss_a, miss_b = match_picks(cat_s, det_s, tol_samples=tol_samples)
            for ai, bj in pairs:
                s_deltas.append(float(det_s[bj]) - float(cat_s[ai]))
            s_tp += len(pairs)
            s_fn += len(miss_a)
            for bj in miss_b:
                if _is_duplicate(float(det_s[bj]), cat_s, tol_samples):
                    s_dup += 1
                else:
                    s_fp += 1

    return dict(
        label=label,
        reference=reference_label,
        match_tolerance_samples=tol_samples,
        n_stations=len(all_stations),
        stations_with_detections=len(detected_by_station),
        P=_phase_metrics(p_deltas, p_tp, p_fn, p_fp, p_dup, p_cat, p_det),
        S=_phase_metrics(s_deltas, s_tp, s_fn, s_fp, s_dup, s_cat, s_det),
    )


def catalog_from_manifest_stations(stations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize manifest station entries to p_sample/s_sample keys."""
    out: Dict[str, Dict[str, Any]] = {}
    for sta, meta in stations.items():
        out[sta] = {
            "p_sample": meta.get("p_sample"),
            "s_sample": meta.get("s_sample"),
        }
    return out
