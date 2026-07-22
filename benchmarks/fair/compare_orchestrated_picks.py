#!/usr/bin/env python3
"""Compare orchestrated RAPID picks against STEAD/TXED catalog labels.

Reads the per-station picks saved by a ``RunEQCCTPro`` run (ASCII/CSV station
files ``<output_dir>/<chunk>/<station>_outputs/X_prediction_results.csv``) and
the ``manifest.json`` produced by ``build_seisbench_network.py``, then reports
the pick-quality metrics Camilo asked for:

  * total catalog vs detected picks, matched / missing / additional / duplicated
  * precision / recall / F1 for matched picks
  * mean / median / std of ΔT (detected - catalog), P50 / P95 / P99 of |ΔT|
  * fraction of matched picks within +/- 1, 5, 10 samples (at 100 Hz)

Detected pick absolute times are converted back to a 100 Hz sample offset using
the fixed network start time ``T0`` stored in the manifest, so they are directly
comparable to the catalog sample indices. P and S phases are scored separately.

Usable standalone::

    python benchmarks/fair/compare_orchestrated_picks.py \\
        --manifest data/seisbench_networks/stead_580st/manifest.json \\
        --picks-dir results/seisbench_sweep/runs/<tag>/picks \\
        --out results/seisbench_sweep/pick_quality/<tag>.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from obspy import UTCDateTime

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rapid.quality import match_picks  # noqa: E402

TOL_SAMPLES = 50
SR = 100.0
TOLERANCE_BUCKETS = (1, 5, 10)


def _parse_time_to_sample(time_str: str, t0: UTCDateTime) -> Optional[float]:
    s = (time_str or "").strip()
    if not s or s.lower() in ("none", "nan", "na"):
        return None
    try:
        return (UTCDateTime(s) - t0) * SR
    except Exception:
        return None


def _read_station_picks(csv_path: Path, t0: UTCDateTime):
    """Return (p_samples, s_samples) detected for one station file."""
    p_samples: List[float] = []
    s_samples: List[float] = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ps = _parse_time_to_sample(row.get("p_arrival_time", ""), t0)
                if ps is not None:
                    p_samples.append(ps)
                ss = _parse_time_to_sample(row.get("s_arrival_time", ""), t0)
                if ss is not None:
                    s_samples.append(ss)
    except FileNotFoundError:
        pass
    return np.asarray(sorted(p_samples)), np.asarray(sorted(s_samples))


def _phase_metrics(deltas: List[float], tp: int, fn: int, fp: int,
                   dup: int, n_catalog: int, n_detected: int) -> Dict:
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) and np.isfinite(precision) and np.isfinite(recall)
          else float("nan"))
    out = dict(
        n_catalog=n_catalog,
        n_detected=n_detected,
        matched=tp,
        missing=fn,
        additional=fp,
        duplicated=dup,
        precision=precision,
        recall=recall,
        f1=f1,
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
        out.update(mean_dt=float("nan"), median_dt=float("nan"), std_dt=float("nan"),
                   p50_abs_dt=float("nan"), p95_abs_dt=float("nan"), p99_abs_dt=float("nan"))
        for b in TOLERANCE_BUCKETS:
            out[f"frac_within_{b}"] = float("nan")
    return out


def compare_network_picks(
    *,
    manifest_path: Path,
    picks_dir: Path,
    out_json: Optional[Path] = None,
    label: str = "",
    p_threshold: float = 0.3,
    s_threshold: float = 0.3,
) -> Optional[Dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    stations = manifest["stations"]
    t0 = UTCDateTime(manifest["meta"]["t0"])

    # Index saved station files by station code.
    files: Dict[str, Path] = {}
    for path in glob.glob(str(Path(picks_dir) / "**" / "*_outputs" / "X_prediction_results.csv"),
                          recursive=True):
        station = Path(path).parent.name[: -len("_outputs")]
        files[station] = Path(path)
    if not files:
        print(f"WARNING: no station pick files under {picks_dir}")
        return None

    p_deltas: List[float] = []
    s_deltas: List[float] = []
    p_tp = p_fn = p_fp = p_dup = p_cat = p_det = 0
    s_tp = s_fn = s_fp = s_dup = s_cat = s_det = 0

    for station, meta in stations.items():
        cat_p = meta.get("p_sample")
        cat_s = meta.get("s_sample")
        det_p, det_s = (np.asarray([]), np.asarray([]))
        if station in files:
            det_p, det_s = _read_station_picks(files[station], t0)

        # P phase
        if cat_p is not None:
            p_cat += 1
            p_det += int(det_p.size)
            cat_arr = np.asarray([float(cat_p)])
            pairs, miss_a, miss_b = match_picks(cat_arr, det_p, tol_samples=TOL_SAMPLES)
            for ai, bj in pairs:
                p_deltas.append(float(det_p[bj]) - float(cat_arr[ai]))
            p_tp += len(pairs)
            p_fn += len(miss_a)
            # additional vs duplicated: extra detections within tol of the catalog
            # pick are duplicates, the rest are genuine additional picks.
            for bj in miss_b:
                if abs(float(det_p[bj]) - float(cat_p)) <= TOL_SAMPLES:
                    p_dup += 1
                else:
                    p_fp += 1

        # S phase
        if cat_s is not None:
            s_cat += 1
            s_det += int(det_s.size)
            cat_arr = np.asarray([float(cat_s)])
            pairs, miss_a, miss_b = match_picks(cat_arr, det_s, tol_samples=TOL_SAMPLES)
            for ai, bj in pairs:
                s_deltas.append(float(det_s[bj]) - float(cat_arr[ai]))
            s_tp += len(pairs)
            s_fn += len(miss_a)
            for bj in miss_b:
                if abs(float(det_s[bj]) - float(cat_s)) <= TOL_SAMPLES:
                    s_dup += 1
                else:
                    s_fp += 1

    result = dict(
        label=label,
        dataset=manifest["meta"]["dataset"],
        n_stations=manifest["meta"]["n_stations"],
        n_unique=manifest["meta"].get("n_unique"),
        match_tolerance_samples=TOL_SAMPLES,
        p_threshold=p_threshold,
        s_threshold=s_threshold,
        stations_with_picks=len(files),
        P=_phase_metrics(p_deltas, p_tp, p_fn, p_fp, p_dup, p_cat, p_det),
        S=_phase_metrics(s_deltas, s_tp, s_fn, s_fp, s_dup, s_cat, s_det),
    )

    if out_json is not None:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(result, indent=2))

    _print_summary(result)
    return result


def _print_summary(r: Dict) -> None:
    print(f"\n[{r['label']}] {r['dataset']} {r['n_stations']} stations "
          f"(unique={r['n_unique']}, stations_with_picks={r['stations_with_picks']})")
    for phase in ("P", "S"):
        m = r[phase]
        if not m["n_catalog"]:
            continue
        print(f"  {phase}: catalog={m['n_catalog']} detected={m['n_detected']} "
              f"matched={m['matched']} missing={m['missing']} additional={m['additional']} dup={m['duplicated']}")
        print(f"     precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
              f"| meanΔT={m['mean_dt']:.2f} medianΔT={m['median_dt']:.2f} "
              f"P95|ΔT|={m['p95_abs_dt']:.1f} samp")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--picks-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--label", default="")
    ap.add_argument("--p-threshold", type=float, default=0.3)
    ap.add_argument("--s-threshold", type=float, default=0.3)
    args = ap.parse_args()

    compare_network_picks(
        manifest_path=args.manifest,
        picks_dir=args.picks_dir,
        out_json=args.out,
        label=args.label or args.manifest.parent.name,
        p_threshold=args.p_threshold,
        s_threshold=args.s_threshold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
