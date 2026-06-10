#!/usr/bin/env python3
"""Comprehensive pick quality analysis for reviewer feedback.

Runs annotate() and lean inference paths on real SeisBench catalog traces
(STEAD / TXED) using the same preprocessing/windowing as the matrix benchmark,
then reports matched / missing / additional picks and ΔT statistics.

Usage:
    python scripts/run_pick_quality_analysis.py \\
        --n-traces 50 \\
        --dataset stead \\
        --devices cpu cuda:0 \\
        --output results/pick_quality_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seisbench.models as sbm
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rapid.backends.baseline import BaselineAnnotate
from rapid.backends.base import BackendError
from rapid.backends.lean_pytorch import LeanPyTorchBackend
from rapid.quality import extract_picks_simple
from rapid.seisbench_matrix import (  # noqa: E402
    _annotate_stream_to_window_pred,
    _cut_raw_window,
)
from rapid.seisbench_precision_eval import (  # noqa: E402
    DEFAULT_MODELS,
    ModelSpec,
    _finite_int,
    catalog_mask,
    catalog_pick_columns,
    cut_window,
    load_dataset,
    parse_models_arg,
    phase_indices,
    preprocess_array,
    waves_to_stream,
)

MATCH_TOLERANCE_SAMPLES = 50
PICK_THRESHOLD = 0.3

METHODS: Tuple[Dict[str, Any], ...] = (
    {"name": "annotate_fp32", "kind": "annotate"},
    {"name": "lean_fp16", "kind": "lean", "dtype": "fp16", "compile": False},
    {"name": "lean_bf16", "kind": "lean", "dtype": "bf16", "compile": False},
    {"name": "lean_bf16_compile", "kind": "lean", "dtype": "bf16", "compile": True},
)


@dataclass
class PickResult:
    trace_idx: int
    method: str
    device: str
    model: str
    catalog_p: Optional[int]
    catalog_s: Optional[int]
    detected_p: List[int]
    detected_s: List[int]
    wall_time_s: float
    error: Optional[str] = None


@dataclass
class MatchResult:
    n_catalog: int
    n_detected: int
    n_matched: int
    n_missing: int
    n_additional: int
    n_duplicates: int
    deltas: List[int]


def match_picks_to_catalog(
    catalog_pick: Optional[int],
    detected_picks: List[int],
    tolerance: int = MATCH_TOLERANCE_SAMPLES,
) -> MatchResult:
    if catalog_pick is None:
        return MatchResult(0, len(detected_picks), 0, 0, len(detected_picks), 0, [])

    if not detected_picks:
        return MatchResult(1, 0, 0, 1, 0, 0, [])

    matched_indices: List[int] = []
    deltas: List[int] = []
    for i, det in enumerate(detected_picks):
        delta = int(det) - int(catalog_pick)
        if abs(delta) <= tolerance:
            matched_indices.append(i)
            deltas.append(delta)

    n_matched = min(1, len(matched_indices))
    n_duplicates = max(0, len(matched_indices) - 1)
    n_additional = len(detected_picks) - len(matched_indices)
    n_missing = 1 - n_matched

    return MatchResult(
        n_catalog=1,
        n_detected=len(detected_picks),
        n_matched=n_matched,
        n_missing=n_missing,
        n_additional=n_additional,
        n_duplicates=n_duplicates,
        deltas=deltas[:1],
    )


def extract_all_picks(
    pred: np.ndarray,
    *,
    p_idx: int,
    s_idx: int,
    threshold: float = PICK_THRESHOLD,
    min_separation: int = 50,
) -> Tuple[List[int], List[int]]:
    if pred.size == 0 or pred.ndim != 3:
        return [], []
    p_picks = extract_picks_simple(
        pred[0, :, p_idx], threshold=threshold, min_separation=min_separation
    )
    s_picks = extract_picks_simple(
        pred[0, :, s_idx], threshold=threshold, min_separation=min_separation
    )
    return p_picks.tolist(), s_picks.tolist()


def compute_statistics(deltas: List[int], fs_hz: float = 100.0) -> Dict[str, float]:
    if not deltas:
        nan = float("nan")
        return {
            "n": 0,
            "mean_samples": nan,
            "median_samples": nan,
            "std_samples": nan,
            "p50_samples": nan,
            "p95_samples": nan,
            "p99_samples": nan,
            "mean_ms": nan,
            "median_ms": nan,
            "std_ms": nan,
            "p50_ms": nan,
            "p95_ms": nan,
            "p99_ms": nan,
            "pct_within_1_sample": 0.0,
            "pct_within_5_samples": 0.0,
            "pct_within_10_samples": 0.0,
        }

    arr = np.asarray(deltas, dtype=np.float64)
    abs_arr = np.abs(arr)
    ms = 1000.0 / fs_hz

    return {
        "n": int(len(deltas)),
        "mean_samples": float(np.mean(arr)),
        "median_samples": float(np.median(arr)),
        "std_samples": float(np.std(arr)),
        "p50_samples": float(np.percentile(abs_arr, 50)),
        "p95_samples": float(np.percentile(abs_arr, 95)),
        "p99_samples": float(np.percentile(abs_arr, 99)),
        "mean_ms": float(np.mean(arr) * ms),
        "median_ms": float(np.median(arr) * ms),
        "std_ms": float(np.std(arr) * ms),
        "p50_ms": float(np.percentile(abs_arr, 50) * ms),
        "p95_ms": float(np.percentile(abs_arr, 95) * ms),
        "p99_ms": float(np.percentile(abs_arr, 99) * ms),
        "pct_within_1_sample": float(np.mean(abs_arr <= 1) * 100),
        "pct_within_5_samples": float(np.mean(abs_arr <= 5) * 100),
        "pct_within_10_samples": float(np.mean(abs_arr <= 10) * 100),
    }


def aggregate_results(
    results: List[PickResult],
    model: str,
    method: str,
    device: str,
) -> Dict[str, Any]:
    filtered = [
        r
        for r in results
        if r.model == model and r.method == method and r.device == device and r.error is None
    ]
    if not filtered:
        return {"error": "no_data"}

    all_deltas: List[int] = []
    total_catalog_p = total_detected_p = total_matched_p = 0
    total_missing_p = total_additional_p = total_duplicates_p = 0

    for r in filtered:
        match_p = match_picks_to_catalog(r.catalog_p, r.detected_p)
        all_deltas.extend(match_p.deltas)
        total_catalog_p += match_p.n_catalog
        total_detected_p += match_p.n_detected
        total_matched_p += match_p.n_matched
        total_missing_p += match_p.n_missing
        total_additional_p += match_p.n_additional
        total_duplicates_p += match_p.n_duplicates

    precision = total_matched_p / total_detected_p if total_detected_p > 0 else 0.0
    recall = total_matched_p / total_catalog_p if total_catalog_p > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "model": model,
        "method": method,
        "device": device,
        "n_traces": len(filtered),
        "total_catalog_picks": total_catalog_p,
        "total_detected_picks": total_detected_p,
        "total_matched": total_matched_p,
        "total_missing": total_missing_p,
        "total_additional": total_additional_p,
        "total_duplicates": total_duplicates_p,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "delta_statistics": compute_statistics(all_deltas),
        "wall_time_mean_s": float(np.mean([r.wall_time_s for r in filtered])),
    }


def build_trace_items(
    ds,
    *,
    sb_model,
    sample_indices: np.ndarray,
    p_col: str,
    s_col: Optional[str],
    n_samples: int,
) -> List[Dict[str, Any]]:
    sr = float(sb_model.sampling_rate)
    in_samples = int(sb_model.in_samples)
    items: List[Dict[str, Any]] = []

    for row_idx in sample_indices:
        row_idx = int(row_idx)
        try:
            waves, meta = ds.get_sample(row_idx, sampling_rate=sr)
        except Exception:
            continue

        co = str(meta.get("trace_component_order") or "ZNE")
        if waves.ndim != 2:
            continue

        p_cat = _finite_int(meta.get(p_col))
        if p_cat is None or not (0 <= p_cat < waves.shape[1]):
            continue

        s_cat = _finite_int(meta.get(s_col)) if s_col else None
        if s_cat is not None and not (0 <= s_cat < waves.shape[1]):
            s_cat = None

        try:
            waves_win, raw_start, p_idx_in_win = _cut_raw_window(
                waves, n_samples=n_samples, p_sample=p_cat
            )
        except ValueError:
            continue

        arr_full = preprocess_array(sb_model, waves_win, sr, co)
        if arr_full is None:
            continue

        t_pp = int(arr_full.shape[1])
        if t_pp != int(waves_win.shape[1]):
            continue

        try:
            win, model_start, p_mod = cut_window(arr_full, in_samples, int(p_idx_in_win))
        except ValueError:
            continue

        s_mod: Optional[int] = None
        if s_cat is not None:
            s_idx_arr = int(s_cat) - int(raw_start)
            if 0 <= s_idx_arr < t_pp:
                s_off = s_idx_arr - int(model_start)
                if 0 <= s_off < in_samples:
                    s_mod = int(s_off)

        raw_stream = waves_to_stream(waves_win, sr, co)
        batch = win[None, ...].astype(np.float32, copy=False)

        items.append(
            {
                "trace_idx": row_idx,
                "raw_stream": raw_stream,
                "batch": batch,
                "p_win": int(p_mod),
                "s_win": s_mod,
                "preprocessed_T": t_pp,
                "model_start": int(model_start),
                "in_samples": in_samples,
            }
        )

    return items


def run_annotate(
    spec: ModelSpec,
    device: str,
    item: Dict[str, Any],
) -> Tuple[np.ndarray, float]:
    backend = BaselineAnnotate(spec.parent, spec.child, device=device)
    backend.load()
    try:
        t0 = time.perf_counter()
        ann = backend.annotate_stream(item["raw_stream"])
        wall = time.perf_counter() - t0
        pred = _annotate_stream_to_window_pred(
            ann,
            backend.model,
            preprocessed_T=item["preprocessed_T"],
            window_start=item["model_start"],
            in_samples=item["in_samples"],
            input_stream=item["raw_stream"],
        )
        if pred is None:
            raise RuntimeError("annotate() prediction mapping failed")
        return pred, wall
    finally:
        backend.close()


def run_lean(
    spec: ModelSpec,
    device: str,
    item: Dict[str, Any],
    *,
    dtype: str,
    compile_model: bool,
) -> Tuple[np.ndarray, float]:
    backend = LeanPyTorchBackend(
        spec.parent,
        spec.child,
        device=device,
        dtype=dtype,
        compile=compile_model,
    )
    backend.load()
    try:
        t0 = time.perf_counter()
        pred = backend.infer_batch(item["batch"])
        wall = time.perf_counter() - t0
        return pred, wall
    finally:
        backend.close()


def method_allowed(spec: ModelSpec, method: Dict[str, Any]) -> bool:
    if method["kind"] == "annotate":
        return True
    if spec.parent == "EQTransformer" and method.get("dtype") == "fp16":
        return False
    return True


def run_once(
    *,
    ds,
    models: List[ModelSpec],
    devices: List[str],
    args: argparse.Namespace,
    seed: int,
    output_path: Path,
) -> None:
    """Execute one full pick-quality pass for a given seed and write a JSON file."""
    p_col, s_col = catalog_pick_columns(ds)
    mask = catalog_mask(ds, require_s=False)
    valid_indices = np.flatnonzero(mask)

    n_sample = min(args.n_traces, valid_indices.size)
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(valid_indices, size=n_sample, replace=False)
    print(f"  Sampled {n_sample} traces (seed {seed})")

    all_results: List[PickResult] = []

    for spec in models:
        print(f"\n=== {spec.label} ===")
        try:
            sb_model = getattr(sbm, spec.parent).from_pretrained(spec.child)
            p_i, s_i = phase_indices(sb_model)
        except Exception as exc:
            print(f"  Skip model: {exc}")
            continue

        trace_items = build_trace_items(
            ds,
            sb_model=sb_model,
            sample_indices=sample_indices,
            p_col=p_col,
            s_col=s_col,
            n_samples=args.n_samples,
        )
        print(f"  Prepared {len(trace_items)} trace windows")

        if not trace_items:
            continue

        for device in devices:
            for method in METHODS:
                if not method_allowed(spec, method):
                    continue

                print(f"  {method['name']} @ {device}...", end=" ", flush=True)
                ok = err = 0

                for item in trace_items:
                    try:
                        if method["kind"] == "annotate":
                            pred, wall = run_annotate(spec, device, item)
                        else:
                            pred, wall = run_lean(
                                spec,
                                device,
                                item,
                                dtype=method["dtype"],
                                compile_model=method["compile"],
                            )

                        p_picks, s_picks = extract_all_picks(
                            pred, p_idx=p_i, s_idx=s_i, threshold=PICK_THRESHOLD
                        )
                        all_results.append(
                            PickResult(
                                trace_idx=item["trace_idx"],
                                method=method["name"],
                                device=device,
                                model=spec.label,
                                catalog_p=item["p_win"],
                                catalog_s=item["s_win"],
                                detected_p=p_picks,
                                detected_s=s_picks,
                                wall_time_s=wall,
                            )
                        )
                        ok += 1
                    except (BackendError, RuntimeError, Exception) as exc:
                        all_results.append(
                            PickResult(
                                trace_idx=item["trace_idx"],
                                method=method["name"],
                                device=device,
                                model=spec.label,
                                catalog_p=item["p_win"],
                                catalog_s=item["s_win"],
                                detected_p=[],
                                detected_s=[],
                                wall_time_s=0.0,
                                error=str(exc),
                            )
                        )
                        err += 1

                print(f"{ok} ok, {err} errors")

    aggregated: List[Dict[str, Any]] = []
    for spec in models:
        for method in METHODS:
            if not method_allowed(spec, method):
                continue
            for device in devices:
                agg = aggregate_results(all_results, spec.label, method["name"], device)
                if "error" not in agg:
                    aggregated.append(agg)

    output_data = {
        "metadata": {
            "dataset": args.dataset,
            "n_traces_requested": args.n_traces,
            "n_traces_sampled": int(n_sample),
            "n_samples": args.n_samples,
            "devices": devices,
            "models": [m.label for m in models],
            "match_tolerance_samples": MATCH_TOLERANCE_SAMPLES,
            "pick_threshold": PICK_THRESHOLD,
            "seed": seed,
            "timestamp": time.time(),
        },
        "raw_results": [asdict(r) for r in all_results],
        "aggregated": aggregated,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("SUMMARY: P-Pick Detection")
    print("=" * 60)
    print(
        f"{'Model':<15} {'Method':<22} {'Device':<8} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'Med ΔT':>10}"
    )
    print("-" * 80)
    for agg in aggregated:
        med = agg["delta_statistics"]["median_ms"]
        med_str = f"{med:.1f} ms" if not np.isnan(med) else "N/A"
        print(
            f"{agg['model']:<15} {agg['method']:<22} {agg['device']:<8} "
            f"{agg['precision']:>6.3f} {agg['recall']:>6.3f} {agg['f1']:>6.3f} {med_str:>10}"
        )


def resolve_output_path(args: argparse.Namespace, run_idx: int, seed: int, total_runs: int) -> Path:
    """Dataset- and run-aware output path that never overwrites across datasets/runs.

    A single run with an explicit --output keeps that exact path for backward
    compatibility; otherwise files are named per dataset, run, and seed.
    """
    if args.output is not None and total_runs == 1:
        return args.output
    return args.out_dir / f"pick_quality_{args.dataset}_run{run_idx}_seed{seed}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Comprehensive pick quality analysis")
    parser.add_argument("--n-traces", type=int, default=50, help="Traces per dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="stead",
        choices=["stead", "txed"],
        help="SeisBench dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Explicit output JSON path (only honored for a single run). "
            "If omitted, files are written to --out-dir as "
            "pick_quality_<dataset>_run<k>_seed<seed>.json so datasets and runs never overwrite."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Directory for auto-named per-dataset/per-run output files",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs; each run uses seed=--seed+(run-1) for an independent trace sample",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=["cpu", "cuda:0"],
        help="Devices to test",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="'all' or comma-separated model labels (pn, pnl, eqt, eqt-nc)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=6000,
        help="P-centered raw window length (matches matrix benchmark)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.runs < 1:
        print("--runs must be >= 1", file=sys.stderr)
        return 1
    if args.output is not None and args.runs > 1:
        print(
            "Note: --output is ignored when --runs > 1; using --out-dir naming so runs do not overwrite.",
            file=sys.stderr,
        )

    print("Pick Quality Analysis")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"N traces: {args.n_traces}")
    print(f"Devices: {args.devices}")
    print(f"Runs: {args.runs} (base seed {args.seed})")
    print()

    cuda_available = torch.cuda.is_available()
    devices = [d for d in args.devices if d == "cpu" or (cuda_available and "cuda" in d)]
    print(f"Available devices: {devices}")

    try:
        models = parse_models_arg(args.models)
    except ValueError as exc:
        print(f"Error parsing --models: {exc}", file=sys.stderr)
        return 1

    print(f"Models: {[m.label for m in models]}")

    try:
        ds = load_dataset(args.dataset)
    except Exception as exc:
        print(f"Error loading dataset: {exc}", file=sys.stderr)
        return 1

    print(f"Dataset loaded: {len(ds)} traces")
    p_col, s_col = catalog_pick_columns(ds)
    print(f"Catalog columns: P={p_col}, S={s_col}")

    mask = catalog_mask(ds, require_s=False)
    print(f"Traces with valid P picks: {int(mask.sum())}")

    for k in range(args.runs):
        seed = args.seed + k
        out_path = resolve_output_path(args, run_idx=k + 1, seed=seed, total_runs=args.runs)
        print("\n" + "#" * 60)
        print(f"# RUN {k + 1}/{args.runs}  dataset={args.dataset}  seed={seed}")
        print("#" * 60)
        run_once(
            ds=ds,
            models=models,
            devices=devices,
            args=args,
            seed=seed,
            output_path=out_path,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
