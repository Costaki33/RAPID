#!/usr/bin/env python3
"""Benchmark process-based dual-GPU runner.

Loads traces the same way as ``rapid.seisbench_matrix`` (get_sample, P-centered
window, preprocess_array, duplicate streams) so results match the main matrix.

NOTE: Each trial spawns two worker processes and reloads weights. Wall time
includes that overhead. GPU-only times are in gpu0_time_s and gpu1_time_s.

Usage:
    python scripts/benchmark_dual_gpu_process.py --output results/dual_gpu_process.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import seisbench.models as sbm

from rapid.data import WindowSpec, build_megabatch, preprocess_for_model, stream_to_3c_array
from rapid.runners.dual_gpu_process import run_dual_gpu_process
from rapid.seisbench_matrix import _cut_raw_window, _dup_streams
from rapid.seisbench_precision_eval import (
    catalog_mask,
    catalog_pick_columns,
    cut_window,
    load_dataset,
    preprocess_array,
    waves_to_stream,
)

MODELS: List[Tuple[str, str]] = [
    ("PhaseNet", "original"),
    ("PhaseNetLight", "stead"),
    ("EQTransformer", "original"),
    ("EQTransformer", "original_nonconservative"),
]

DATASETS = ["stead", "txed"]
N_STATIONS_LIST = [64, 256, 580]
BATCH_SIZES = [256, 512]
N_TRACES = 50
N_REPEATS = 2


def _finite_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(round(v))


def get_model_label(parent: str, child: str) -> str:
    if parent == "EQTransformer" and child == "original_nonconservative":
        return "EQT-NC"
    return parent


def run_single_trial(
    parent_model: str,
    child_model: str,
    dtype: str,
    compile_model: bool,
    dataset_name: str,
    trace_row: int,
    n_stations: int,
    batch_size: int,
    repeat: int,
    n_samples: int,
) -> dict:
    """Run a single trial. ``trace_row`` is a SeisBench dataset row index."""
    ds = load_dataset(dataset_name)

    model_cls = getattr(sbm, parent_model)
    sb_model = model_cls.from_pretrained(child_model)
    sr = float(sb_model.sampling_rate)
    in_samples = int(sb_model.in_samples)

    p_col, s_col = catalog_pick_columns(ds)
    try:
        waves, meta = ds.get_sample(trace_row, sampling_rate=sr)
    except Exception as e:
        return {"status": "error", "error": f"get_sample: {e}"}

    co = str(meta.get("trace_component_order") or "ZNE")
    if waves.ndim != 2:
        return {"status": "error", "error": "waves.ndim != 2"}

    p_cat = _finite_int(meta.get(p_col))
    if p_cat is None or not (0 <= p_cat < waves.shape[1]):
        return {"status": "error", "error": "invalid catalog P sample"}

    try:
        waves_win, _raw_start, p_idx_in_win = _cut_raw_window(
            waves, n_samples=n_samples, p_sample=p_cat
        )
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    arr_full = preprocess_array(sb_model, waves_win, sr, co)
    if arr_full is None:
        return {"status": "error", "error": "preprocess_array failed"}

    t_raw = int(waves_win.shape[1])
    t_pp = int(arr_full.shape[1])
    if t_pp != t_raw:
        return {
            "status": "error",
            "error": f"preprocess length {t_pp} != raw window {t_raw}",
        }

    try:
        _win, _model_start, _p_mod = cut_window(
            arr_full, in_samples, int(p_idx_in_win)
        )
    except ValueError as e:
        return {"status": "error", "error": str(e)}

    raw_stream = waves_to_stream(waves_win, sr, co)
    streams_n = _dup_streams(raw_stream, n_stations, trace_row)

    argdict = {"sampling_rate": sb_model.sampling_rate}
    arrays: List[Tuple[str, np.ndarray]] = []
    for sta, st in streams_n:
        pre = preprocess_for_model(sb_model, st, argdict=argdict)
        arr = stream_to_3c_array(
            pre, component_order=getattr(sb_model, "component_order", None) or "ZNE"
        )
        if arr is None:
            return {"status": "error", "error": "stream_to_3c_array failed"}
        arrays.append((sta, arr))

    spec = WindowSpec(in_samples=in_samples, overlap_samples=0)
    megabatch = build_megabatch(arrays, spec)

    if parent_model == "EQTransformer" and dtype == "fp16":
        return {
            "status": "skipped",
            "reason": "EQTransformer cannot use FP16 (sentinel overflow)",
        }

    t0 = time.perf_counter()
    result = run_dual_gpu_process(
        parent_model=parent_model,
        child_model=child_model,
        dtype=dtype,
        megabatch=megabatch,
        batch_size=batch_size,
        compile_model=compile_model,
        devices=("cuda:0", "cuda:1"),
        warmup_iters=2,
    )
    total_time = time.perf_counter() - t0

    return {
        "status": "success",
        "sb_trace_row": trace_row,
        "wall_time_s": result.wall_time_s,
        "gpu0_time_s": result.gpu0_time_s,
        "gpu1_time_s": result.gpu1_time_s,
        "total_time_s": total_time,
        "n_windows": result.n_windows,
        "extra": result.extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=str,
        default="results/dual_gpu_process.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=N_TRACES,
        help="Number of traces per dataset",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=6000,
        help="P-centered raw window length (same as matrix n_samples)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for trace subsampling",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Models to test (default: all)",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        nargs="+",
        default=["bf16"],
        help="Data types to test",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also test with torch.compile",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    models = MODELS
    if args.models:
        models = [
            (p, c)
            for p, c in MODELS
            if p in args.models or get_model_label(p, c) in args.models
        ]

    dtypes = args.dtypes
    compile_flags = [False]
    if args.compile:
        compile_flags = [False, True]

    rng = np.random.default_rng(args.seed)
    dataset_traces: Dict[str, List[int]] = {}
    for dname in DATASETS:
        try:
            ds = load_dataset(dname)
        except Exception as exc:
            print(f"[skip] dataset {dname!r}: {exc}", flush=True)
            dataset_traces[dname] = []
            continue
        mask = catalog_mask(ds, require_s=False)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            dataset_traces[dname] = []
            continue
        take = min(args.n_traces, idxs.size)
        chosen = rng.choice(idxs, size=take, replace=False)
        dataset_traces[dname] = [int(x) for x in chosen]

    n_trace_lists = sum(1 for v in dataset_traces.values() if v)
    if n_trace_lists == 0:
        print("No valid traces in any dataset; exit.", flush=True)
        return

    total_trials = (
        len(models)
        * n_trace_lists
        * len(N_STATIONS_LIST)
        * len(BATCH_SIZES)
        * len(dtypes)
        * len(compile_flags)
        * args.n_traces
        * N_REPEATS
    )

    print("=" * 70)
    print("Process-based Dual-GPU Benchmark")
    print("=" * 70)
    print(f"Models: {[get_model_label(p, c) for p, c in models]}")
    print(f"Datasets: {DATASETS}")
    print(f"n_samples (raw window): {args.n_samples}")
    print(f"Station counts: {N_STATIONS_LIST}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Dtypes: {dtypes}")
    print(f"Compile: {compile_flags}")
    print(f"Traces per dataset (cap): {args.n_traces}")
    print(f"Repeats: {N_REPEATS}")
    print(f"Approx trials upper bound: {total_trials}")
    print(f"Output: {output_path}")
    print("=" * 70)

    completed = 0
    errors = 0
    skipped = 0

    with open(output_path, "a") as f:
        for parent, child in models:
            label = get_model_label(parent, child)

            for dataset_name in DATASETS:
                trace_rows = dataset_traces.get(dataset_name) or []
                if not trace_rows:
                    continue

                for trace_row in trace_rows:
                    for n_stations in N_STATIONS_LIST:
                        for batch_size in BATCH_SIZES:
                            for dtype in dtypes:
                                for compile_model in compile_flags:
                                    for repeat in range(N_REPEATS):
                                        trial_key = (
                                            f"{label}/{dataset_name}/row{trace_row}/"
                                            f"n{n_stations}/bs{batch_size}/{dtype}/"
                                            f"compile={compile_model}/rep{repeat}"
                                        )

                                        row: Dict[str, Any] = {
                                            "kind": "dual_gpu_process",
                                            "model_parent": parent,
                                            "model_child": child,
                                            "model_label": label,
                                            "dataset": dataset_name,
                                            "sb_trace_row": trace_row,
                                            "n_samples_raw": args.n_samples,
                                            "n_stations": n_stations,
                                            "batch_size": batch_size,
                                            "dtype": dtype,
                                            "compile": compile_model,
                                            "repeat": repeat,
                                            "timestamp_s": time.time(),
                                        }

                                        try:
                                            result = run_single_trial(
                                                parent_model=parent,
                                                child_model=child,
                                                dtype=dtype,
                                                compile_model=compile_model,
                                                dataset_name=dataset_name,
                                                trace_row=trace_row,
                                                n_stations=n_stations,
                                                batch_size=batch_size,
                                                repeat=repeat,
                                                n_samples=args.n_samples,
                                            )
                                            row.update(result)

                                            if result.get("status") == "skipped":
                                                skipped += 1
                                                status = "SKIP"
                                            elif result.get("status") == "error":
                                                errors += 1
                                                status = "ERR"
                                            else:
                                                completed += 1
                                                wt = float(result["wall_time_s"])
                                                status = f"{wt:.3f}s"

                                        except Exception as e:
                                            row["status"] = "error"
                                            row["error"] = str(e)
                                            row["traceback"] = traceback.format_exc()
                                            errors += 1
                                            status = "ERR"

                                        f.write(json.dumps(row) + "\n")
                                        f.flush()

                                        print(
                                            f"[{completed+errors+skipped}] {trial_key}: {status}",
                                            flush=True,
                                        )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Completed: {completed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
