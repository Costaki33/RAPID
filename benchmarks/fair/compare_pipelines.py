#!/usr/bin/env python3
"""Apples-to-apples comparison: annotate() vs lean PyTorch.

For each model and station count, runs both paths on the same input traces and
records:
  - wall_time_s         end-to-end (matches existing matrix metric)
  - forward_only_s      time spent in model.forward() (via a forward hook)
  - preprocess_s        time outside the forward pass (wall - forward)

Forward hooks are attached to the same model instance used by annotate() and to
the lean backend's underlying model, so both numbers come from identical kernel
launches on identical input shapes.

Usage:
    python benchmarks/fair/compare_pipelines.py --dataset stead --n-traces 3 --n-stations 64 256 580
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import seisbench.models as sbm

from rapid.backends.baseline import BaselineAnnotate
from rapid.backends.lean_pytorch import LeanPyTorchBackend
from rapid.data import (
    WindowSpec,
    build_megabatch,
    preprocess_for_model,
    stream_to_3c_array,
)
from rapid.seisbench_matrix import _cut_raw_window, _dup_streams
from rapid.seisbench_precision_eval import (
    catalog_mask,
    catalog_pick_columns,
    load_dataset,
    preprocess_array,
    waves_to_stream,
)


MODELS = [
    ("PhaseNet", "original", "PhaseNet"),
    ("PhaseNetLight", "stead", "PhaseNetLight"),
    ("EQTransformer", "original", "EQTransformer"),
    ("EQTransformer", "original_nonconservative", "EQT-NC"),
]


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


class ForwardTimer:
    """Attach a pre-forward and post-forward hook to a torch.nn.Module to time
    each call. ``total_s`` is the sum across all calls during the measured run.
    """

    def __init__(self, module: torch.nn.Module, device: str):
        self.module = module
        self.device = device
        self.use_cuda = device.startswith("cuda")
        self.total_s: float = 0.0
        self.n_calls: int = 0
        self._t0: Optional[float] = None
        self._pre_handle = None
        self._post_handle = None

    def _pre(self, module, inputs):
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        self._t0 = time.perf_counter()

    def _post(self, module, inputs, output):
        if self.use_cuda:
            torch.cuda.synchronize(self.device)
        if self._t0 is not None:
            self.total_s += time.perf_counter() - self._t0
            self.n_calls += 1
            self._t0 = None

    def __enter__(self):
        self.total_s = 0.0
        self.n_calls = 0
        self._pre_handle = self.module.register_forward_pre_hook(self._pre)
        self._post_handle = self.module.register_forward_hook(self._post)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._pre_handle is not None:
            self._pre_handle.remove()
        if self._post_handle is not None:
            self._post_handle.remove()


def gather_traces(
    dataset_name: str,
    n_traces: int,
    seed: int,
) -> List[Tuple[int, np.ndarray, Dict[str, Any]]]:
    ds = load_dataset(dataset_name)
    p_col, _ = catalog_pick_columns(ds)
    mask = catalog_mask(ds, require_s=False)
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return []
    rng = np.random.default_rng(seed)
    take = min(n_traces, idxs.size)
    chosen = rng.choice(idxs, size=take, replace=False)

    out: List[Tuple[int, np.ndarray, Dict[str, Any]]] = []
    for row in chosen:
        try:
            waves, meta = ds.get_sample(int(row), sampling_rate=100.0)
        except Exception:
            continue
        p_cat = _finite_int(meta.get(p_col))
        if p_cat is None or not (0 <= p_cat < waves.shape[1]):
            continue
        if waves.ndim != 2:
            continue
        out.append((int(row), waves, meta))
    return out


def run_one(
    parent: str,
    child: str,
    label: str,
    waves: np.ndarray,
    meta: Dict[str, Any],
    n_stations: int,
    dtype: str,
    device: str,
    batch_size: int,
    n_samples: int,
    warmup_iters: int = 1,
) -> Dict[str, Any]:
    sb_model = getattr(sbm, parent).from_pretrained(child)
    sr = float(sb_model.sampling_rate)
    in_samples = int(sb_model.in_samples)
    co = str(meta.get("trace_component_order") or "ZNE")
    p_col, _ = catalog_pick_columns(load_dataset_cached(meta))
    p_cat = _finite_int(meta.get(p_col))

    waves_win, _, _ = _cut_raw_window(waves, n_samples=n_samples, p_sample=p_cat)
    raw_stream = waves_to_stream(waves_win, sr, co)
    streams_n = _dup_streams(raw_stream, n_stations, trace_row=0)

    # -------- BASELINE annotate() --------
    bl = BaselineAnnotate(parent, child, device=device, dtype="fp32")
    bl.load()

    # Warmup baseline
    for _ in range(warmup_iters):
        merged = streams_n[0][1].copy()
        for _, s in streams_n[1:2]:
            merged += s
        _ = bl.annotate_stream(merged)

    merged = streams_n[0][1].copy()
    for _, s in streams_n[1:]:
        merged += s

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with ForwardTimer(bl._model, device) as ft_bl:
        _ = bl.annotate_stream(merged)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    wall_bl = time.perf_counter() - t0
    fwd_bl = ft_bl.total_s
    n_calls_bl = ft_bl.n_calls
    bl.close()

    # -------- LEAN pytorch --------
    ln = LeanPyTorchBackend(parent, child, device=device, dtype=dtype, compile=False)
    ln.load()

    # Warmup lean
    for _ in range(warmup_iters):
        dummy = np.zeros((min(batch_size, 8), 3, in_samples), dtype=np.float32)
        _ = ln.infer_batch(dummy)

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with ForwardTimer(ln._fwd_model, device) as ft_ln:
        # Replicate run_lean_single stages
        # 1) Preprocess per station
        t_pre = time.perf_counter()
        arrays: List[Tuple[str, np.ndarray]] = []
        argdict = {"sampling_rate": sb_model.sampling_rate}
        for sta, st in streams_n:
            pre = preprocess_for_model(sb_model, st, argdict=argdict)
            arr = stream_to_3c_array(
                pre,
                component_order=getattr(sb_model, "component_order", None) or "ZNE",
            )
            if arr is None:
                continue
            arrays.append((sta, arr))
        t_pre_end = time.perf_counter()

        # 2) Build megabatch
        spec = WindowSpec(in_samples=in_samples, overlap_samples=0)
        mb = build_megabatch(arrays, spec)
        t_mb_end = time.perf_counter()

        # 3) Forward (chunked)
        _ = ln.infer_chunked(mb.windows, batch_size=batch_size)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
    wall_ln = time.perf_counter() - t0
    fwd_ln = ft_ln.total_s
    n_calls_ln = ft_ln.n_calls
    ln.close()

    return {
        "model_label": label,
        "n_stations": int(n_stations),
        "dtype": dtype,
        "batch_size": int(batch_size),
        "n_windows": int(mb.total_windows),
        "baseline": {
            "wall_s": wall_bl,
            "forward_only_s": fwd_bl,
            "pipeline_overhead_s": max(wall_bl - fwd_bl, 0.0),
            "n_forward_calls": n_calls_bl,
        },
        "lean": {
            "wall_s": wall_ln,
            "forward_only_s": fwd_ln,
            "preprocess_s": t_pre_end - t_pre,
            "megabatch_s": t_mb_end - t_pre_end,
            "pipeline_overhead_s": max(wall_ln - fwd_ln, 0.0),
            "n_forward_calls": n_calls_ln,
        },
    }


_DATASET_CACHE: Dict[int, Any] = {}


def load_dataset_cached(meta: Dict[str, Any]):
    # We don't have the dataset name in meta; just keep a single global cache.
    # The script knows its dataset_name from CLI and creates a fresh dataset per
    # invocation, so we cache once.
    key = id(meta)
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]
    # If empty, we need an existing dataset object; the caller stores it.
    return _CACHED_DS[0]


_CACHED_DS: List[Any] = [None]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="stead")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-traces", type=int, default=3)
    ap.add_argument(
        "--n-stations", type=int, nargs="+", default=[64, 256, 580]
    )
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-samples", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model labels to test (default: all)",
    )
    ap.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16"],
        help="Lean dtypes to compare (annotate() is always FP32)",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Optional JSONL output path",
    )
    args = ap.parse_args()

    # Cache one dataset object for catalog_pick_columns inside run_one.
    _CACHED_DS[0] = load_dataset(args.dataset)

    models = MODELS
    if args.models:
        wanted = set(args.models)
        models = [m for m in MODELS if m[2] in wanted or m[0] in wanted]

    traces = gather_traces(args.dataset, args.n_traces, args.seed)
    if not traces:
        print(f"No valid traces found in {args.dataset}")
        return

    print("=" * 78)
    print(
        f"Pipeline comparison: dataset={args.dataset}  device={args.device}  "
        f"batch_size={args.batch_size}  n_traces={len(traces)}"
    )
    print("=" * 78)

    out_f = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.output, "a")

    for parent, child, label in models:
        for dtype in args.dtypes:
            if parent == "EQTransformer" and dtype == "fp16":
                print(f"\n[SKIP] {label} fp16 (sentinel overflow)\n")
                continue

            print(
                f"\n{'-' * 78}\n{label}  dtype={dtype}  parent={parent}  child={child}\n{'-' * 78}"
            )
            print(
                f"{'n_st':>5} | {'wall_bl':>9} {'fwd_bl':>9} {'over_bl':>9} | "
                f"{'wall_ln':>9} {'fwd_ln':>9} {'pre_ln':>9} {'mb_ln':>9} | "
                f"{'speedup_wall':>12} {'speedup_fwd':>11}"
            )

            for n_st in args.n_stations:
                # Average over traces
                rows = []
                for row, waves, meta in traces:
                    try:
                        r = run_one(
                            parent,
                            child,
                            label,
                            waves,
                            meta,
                            n_stations=n_st,
                            dtype=dtype,
                            device=args.device,
                            batch_size=args.batch_size,
                            n_samples=args.n_samples,
                        )
                        r["trace_row"] = row
                        r["dataset"] = args.dataset
                        rows.append(r)
                        if out_f is not None:
                            out_f.write(json.dumps(r) + "\n")
                            out_f.flush()
                    except Exception as e:
                        print(f"   ERR @ n_st={n_st} row={row}: {e}")

                if not rows:
                    continue

                def med(key1, key2):
                    vals = [r[key1][key2] for r in rows]
                    return sorted(vals)[len(vals) // 2]

                wall_bl = med("baseline", "wall_s")
                fwd_bl = med("baseline", "forward_only_s")
                over_bl = med("baseline", "pipeline_overhead_s")
                wall_ln = med("lean", "wall_s")
                fwd_ln = med("lean", "forward_only_s")
                pre_ln = med("lean", "preprocess_s")
                mb_ln = med("lean", "megabatch_s")
                sp_wall = wall_bl / wall_ln if wall_ln > 0 else float("nan")
                sp_fwd = fwd_bl / fwd_ln if fwd_ln > 0 else float("nan")

                print(
                    f"{n_st:>5} | {wall_bl:>9.4f} {fwd_bl:>9.4f} {over_bl:>9.4f} | "
                    f"{wall_ln:>9.4f} {fwd_ln:>9.4f} {pre_ln:>9.4f} {mb_ln:>9.4f} | "
                    f"{sp_wall:>11.2f}x {sp_fwd:>10.2f}x"
                )

    if out_f is not None:
        out_f.close()
    print()
    print("Legend:")
    print("  wall_*  = end-to-end wall time")
    print("  fwd_*   = time spent in model.forward() only (hooked)")
    print("  over_bl = baseline pipeline overhead (wall - forward)")
    print("  pre_ln  = lean per-station preprocess loop")
    print("  mb_ln   = lean megabatch build")
    print("  speedup_wall = baseline_wall / lean_wall (>1 means lean is faster)")
    print("  speedup_fwd  = baseline_fwd  / lean_fwd  (apples-to-apples GPU)")


if __name__ == "__main__":
    main()
