"""CLI for the pipelined runners.

Examples::

    # Single GPU, 8 CPU preprocess workers, FP16 megabatch.
    python scripts/run_pipelined.py \\
        --dataset-dir "$DATA_DIR" --model PhaseNet --child original \\
        --n-stations 228 --batch-size 512 --dtype fp16 \\
        --mode single_gpu --n-cpu-workers 8

    # Dual GPU, 8 CPU workers per GPU.
    python scripts/run_pipelined.py \\
        --dataset-dir "$DATA_DIR" --model PhaseNet --child original \\
        --n-stations 580 --batch-size 512 --dtype fp16 \\
        --mode dual_gpu --n-cpu-workers 8

    # Baseline annotate() on 2 GPUs (fair comparison for dual_gpu above).
    python scripts/run_pipelined.py \\
        --dataset-dir "$DATA_DIR" --model PhaseNet --child original \\
        --n-stations 580 --mode baseline_dual_gpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--child", required=True)
    p.add_argument("--n-stations", type=int, default=228)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--overlap-samples", type=int, default=0)
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--backend", default="lean_pytorch")
    p.add_argument(
        "--mode",
        choices=["single_gpu", "dual_gpu", "baseline_dual_gpu", "baseline_single_gpu"],
        default="single_gpu",
    )
    p.add_argument("--n-cpu-workers", type=int, default=8,
                   help="Per GPU for dual_gpu; absolute for single_gpu.")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out-json", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from rapid.data import load_all_streams, select_stations
    from rapid.runners.pipelined import (
        run_baseline_dual_gpu,
        run_pipelined_dual_gpu,
        run_pipelined_single_gpu,
    )
    from rapid.runners.single_gpu import run_baseline_single
    from rapid.backends.baseline import BaselineAnnotate

    print(f"Loading {args.n_stations} stations ...", flush=True)
    t = time.perf_counter()
    stations = select_stations(args.dataset_dir, args.n_stations)
    streams = load_all_streams(args.dataset_dir, stations)
    print(f"  loaded {len(streams)} stations in {time.perf_counter()-t:.2f} s")

    rows = []
    for repeat in range(args.repeats):
        t0 = time.perf_counter()
        if args.mode == "single_gpu":
            res = run_pipelined_single_gpu(
                parent_model=args.model, child_model=args.child,
                streams=streams, n_cpu_workers=args.n_cpu_workers,
                batch_size=args.batch_size, overlap_samples=args.overlap_samples,
                dtype=args.dtype, backend_name=args.backend,
            )
            row: Dict[str, Any] = dict(
                mode="single_gpu",
                wall_time_s=res.wall_time_s,
                end_to_end_wall_s=res.end_to_end_wall_s,
                gpu_forward_s=res.per_gpu[0]["gpu_forward_s"],
                gpu_idle_s=res.per_gpu[0]["gpu_idle_s"],
                preprocess_total_s=res.per_gpu[0]["preprocess_total_s"],
                gpu_utilization_pct=res.gpu_utilization_pct(),
                n_stations=res.sum_stations,
                n_windows=res.sum_windows,
                n_gpu_submits=res.per_gpu[0]["n_gpu_submits"],
            )
        elif args.mode == "dual_gpu":
            res = run_pipelined_dual_gpu(
                parent_model=args.model, child_model=args.child,
                streams=streams, n_cpu_workers_per_gpu=args.n_cpu_workers,
                batch_size=args.batch_size, overlap_samples=args.overlap_samples,
                dtype=args.dtype, backend_name=args.backend, num_gpus=2,
            )
            row = dict(
                mode="dual_gpu",
                wall_time_s=res.wall_time_s,
                end_to_end_wall_s=res.end_to_end_wall_s,
                per_gpu=res.per_gpu,
                gpu_utilization_pct=res.gpu_utilization_pct(),
                n_stations=res.sum_stations,
                n_windows=res.sum_windows,
            )
        elif args.mode == "baseline_dual_gpu":
            res = run_baseline_dual_gpu(
                parent_model=args.model, child_model=args.child,
                streams=streams, num_gpus=2,
            )
            row = dict(
                mode="baseline_dual_gpu",
                wall_time_s=res.wall_time_s,
                end_to_end_wall_s=res.end_to_end_wall_s,
                per_gpu=res.per_gpu,
                n_stations=res.sum_stations,
            )
        else:  # baseline_single_gpu
            bk = BaselineAnnotate(
                parent_model=args.model, child_model=args.child,
                device="cuda:0", dtype="fp32",
            )
            bk.load()
            r = run_baseline_single(bk, streams)
            bk.close()
            row = dict(
                mode="baseline_single_gpu",
                wall_time_s=r.total_s,
                stage_times_s=r.stage_times,
                n_stations=r.n_stations,
            )

        row.update(
            repeat=repeat,
            model=f"{args.model}/{args.child}",
            n_stations_requested=args.n_stations,
            batch_size=args.batch_size,
            overlap_samples=args.overlap_samples,
            dtype=args.dtype,
            backend=args.backend,
            n_cpu_workers=args.n_cpu_workers,
        )
        print(json.dumps(row, indent=2), flush=True)
        rows.append(row)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
