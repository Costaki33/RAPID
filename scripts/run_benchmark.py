"""Run a single RAPID benchmark configuration.

Useful for ad-hoc checks without touching the full matrix runner. Example::

    python scripts/run_benchmark.py \\
        --model PhaseNet --child original \\
        --backend lean_pytorch --dtype fp16 \\
        --device cuda:0 --n-stations 228 \\
        --batch-size 256 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a single RAPID benchmark.")
    p.add_argument("--dataset-dir", required=True,
                   help="Path to the timechunk directory containing <station>/*.mseed.")
    p.add_argument("--model", required=True,
                   help="SeisBench parent model class, e.g. PhaseNet, EQTransformer.")
    p.add_argument("--child", required=True,
                   help="Pretrained weights name, e.g. 'original', 'stead'.")
    p.add_argument("--backend", default="lean_pytorch",
                   help="Backend name (baseline_annotate, lean_pytorch, onnx, tensorrt).")
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-stations", type=int, default=228)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--overlap-samples", type=int, default=0)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument("--compile", action="store_true",
                   help="Apply torch.compile (lean_pytorch only).")
    p.add_argument("--onnx-path", default=None)
    p.add_argument("--engine-path", default=None)
    p.add_argument("--out-json", default=None,
                   help="If provided, write a JSON summary to this path.")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    from rapid.backends import get_backend, available_backends
    from rapid.data import load_all_streams, select_stations
    from rapid.runners.single_gpu import run_baseline_single, run_lean_single

    if args.backend not in available_backends():
        print(f"Backend {args.backend!r} not available. Have: {available_backends()}",
              file=sys.stderr)
        return 2

    stations = select_stations(args.dataset_dir, args.n_stations)
    print(f"Loading {len(stations)} stations from {args.dataset_dir} ...")
    t0 = time.perf_counter()
    streams = load_all_streams(args.dataset_dir, stations)
    print(f"  loaded {len(streams)} stations in {time.perf_counter()-t0:.2f} s")

    cls = get_backend(args.backend)
    init_kwargs: Dict[str, Any] = dict(
        parent_model=args.model, child_model=args.child,
        device=args.device, dtype=args.dtype,
    )
    if args.backend == "lean_pytorch" and args.compile:
        init_kwargs["compile"] = True
    if args.backend == "onnx" and args.onnx_path:
        init_kwargs["onnx_path"] = args.onnx_path
    if args.backend == "tensorrt" and args.engine_path:
        init_kwargs["engine_path"] = args.engine_path

    results: List[Dict[str, Any]] = []
    for repeat in range(args.repeats):
        bk = cls(**init_kwargs)
        bk.load()
        try:
            if args.backend == "baseline_annotate":
                res = run_baseline_single(bk, streams)
            else:
                res = run_lean_single(
                    bk, streams,
                    batch_size=args.batch_size,
                    overlap_samples=args.overlap_samples,
                    warmup_iters=args.warmup_iters,
                )
        finally:
            bk.close()

        row = dict(
            repeat=repeat,
            backend=args.backend,
            dtype=args.dtype,
            device=args.device,
            model=f"{args.model}/{args.child}",
            n_stations=res.n_stations,
            n_windows=res.n_windows,
            batch_size=args.batch_size,
            overlap_samples=args.overlap_samples,
            stage_times_s=res.stage_times,
            total_s=res.total_s,
            throughput_stations_per_s=res.throughput_stations_per_s,
        )
        print(json.dumps(row, indent=2))
        results.append(row)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
