"""FP16 vs FP32 probability/pick-drift comparison on a single backend.

Runs the lean PyTorch backend once in FP32 and once in FP16 on the same
station list, collects the raw predictions, and computes:

    - probability-trace drift (MAE, max abs err, RMSE, Pearson)
    - pick-time drift at a threshold (median / p95 / max delta in samples)

This is the placeholder for the final 100-event validation; when the manual
picks arrive, swap in their ground truth and extend it to report per-event
pick-time errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--child", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-stations", type=int, default=228)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pick-threshold", type=float, default=0.3)
    ap.add_argument("--pick-channel", type=int, default=1,
                   help="Channel index in (B,T,C) post-processed preds to compare picks on.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    from rapid.backends import get_backend
    from rapid.data import load_all_streams, select_stations
    from rapid.runners.single_gpu import run_lean_single
    from rapid.quality import (
        compare_probabilities, as_dict, pick_time_drift_samples,
    )

    stations = select_stations(args.dataset_dir, args.n_stations)
    streams = load_all_streams(args.dataset_dir, stations)

    cls = get_backend("lean_pytorch")

    results = {}
    preds = {}
    for dtype in ("fp32", "fp16"):
        bk = cls(parent_model=args.model, child_model=args.child,
                 device=args.device, dtype=dtype)
        bk.load()
        t0 = time.perf_counter()
        res = run_lean_single(bk, streams, batch_size=args.batch_size, warmup_iters=1)
        wall = time.perf_counter() - t0
        bk.close()
        results[dtype] = {
            "wall_s": wall,
            "total_s": res.total_s,
            "stage_times_s": res.stage_times,
            "n_windows": res.n_windows,
        }
        preds[dtype] = res.predictions

    prob_stats = compare_probabilities(preds["fp32"], preds["fp16"])
    pick_stats = pick_time_drift_samples(
        preds["fp32"], preds["fp16"],
        channel=args.pick_channel, threshold=args.pick_threshold,
    )

    speedup = results["fp32"]["total_s"] / max(results["fp16"]["total_s"], 1e-9)

    summary = {
        "model": f"{args.model}/{args.child}",
        "device": args.device,
        "n_stations": args.n_stations,
        "batch_size": args.batch_size,
        "fp32": results["fp32"],
        "fp16": results["fp16"],
        "speedup_fp16_over_fp32": speedup,
        "probability_drift": as_dict(prob_stats),
        "pick_time_drift": pick_stats,
    }
    print(json.dumps(summary, indent=2))
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
