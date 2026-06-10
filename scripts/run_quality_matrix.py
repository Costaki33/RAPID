"""Sweep-mode quality comparison: every non-reference backend vs FP32 reference.

Why this is separate from the timing matrix
-------------------------------------------
The speed sweep trims FP16 on CPU, cycles through batch sizes, runs
multiple repeats, etc. — none of which affects output quality (deterministic
model weights, deterministic preprocessing). Running quality comparisons
inside every timing cell would roughly double the sweep time for
information we only need once per (model, dtype) pair.

So we run timing and quality in **two complementary passes**:

1. ``scripts/run_matrix.py`` — timing / memory / telemetry (speed side).
2. ``scripts/run_quality_matrix.py`` — output-drift stats vs FP32 reference.

Output layout
-------------
Each row of ``results/quality.jsonl`` answers "how much does this
precision reduction change the output on this model at this workload?":

    {
      "model_label": "PhaseNet",
      "dtype": "fp16",
      "compile": false,
      "n_stations": 228,
      "ref_dtype": "fp32",
      "prob_mae": ..., "prob_rmse": ..., "prob_max_abs_err": ..., "prob_pearson": ...,
      "pick_mean_delta_samples": ..., "pick_median_delta_samples": ...,
      "pick_p95_abs_delta_samples": ..., "pick_max_abs_delta_samples": ...,
      "pick_n_pairs": ..., "pick_n_missing_ref": ..., "pick_n_missing_test": ...,
      ...
    }

Interpretation (at 100 Hz sampling):
- ``pick_p95_abs_delta_samples <= 3`` (≤30 ms at 100 Hz) is what we'd
  accept for deployment.
- ``prob_max_abs_err > ~1e-2`` starts to move picks appreciably.
- ``prob_pearson < 0.999`` means the precision is reshaping probabilities,
  not just scaling them — investigate before shipping.
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


# Default sweep axis: the same backends the timing matrix tests, minus
# the baseline (which is trivially identical to fp32 lean_pytorch for
# output quality purposes — same weights, same forward path).
DEFAULT_DTYPES = [
    {"dtype": "fp16", "compile": False},
    {"dtype": "bf16", "compile": False},
    {"dtype": "fp16", "compile": True},
    {"dtype": "bf16", "compile": True},
]

# EQTransformer has a hardcoded -1e10 sentinel in its encoder that
# overflows in FP16. The backend catches this early and raises an
# actionable error; we skip those cells here rather than emit error
# rows, same as the timing matrix does.
FP16_UNSUPPORTED_PARENTS = {"EQTransformer"}


def run_once(parent: str, child: str, dtype: str, compile: bool,
             device: str, streams, batch_size: int):
    from rapid.backends import get_backend
    from rapid.runners.single_gpu import run_lean_single

    cls = get_backend("lean_pytorch")
    bk = cls(parent_model=parent, child_model=child, device=device,
             dtype=dtype, compile=compile)
    bk.load()
    t0 = time.perf_counter()
    res = run_lean_single(bk, streams, batch_size=batch_size, warmup_iters=1)
    wall = time.perf_counter() - t0
    bk.close()
    return res.predictions, wall


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log = logging.getLogger("quality_matrix")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Matrix config with dataset_dir + models. "
                         "Reuses configs/full_matrix.json so we don't duplicate "
                         "model/dataset definitions.")
    ap.add_argument("--out-jsonl", default="results/quality.jsonl")
    ap.add_argument("--n-stations", type=int, default=228,
                    help="Workload size. 228 is enough to exercise every "
                         "model's pad/windowing; larger adds little new "
                         "information for quality purposes.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pick-threshold", type=float, default=0.3)
    ap.add_argument("--pick-channel", type=int, default=1,
                    help="(B,T,C) output channel used for pick extraction. "
                         "1 = P for SeisBench post-processed outputs.")
    ap.add_argument("--dtypes", type=str, default=None,
                    help="Comma-separated dtypes to test (e.g. fp16,bf16). "
                         "Default sweeps fp16, bf16, fp16+compile, bf16+compile.")
    ap.add_argument("--models", type=str, default=None,
                    help="Comma-separated model labels to test. Default: "
                         "every model in the config.")
    args = ap.parse_args()

    from rapid.data import load_all_streams, select_stations
    from rapid.quality import (as_dict, compare_probabilities,
                               pick_time_drift_samples)

    cfg = json.loads(Path(args.config).read_text())
    dataset_dir = cfg["dataset_dir"]
    models = cfg["models"]
    if args.models:
        wanted = set(m.strip() for m in args.models.split(","))
        models = [m for m in models if m.get("label", m["parent"]) in wanted]

    dtype_specs: List[Dict[str, Any]] = []
    if args.dtypes:
        for d in args.dtypes.split(","):
            d = d.strip()
            if d.endswith("+compile"):
                dtype_specs.append({"dtype": d[:-len("+compile")], "compile": True})
            else:
                dtype_specs.append({"dtype": d, "compile": False})
    else:
        dtype_specs = list(DEFAULT_DTYPES)

    stations = select_stations(dataset_dir, args.n_stations)
    streams = load_all_streams(dataset_dir, stations)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", encoding="utf-8") as fh:
        for m in models:
            parent, child = m["parent"], m["child"]
            label = m.get("label", parent)
            log.info("=== %s (%s/%s) ===", label, parent, child)

            # Reference: FP32 lean_pytorch. Any speedup vs FP32 is real
            # only if the corresponding quality row passes the
            # acceptance thresholds above.
            try:
                ref_preds, ref_wall = run_once(
                    parent, child, "fp32", False, args.device, streams,
                    args.batch_size,
                )
            except Exception as e:
                log.error("Reference FP32 failed for %s: %s", label, e)
                continue

            for spec in dtype_specs:
                dtype = spec["dtype"]
                compile_ = bool(spec.get("compile", False))
                if dtype == "fp16" and parent in FP16_UNSUPPORTED_PARENTS:
                    log.info("  skip: %s + fp16 unsupported (backend sentinel)", label)
                    continue
                tag = f"{dtype}{'+compile' if compile_ else ''}"
                log.info("  running %s …", tag)
                try:
                    test_preds, test_wall = run_once(
                        parent, child, dtype, compile_, args.device, streams,
                        args.batch_size,
                    )
                except Exception as e:
                    log.error("  %s failed: %s", tag, e)
                    fh.write(json.dumps({
                        "model_label": label, "model_parent": parent,
                        "model_child": child,
                        "dtype": dtype, "compile": compile_,
                        "ref_dtype": "fp32", "device": args.device,
                        "n_stations": args.n_stations,
                        "batch_size": args.batch_size,
                        "error": f"{type(e).__name__}: {e}",
                        "timestamp_s": time.time(),
                    }) + "\n")
                    fh.flush()
                    continue

                prob = as_dict(compare_probabilities(ref_preds, test_preds))
                pick = pick_time_drift_samples(
                    ref_preds, test_preds,
                    channel=args.pick_channel,
                    threshold=args.pick_threshold,
                )

                row = {
                    "model_label": label, "model_parent": parent,
                    "model_child": child,
                    "dtype": dtype, "compile": compile_,
                    "ref_dtype": "fp32", "device": args.device,
                    "n_stations": args.n_stations,
                    "batch_size": args.batch_size,
                    "ref_wall_s": ref_wall,
                    "test_wall_s": test_wall,
                    "speedup_vs_fp32": ref_wall / max(test_wall, 1e-9),
                    # Probability-drift stats (SeisBench post-processed).
                    "prob_mae": prob["mae"],
                    "prob_rmse": prob["rmse"],
                    "prob_max_abs_err": prob["max_abs_err"],
                    "prob_pearson": prob["pearson"],
                    "prob_n_samples": prob["n_samples"],
                    # Pick-time drift at args.pick_threshold. Reference
                    # picks are from FP32; test picks are from the
                    # precision-reduced / compiled path.
                    "pick_n_pairs": pick["n_pairs"],
                    "pick_n_missing_ref": pick.get("n_missing_fp32", 0),
                    "pick_n_missing_test": pick.get("n_missing_fp16", 0),
                    "pick_mean_delta_samples": pick["mean_delta_samples"],
                    "pick_median_delta_samples": pick["median_delta_samples"],
                    "pick_p95_abs_delta_samples": pick["p95_abs_delta_samples"],
                    "pick_max_abs_delta_samples": pick["max_abs_delta_samples"],
                    "pick_threshold": args.pick_threshold,
                    "pick_channel": args.pick_channel,
                    "timestamp_s": time.time(),
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                log.info("    prob RMSE=%.3e, pick p95=%.1f samples, n_pairs=%d",
                         row["prob_rmse"], row["pick_p95_abs_delta_samples"],
                         row["pick_n_pairs"])

    log.info("Quality matrix written to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
