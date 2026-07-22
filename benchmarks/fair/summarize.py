"""Print a quick numerical summary of a matrix JSONL.

Examples::

    python benchmarks/fair/summarize.py --jsonl results/matrix.jsonl
    python benchmarks/fair/summarize.py --jsonl results/matrix.jsonl --n-stations 228
    python benchmarks/fair/summarize.py --jsonl results/matrix.jsonl --model PhaseNet --top 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--model", default=None, help="Filter to this model_label.")
    p.add_argument("--n-stations", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--top", type=int, default=25)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from rapid.analysis import (
        best_variant_per_group,
        cpu_worker_knee,
        env_rows,
        load_results,
        speedup_vs_baseline,
    )

    df = load_results(args.jsonl)
    if args.model:
        df = df[df["model_label"] == args.model]
    if args.n_stations is not None:
        df = df[df["n_stations"] == args.n_stations]
    if args.device:
        df = df[df["device"] == args.device]

    n_env = len(env_rows(args.jsonl))
    n_err = int((df["kind"] == "error").sum())
    n_ok = int(len(df) - n_err)
    print(f"Loaded {len(df)} rows from {args.jsonl}  (ok={n_ok}, errors={n_err}, env={n_env})")

    print("\n=== Speedup vs baseline (median wall time) ===")
    sp = speedup_vs_baseline(df)
    if not sp.empty:
        cols = [
            "model_label", "n_stations", "device", "kind", "variant",
            "wall_time_s_baseline", "wall_time_s_median", "wall_time_s_min",
            "speedup_median", "speedup_best", "n_repeats",
        ]
        cols = [c for c in cols if c in sp.columns]
        print(sp[cols].head(args.top).to_string(index=False))

    print("\n=== Best variant per (model, n_stations, device) ===")
    bv = best_variant_per_group(df)
    if not bv.empty:
        print(bv.head(args.top).to_string(index=False))

    print("\n=== CPU-worker knee (best n_cpu_workers per group) ===")
    knee = cpu_worker_knee(df)
    if not knee.empty:
        print(knee.head(args.top).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
