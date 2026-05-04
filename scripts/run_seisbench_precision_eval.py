#!/usr/bin/env python3
"""Run SeisBench STEAD / TXED / GEOFON / ETHZ precision vs catalog picks.

On the lambda1a host, point SeisBench at the shared cache::

    export SEISBENCH_CACHE_ROOT=/lambda1a/seisbench

Datasets live under ``$SEISBENCH_CACHE_ROOT/datasets/<name>``. Default names are
``stead``, ``txed``, ``geofon``, ``ethz``. INSTANCE (``instancecounts``) is not
in the default list because of download size; pass ``--datasets instancecounts``
if you need it.

Example (GPU, 100 traces per dataset, optional FP16+torch.compile)::

    python scripts/run_seisbench_precision_eval.py \\
        --device cuda:0 \\
        --max-per-dataset 100 \\
        --out results/seisbench_precision_eval.jsonl \\
        --with-fp16-compile

Subset of models (or set env ``RAPID_PRECISION_MODELS=PhaseNet,EQTransformer``)::

    python scripts/run_seisbench_precision_eval.py --models pn,eqt

**Note:** This JSONL measures **pick / probability drift across dtypes**, not
**wall-time speedup**. Throughput speedup lives in ``results/matrix.jsonl`` /
the benchmark summary.

"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# package root: RAPID/
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rapid.seisbench_precision_eval import (  # noqa: E402
    list_model_choices,
    parse_models_arg,
    run_evaluation,
    write_jsonl,
)


def main() -> None:
    _models_default = os.environ.get("RAPID_PRECISION_MODELS", "all")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        default="stead,txed,geofon,ethz",
        help="Comma list: stead,txed,geofon,ethz (optional: instancecounts, …)",
    )
    ap.add_argument(
        "--models",
        default=_models_default,
        help=(
            "'all' or comma-separated labels / aliases (pn, pnl, eqt, eqt-nc). "
            "Default: env RAPID_PRECISION_MODELS if set, else all."
        ),
    )
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="Print available model names and exit.",
    )
    ap.add_argument("--max-per-dataset", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--dtypes",
        default="fp32,fp16,bf16",
        help="Lean dtypes to compare (fp32 is auto-prepended if omitted).",
    )
    ap.add_argument(
        "--require-both-ps",
        action="store_true",
        help="Only use traces with catalog P and S inside the record (stricter).",
    )
    ap.add_argument(
        "--with-fp16-compile",
        action="store_true",
        help="Also run an extra FP16 pass with torch.compile (PhaseNet family only; needs PyTorch 2+).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/seisbench_precision_eval.jsonl"),
    )
    args = ap.parse_args()

    if args.list_models:
        print(list_model_choices(), flush=True)
        raise SystemExit(0)

    cache_msg = os.environ.get("SEISBENCH_CACHE_ROOT", "(default ~/.seisbench)")
    print(f"SEISBENCH_CACHE_ROOT={cache_msg}", flush=True)

    ds_list = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    dtype_list = [x.strip() for x in args.dtypes.split(",") if x.strip()]

    try:
        models = parse_models_arg(args.models)
    except ValueError as exc:
        raise SystemExit(f"{exc}\n{list_model_choices()}") from exc

    rows = run_evaluation(
        datasets=ds_list,
        models=models,
        max_per_dataset=args.max_per_dataset,
        device=args.device,
        dtypes=dtype_list,
        require_both_ps=args.require_both_ps,
        seed=args.seed,
        include_fp16_compile=args.with_fp16_compile,
    )
    write_jsonl(args.out, rows)
    print(f"Wrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
