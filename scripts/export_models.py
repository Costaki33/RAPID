"""Export every model in the config to ONNX and (optionally) TensorRT engines.

Example::

    python scripts/export_models.py \\
        --onnx-dir models_exported/onnx \\
        --trt-dir  models_exported/trt \\
        --opt-batch 256 --max-batch 1024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


DEFAULT_MODELS = [
    ("PhaseNet", "original", 3001),
    ("PhaseNetLight", "stead", 3001),
    ("EQTransformer", "original", 6000),
    ("EQTransformer", "original_nonconservative", 6000),
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="models_exported/onnx")
    ap.add_argument("--trt-dir", default="models_exported/trt")
    ap.add_argument("--opt-batch", type=int, default=256)
    ap.add_argument("--max-batch", type=int, default=1024)
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--skip-trt", action="store_true",
                   help="Skip TensorRT engine build (useful on CPU-only boxes).")
    ap.add_argument("--models", nargs="*",
                   help="Filter to a subset of parent model names (e.g. PhaseNet EQTransformer).")
    args = ap.parse_args()

    from rapid.export import to_onnx, build_trt_engine

    picks = DEFAULT_MODELS
    if args.models:
        wanted = set(args.models)
        picks = [m for m in picks if m[0] in wanted]

    onnx_root = Path(args.onnx_dir); onnx_root.mkdir(parents=True, exist_ok=True)
    trt_root = Path(args.trt_dir); trt_root.mkdir(parents=True, exist_ok=True)

    for parent, child, in_samples in picks:
        onnx_path = onnx_root / f"{parent}_{child}.onnx"
        print(f"[ONNX] {parent}/{child} -> {onnx_path}")
        to_onnx(parent, child, onnx_path)

        if args.skip_trt:
            continue

        for precision in ("fp32", "fp16"):
            plan = trt_root / f"{parent}_{child}_{precision}.plan"
            print(f"[TRT]  {parent}/{child} ({precision}) -> {plan}")
            try:
                build_trt_engine(
                    onnx_path, plan,
                    precision=precision,
                    min_batch=args.min_batch,
                    opt_batch=args.opt_batch,
                    max_batch=args.max_batch,
                    in_samples=in_samples,
                )
            except Exception as e:
                print(f"  skipped: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
