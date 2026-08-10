#!/usr/bin/env python3
"""Minimal example: pick a synthetic network with Model-Actor or Ripper.

Build a network first::

    python examples/build_seisbench_network.py \\
        --dataset stead --n-stations 50 --require-s \\
        --max-pick-sample 2951 --trim-samples 3001

Then pick it::

    python examples/pick_network.py \\
        --input-dir data/seisbench_networks/stead_n50_... \\
        --strategy modelactor --forward classify --n-workers 8

    python examples/pick_network.py \\
        --input-dir ... --strategy modelactor --forward annotate_bf16 \\
        --n-workers 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rapid import pick  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--model", default="PhaseNet")
    ap.add_argument("--strategy", choices=("modelactor", "ripper"), default="modelactor")
    ap.add_argument(
        "--forward",
        choices=("classify", "annotate_bf16", "annotate_fp16", "annotate"),
        default="classify",
    )
    ap.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="bf16")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--gpus", default=None, help="Comma-separated GPU ids, e.g. 0 or 0,1")
    args = ap.parse_args()

    out = args.output_dir or (_ROOT / "results" / "example_picks")
    gpus = None
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]

    pick(
        args.input_dir,
        out,
        model=args.model,
        strategy=args.strategy,
        forward=args.forward,
        dtype=args.dtype,
        n_workers=args.n_workers,
        gpus=gpus,
    )
    print(f"Picks written under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
