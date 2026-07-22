#!/usr/bin/env python3
"""Run the SeisBench dtype + timing matrix (wall time + pick quality).

This is separate from ``run_matrix.py`` (miniSEED network sweep). Rows are
appended to ``results/seisbench_matrix.jsonl`` by default with
``data_source="seisbench"``, ``kind="seisbench"``, and ``runner`` either
``baseline_annotate`` or ``lean_pytorch``.

Example::

    export SEISBENCH_CACHE_ROOT=/lambda1a/seisbench
    python benchmarks/fair/run_seisbench_matrix.py --config configs/seisbench_dtype_matrix.json

Plotting scripts will be re-introduced separately.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rapid.seisbench_matrix import SeisBenchMatrixConfig, run_seisbench_matrix  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/seisbench_dtype_matrix.json"),
    )
    args = ap.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = SeisBenchMatrixConfig.from_dict(raw)
    run_seisbench_matrix(cfg)


if __name__ == "__main__":
    main()
