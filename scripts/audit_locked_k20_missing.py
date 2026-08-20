#!/usr/bin/env python3
"""Audit missing locked unique orch paths (CPU K=20 focus + full unique check)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(
    os.environ.get(
        "RESULTS_ROOT",
        "results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19",
    )
)

# Uncapped CPU, GPU locked K=4, single CORE_GRID slot so each unique path appears once.
os.environ["LAYER"] = "playback,staggered"
os.environ["DEVICES"] = "cpu,gpu"
os.environ["CORE_GRID"] = "20"
os.environ.pop("CPU_K_CAP", None)
os.environ["GPU_K_CAP"] = "4"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.fair.locked_recipe_transfer_matrix import (  # noqa: E402
    cell_ok,
    env_cells,
    result_path,
)


def main() -> int:
    cells = env_cells()
    print(f"RESULTS_ROOT={ROOT}")
    print(f"matrix_cells={len(cells)} (CORE_GRID=20, uncapped CPU, GPU_K_CAP=4)")
    print()
    print("=== Missing unique locked orch paths ===")
    missing = []
    for c in cells:
        p = result_path(ROOT, c)
        ok = cell_ok(p)
        tag = "OK" if ok else "MISSING"
        line = (
            f"{tag:7} {c['layer']:10} {c['model']:15} {c['n_stations']:3}st "
            f"{c['device']:3} k={c['k_ma']:2} {c['arrival']}/{c['fill']}"
        )
        if not ok:
            missing.append((c, p))
            print(line)
            print(f"        {p}")
    print()
    print(f"missing_unique={len(missing)} / {len(cells)}")
    print()
    print("=== CPU locked-want K=20 subset (580 all models + 250 EQCCT) ===")
    want20 = [
        c
        for c in cells
        if c["device"] == "cpu"
        and c["k_ma"] == 20
        and (c["n_stations"] == 580 or c["model"] == "EQCCT")
    ]
    for c in want20:
        p = result_path(ROOT, c)
        tag = "OK" if cell_ok(p) else "MISSING"
        print(
            f"{tag:7} {c['layer']:10} {c['model']:15} {c['n_stations']:3}st "
            f"k={c['k_ma']}"
        )
    miss20 = [c for c in want20 if not cell_ok(result_path(ROOT, c))]
    print(f"kma20_want={len(want20)} missing={len(miss20)}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
