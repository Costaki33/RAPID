#!/usr/bin/env python3
"""Regenerate every manuscript figure (PNG + PDF + audit CSV) in order.

Usage:  python scripts/manuscript_figs/make_all.py [--no-cache]
Outputs: results/annotate_config/stead_iso_2026-09-08/figures/figNN/figNN_*.{png,pdf,csv}
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIGS = [
    "fig01_packaging",
    "fig02_host_cpus",
    "fig03_threads",
    "fig04_batch_size",
    "fig05_dtype",
    "fig06_cpu_vs_gpu_scaling",
    "fig07_memory_vs_batch",
    "fig08_f1_vs_dtype",
    "fig14_winner_configs",
    "fig09_stage_a_overhead",
    "fig15_one_thread_over_optimized",
    "fig10_stage_b_makespan_vs_k",
    "fig11_ripper_over_ma",
    "fig12_stage_c_latency",
    "fig16_stage_c_memory",
]


def main() -> int:
    import common

    if "--no-cache" in sys.argv:
        common.load_native_rows(use_cache=False)
    rc = 0
    t0 = time.perf_counter()
    for name in FIGS:
        print(f"\n### {name}")
        mod = importlib.import_module(name)
        rc |= int(mod.main() or 0)
    print(f"\ndone in {time.perf_counter() - t0:.1f}s -> {common.FIG_DIR}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
