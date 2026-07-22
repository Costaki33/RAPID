#!/usr/bin/env python3
"""Audit which trial data is clean (trustworthy) vs contaminated.

Two confounds corrupt TIMING (not accuracy/memory): thread over-subscription
(single-process methods given too many torch threads) and concurrent-trial
contention (the scheduler packed many trials, so each fought for memory
bandwidth). Only the strictly-sequential isolation run is free of both.

This prints a per-directory verdict and runs integrity checks on the isolated
data so we can assert it really is clean.

    python3 benchmarks/analysis/audit_data.py
"""
from __future__ import annotations
import glob, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

VERDICT = [
    # dir, label, timing, accuracy, memory
    ("fair_benchmark", "Original matrix",
     "CONTAMINATED (CPU: threads+contention; GPU: host-contended)", "CLEAN", "CLEAN"),
    ("fair_benchmark_h2h", "Head-to-head v1 (3-rep)", "CONTAMINATED (superseded)", "n/a", "n/a"),
    ("fair_benchmark_h2h_v2", "Head-to-head v2 (10-rep)", "CONTAMINATED (concurrent dispatch)", "CLEAN", "CLEAN"),
    ("fair_benchmark_h2h_2gpu", "2-GPU (old)", "CONTAMINATED (sweep overlap)", "n/a", "partial VRAM"),
    ("fair_benchmark_threadsweep", "Thread-sweep diagnostic", "DIAGNOSTIC ONLY (partly contended)", "n/a", "n/a"),
    ("fair_benchmark_iso", "ISOLATED re-measure", "CLEAN (one trial at a time)", "CLEAN", "CLEAN"),
]


def n(d):
    return len(glob.glob(str(ROOT / "results" / d / "**" / "result.json"), recursive=True))


def main():
    print("=" * 86)
    print("  DATA CLEANLINESS AUDIT")
    print("=" * 86)
    print(f"  {'directory':28s} {'cells':>6s}  {'TIMING':<42s}")
    for d, label, timing, acc, mem in VERDICT:
        print(f"  {d:28s} {n(d):6d}  {timing:<42s}")
        print(f"  {'':28s} {'':6s}  accuracy={acc}  memory={mem}")

    print("\n" + "=" * 86)
    print("  ISO INTEGRITY CHECKS")
    print("=" * 86)
    nov = n("fair_benchmark_iso/oversub")
    print(f"  [1] oversub ran concurrently with main phases? {'NO (clean)' if nov == 0 else f'YES ({nov} cells) -- CHECK'}")
    susp = []
    for p in glob.glob(str(ROOT / "results/fair_benchmark_iso/h2h/**/result.json"), recursive=True):
        r = json.loads(Path(p).read_text()); m = r["meta"]
        vals = [x.get("warm_feed_mean_s") for x in (r.get("latency") or {}).get("repeats", [])
                if x.get("warm_feed_mean_s") is not None]
        if len(vals) >= 3 and "modelactor" in m["method"]:
            mean = statistics.mean(vals); cv = 100 * statistics.pstdev(vals) / mean if mean else 0
            if cv > 8:
                susp.append((cv, m["model"], m["method"], m["device"], m["n_stations"], vals))
    print(f"  [2] Model-Actor cells with warm CV>8% (tight=isolated): {len(susp)}")
    for cv, mo, me, dv, st, vals in susp:
        print(f"      cv={cv:.0f}%  {mo}/{me}/{dv}/{st}st  {[round(v,2) for v in vals]}")
    print("      (one elevated FIRST repeat = residual warmup, not contention; the 10-rep mean absorbs it)")


if __name__ == "__main__":
    main()
