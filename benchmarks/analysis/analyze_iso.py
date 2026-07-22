#!/usr/bin/env python3
"""Isolated head-to-head analysis: warm per-window latency with mean ± 95% CI AND
tail latency (p95/p99), from the strictly-sequential isolation run.

For a real-time budget the TAIL matters more than the mean -- a picker that
averages 2 s but spikes to 8 s on one window in twenty can still miss the 30 s
deadline for that window. We compute the tail from the per-feed latencies already
stored in each repeat (7 warm feeds x 10 repeats = 70 per-window samples/cell);
no re-run needed.

    python3 benchmarks/analysis/analyze_iso.py
"""
from __future__ import annotations
import glob, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results" / "fair_benchmark_iso" / "h2h"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
_T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]


def load():
    """(model, st, method, device) -> {repeat_warm_means:[...], window_samples:[...]}."""
    D = {}
    for result in glob.glob(str(RES / "**" / "result.json"), recursive=True):
        rp = Path(result)
        try:
            m = json.loads(rp.read_text())["meta"]
        except Exception:
            continue
        key = (m["model"], m["n_stations"], m["method"], m["device"])
        reps_means, samples = [], []
        for rf in sorted((rp.parent / "repeats").glob("repeat_*.json")):
            try:
                rr = json.loads(rf.read_text())
            except Exception:
                continue
            if not rr.get("success"):
                continue
            warm = [f["feed_total_s"] for f in (rr.get("feeds") or []) if f.get("feed_index", 0) >= 1]
            if warm:
                reps_means.append(statistics.mean(warm))
                samples.extend(warm)
        if samples:
            D[key] = {"means": reps_means, "samples": samples}
    return D


def ci95(means):
    n = len(means)
    if n < 2:
        return (means[0], 0.0) if means else (float("nan"), 0.0)
    return statistics.mean(means), _T.get(n, 1.96) * statistics.stdev(means) / (n ** 0.5)


def main():
    D = load()
    if not D:
        print(f"No isolated h2h results yet under {RES}")
        return
    for st in (580, 250):
        rows = {k: v for k, v in D.items() if k[1] == st}
        if not rows:
            continue
        print(f"\n=== {st} stations: warm per-window latency (s) — mean ± 95% CI [p95 / p99 / max] ===")
        print(f"{'model':14s} {'method':10s} {'cpu':>28s} {'gpu':>28s}")
        for model in MODELS:
            line = f"{model:14s} "
            for mlabel, meth in (("annotate", "stream_annotate"), ("MA", "stream_modelactor"),
                                 ("MA-2gpu", "stream_modelactor_2gpu")):
                cells = []
                for dev in ("cpu", "gpu"):
                    v = rows.get((model, st, meth, dev))
                    if not v:
                        cells.append(f"{'–':>28s}"); continue
                    mean, h = ci95(v["means"])
                    s = v["samples"]
                    cells.append(f"{mean:.2f}±{h:.2f} [{pct(s,.95):.2f}/{pct(s,.99):.2f}/{max(s):.2f}]".rjust(28))
                print(f"{model:14s} {mlabel:10s} {cells[0]} {cells[1]}")
        print("  (bracket = p95 / p99 / max over warm windows; n_repeats x 7 warm feeds)")


if __name__ == "__main__":
    main()
