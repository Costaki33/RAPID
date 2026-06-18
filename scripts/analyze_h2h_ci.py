#!/usr/bin/env python3
"""Confidence intervals for the warm head-to-head (annotate vs Model-Actor).

Reads a head-to-head results dir (default results/fair_benchmark_h2h_v2, the
10-repeat run) and reports, per (model, device), the warm per-window latency as
mean +/- 95% CI and the annotate/Model-Actor speedup with a bootstrap 95% CI --
so the paper's multipliers carry uncertainty instead of point estimates.

    python3 scripts/analyze_h2h_ci.py [results_dir]
"""
from __future__ import annotations
import glob, json, sys, statistics, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "fair_benchmark_h2h_v2"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]


def canon(m): return "w6000" if m in ("EQTransformer", "EQT-NC") else "w6000ov03"


def load():
    D = {}
    for p in glob.glob(str(RES / "**" / "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text()); m = r["meta"]
        except Exception:
            continue
        if canon(m["model"]) not in m.get("tag", "") and m["model"] not in ("EQTransformer", "EQT-NC"):
            continue
        reps = (r.get("latency") or {}).get("repeats") or []
        vals = [x.get("warm_feed_mean_s") for x in reps if x.get("warm_feed_mean_s") is not None]
        if vals:
            D.setdefault((m["model"], m["n_stations"], m["device"], m["method"]), []).extend(vals)
    return D


def ci95(vals):
    n = len(vals)
    if n < 2:
        return (vals[0], 0.0) if vals else (float("nan"), 0.0)
    mean = statistics.mean(vals)
    se = statistics.stdev(vals) / (n ** 0.5)
    # t_0.975 approx for small n
    t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}.get(n, 1.96)
    return mean, t * se


def boot_ratio(a, b, n=10000):
    """Bootstrap 95% CI for mean(a)/mean(b) (annotate / model-actor)."""
    rng = random.Random(0)
    rs = []
    for _ in range(n):
        ra = statistics.mean(rng.choices(a, k=len(a)))
        rb = statistics.mean(rng.choices(b, k=len(b)))
        if rb > 0:
            rs.append(ra / rb)
    rs.sort()
    return statistics.mean([x / y for x, y in zip(a, b)]) if False else (
        statistics.mean(a) / statistics.mean(b), rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs))])


def main():
    D = load()
    n_any = max((len(v) for v in D.values()), default=0)
    print(f"Head-to-head CI report  ({RES.name}, up to n={n_any} repeats/config)\n")
    for st in (580, 250):
        rows = {k: v for k, v in D.items() if k[1] == st}
        if not rows:
            continue
        print(f"=== {st} stations: warm per-window latency, mean ± 95% CI (s) ===")
        print(f"{'model':14s} {'CPU annotate':>18s} {'CPU Model-Actor':>18s} {'CPU speedup [95% CI]':>24s} "
              f"{'GPU annotate':>16s} {'GPU Model-Actor':>16s}")
        for model in MODELS:
            def g(dev, meth):
                return rows.get((model, st, dev, meth))
            ca, cm = g("cpu", "stream_annotate"), g("cpu", "stream_modelactor")
            ga, gm = g("gpu", "stream_annotate"), g("gpu", "stream_modelactor")

            def fmt(v):
                if not v:
                    return "–"
                mean, h = ci95(v)
                return f"{mean:.2f}±{h:.2f}"
            sp = "–"
            if ca and cm:
                r, lo, hi = boot_ratio(ca, cm)
                sp = f"{r:.1f}x [{lo:.1f}–{hi:.1f}]"
            print(f"{model:14s} {fmt(ca):>18s} {fmt(cm):>18s} {sp:>24s} {fmt(ga):>16s} {fmt(gm):>16s}")
        print()


if __name__ == "__main__":
    main()
