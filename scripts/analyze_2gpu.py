#!/usr/bin/env python3
"""Two-GPU Model-Actor vs the single-GPU / CPU / annotate baselines.

Pulls warm per-window latency for the actor-pool-split-across-2-GPUs run
(results/fair_benchmark_h2h_2gpu) and sets it beside the single-device numbers
from the 10-repeat head-to-head (results/fair_benchmark_h2h_v2): single-GPU
Model-Actor, GPU annotate, and CPU Model-Actor. Answers: does splitting the pool
across both GPUs close the gap to annotate -- and to the GPU-free CPU pool?

    python3 scripts/analyze_2gpu.py
"""
from __future__ import annotations
import glob, json, statistics, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TWO = ROOT / "results" / "fair_benchmark_h2h_2gpu"
V2 = ROOT / "results" / "fair_benchmark_h2h_v2"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
_T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def warm_vals(root, want_method=None):
    """(model, n_stations, device, method) -> [warm_feed_mean_s per repeat]."""
    D = {}
    for p in glob.glob(str(root / "**" / "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text()); m = r["meta"]
        except Exception:
            continue
        meth = m.get("method")
        if want_method and meth != want_method:
            continue
        reps = (r.get("latency") or {}).get("repeats") or []
        vals = [x.get("warm_feed_mean_s") for x in reps if x.get("warm_feed_mean_s") is not None]
        if vals:
            D.setdefault((m["model"], m["n_stations"], m["device"], meth), []).extend(vals)
    return D


def ci95(vals):
    if not vals:
        return None
    n = len(vals)
    if n < 2:
        return vals[0], 0.0
    return statistics.mean(vals), _T.get(n, 1.96) * statistics.stdev(vals) / (n ** 0.5)


def boot(a, b, n=10000):
    rng = random.Random(0); rs = []
    for _ in range(n):
        rb = statistics.mean(rng.choices(b, k=len(b)))
        if rb > 0:
            rs.append(statistics.mean(rng.choices(a, k=len(a))) / rb)
    rs.sort()
    return statistics.mean(a) / statistics.mean(b), rs[int(.025 * len(rs))], rs[int(.975 * len(rs))]


def main():
    two = warm_vals(TWO)                              # stream_modelactor_2gpu
    v2 = warm_vals(V2)                                # all single-device methods
    n2 = max((len(v) for v in two.values()), default=0)
    print(f"Two-GPU actor-pool split vs baselines  (2gpu n={n2}, v2 n=10)\n")
    for st in (580, 250):
        print(f"=== {st} stations: warm per-window latency, mean ± 95% CI (s) ===")
        print(f"{'model':14s} {'MA 1-GPU':>12s} {'MA 2-GPU':>12s} {'2gpu speedup':>16s} "
              f"{'GPU annotate':>13s} {'CPU MA':>12s}")
        for model in MODELS:
            g1 = v2.get((model, st, "gpu", "stream_modelactor"))
            g2 = two.get((model, st, "gpu", "stream_modelactor_2gpu"))
            ann = v2.get((model, st, "gpu", "stream_annotate"))
            cpu = v2.get((model, st, "cpu", "stream_modelactor"))

            def f(v):
                c = ci95(v)
                return f"{c[0]:.2f}±{c[1]:.2f}" if c else "–"
            sp = "–"
            if g1 and g2:
                r, lo, hi = boot(g1, g2)             # 1-GPU / 2-GPU = how much faster 2-GPU is
                sp = f"{r:.1f}× [{lo:.1f}–{hi:.1f}]"
            print(f"{model:14s} {f(g1):>12s} {f(g2):>12s} {sp:>16s} {f(ann):>13s} {f(cpu):>12s}")
        print()
    # verdict line
    print("Reads: 2gpu speedup = MA 1-GPU ÷ MA 2-GPU (how much the split buys).")
    print("Compare MA 2-GPU against GPU annotate (does the split let actors win on GPU?)")
    print("and against CPU MA (does it beat the GPU-free pool?).")


if __name__ == "__main__":
    main()
