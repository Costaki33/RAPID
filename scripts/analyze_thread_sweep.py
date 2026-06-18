#!/usr/bin/env python3
"""Thread-sensitivity sweep: total CPU time vs torch intra-op thread count.

Shows that for these seismic models on CPU, intra-op threading does not convert
cores into throughput -- it is neutral-to-harmful for batched annotate/slipstream
and catastrophic for per-station classify() -- so each single-process method has
an optimum well below the 20-core budget. Prints a table + the per-method optimum
and (if matplotlib is available) writes docs/figures_v3/fig_thread_sweep.png.

    python3 scripts/analyze_thread_sweep.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fair_benchmark_threadsweep"
OUT = ROOT / "docs" / "figures_v3"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
METHODS = ["classify", "annotate", "slipstream"]


def load():
    """(method, model) -> {threads: total_s}."""
    D = {}
    for p in glob.glob(str(RES / "**" / "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text()); m = r["meta"]
        except Exception:
            continue
        t = (r.get("timing") or {}).get("total_s_mean")
        thr = m.get("torch_threads")
        if t is None or thr is None:
            continue
        D.setdefault((m["method"], m["model"]), {})[int(thr)] = t
    return D


def main():
    D = load()
    if not D:
        print(f"No results yet under {RES}")
        return
    for meth in METHODS:
        rows = {model: D.get((meth, model), {}) for model in MODELS}
        if not any(rows.values()):
            continue
        threads = sorted({t for r in rows.values() for t in r})
        print(f"\n=== {meth}: total CPU time (s), STEAD 580 st, 20 cores, by torch intra-op threads ===")
        hdr = "thr  " + "  ".join(f"{m:>13s}" for m in MODELS)
        print(hdr)
        for t in threads:
            cells = "  ".join((f"{rows[m][t]:13.1f}" if t in rows[m] else f"{'–':>13s}") for m in MODELS)
            print(f"{t:<4d} {cells}")
        # per-model optimum
        opt = "opt  " + "  ".join(
            (f"{min(rows[m], key=rows[m].get):>4d}t={min(rows[m].values()):6.1f}s"
             if rows[m] else f"{'–':>13s}") for m in MODELS)
        print(opt)

    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib unavailable; skipped figure)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)
    colors = {"PhaseNet": "#2e86c1", "PhaseNetLight": "#27ae60",
              "EQTransformer": "#e67e22", "EQT-NC": "#c0392b"}
    for ax, meth in zip(axes, METHODS):
        for model in MODELS:
            r = D.get((meth, model), {})
            if not r:
                continue
            xs = sorted(r)
            ax.plot(xs, [r[x] for x in xs], "o-", label=model, color=colors[model])
        ax.axhline(30, color="k", ls=":", lw=1)
        ax.text(1, 32, "30 s budget", fontsize=7)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("torch intra-op threads")
        ax.set_title(meth)
        ax.grid(True, which="both", alpha=0.3)
        if meth == "classify":
            ax.set_ylabel("total CPU time (s, log)")
            ax.legend(fontsize=7)
    fig.suptitle("Thread sensitivity on CPU: intra-op threading does not buy throughput "
                 "(catastrophic for per-station classify) — STEAD 580 st, 20 cores", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_thread_sweep.png", bbox_inches="tight")
    print(f"\nWrote {OUT/'fig_thread_sweep.png'}")


if __name__ == "__main__":
    main()
