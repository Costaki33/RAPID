#!/usr/bin/env python3
"""Head-to-head: warm Model-Actor vs warm batched annotate(), CPU and GPU.

Reads the isolated head-to-head run (results/fair_benchmark_h2h) and writes the
warm per-window latency figure + table that anchors the paper's central claim:
on CPU, persistent-actor orchestration beats warm annotate() for the heavy
(accurate) models; on GPU, annotate() wins. -> docs/figures_v3/.
"""
from __future__ import annotations
import glob, json, statistics, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fair_benchmark_h2h_v2"   # 10-repeat run
OUT = ROOT / "docs" / "figures_v3"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})


def canon(m): return "w6000" if m in ("EQTransformer", "EQT-NC") else "w6000ov03"


def load():
    D = {}
    for p in glob.glob(str(RES / "**" / "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text()); m = r["meta"]
        except Exception:
            continue
        lat = r.get("latency") or {}
        w = lat.get("warm_feed_mean_s_mean"); c = lat.get("cold_feed_total_s_mean")
        if w is None:
            continue
        if canon(m["model"]) not in m.get("tag", "") and m["model"] not in ("EQTransformer", "EQT-NC"):
            continue
        key = (m["model"], m["n_stations"], m["device"], m["method"])
        D.setdefault(key, []).append((c, w))
    return D


def best(D, model, st, dev, meth):
    v = D.get((model, st, dev, meth))
    return min(v, key=lambda x: x[1]) if v else None


def load_repeats():
    """Per-repeat warm latencies per config, for confidence intervals."""
    R = {}
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
            R.setdefault((m["model"], m["n_stations"], m["device"], m["method"]), []).extend(vals)
    return R


_T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}


def ci95(vals):
    n = len(vals)
    if not vals:
        return None
    if n < 2:
        return vals[0], 0.0
    se = statistics.stdev(vals) / (n ** 0.5)
    return statistics.mean(vals), _T.get(n, 1.96) * se


def boot_ratio(a, b, n=10000):
    rng = random.Random(0); rs = []
    for _ in range(n):
        rb = statistics.mean(rng.choices(b, k=len(b)))
        if rb > 0:
            rs.append(statistics.mean(rng.choices(a, k=len(a))) / rb)
    rs.sort()
    return statistics.mean(a) / statistics.mean(b), rs[int(0.025 * len(rs))], rs[int(0.975 * len(rs))]


def fig(D, st=580):
    fig, ax = plt.subplots(figsize=(9, 4.3))
    width = 0.2
    x = range(len(MODELS))
    series = [
        ("annotate — CPU", "cpu", "stream_annotate", "#e74c3c"),
        ("Model-Actor — CPU (GPU-free)", "cpu", "stream_modelactor", "#2e86c1"),
        ("annotate — GPU", "gpu", "stream_annotate", "#f1948a"),
        ("Model-Actor — GPU", "gpu", "stream_modelactor", "#aed6f1"),
    ]
    for i, (label, dev, meth, color) in enumerate(series):
        ys = []
        for model in MODELS:
            b = best(D, model, st, dev, meth)
            ys.append(b[1] if b else 0)
        ax.bar([xi + (i - 1.5) * width for xi in x], ys, width, label=label, color=color)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODELS)
    ax.set_ylabel("warm per-window latency (s)")
    ax.set_title(f"Warm per-window latency: Model-Actor vs annotate(), {st} stations\n"
                 "On CPU, Model-Actor beats annotate for the heavy models; on GPU, annotate wins")
    ax.legend(fontsize=8)
    # annotate the CPU speedups
    for j, model in enumerate(MODELS):
        ca = best(D, model, st, "cpu", "stream_annotate"); cm = best(D, model, st, "cpu", "stream_modelactor")
        if ca and cm:
            ax.text(j - 0.75 * width, max(ca[1], cm[1]) + 0.15, f"{ca[1]/cm[1]:.1f}x", fontsize=8, color="#2e86c1", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "fig_head_to_head.png", bbox_inches="tight")
    plt.close(fig)


def table(R):
    n = max((len(v) for v in R.values()), default=0)
    lines = ["## T5. Head-to-head: warm per-window latency, Model-Actor vs annotate() (measured)\n",
             f"_Mean ± 95% CI over n={n} repeats; speedup CI by bootstrap (10k resamples)._\n"]
    for st in (580, 250):
        rows = {k: v for k, v in R.items() if k[1] == st}
        if not rows:
            continue
        lines.append(f"\n### {st} stations (warm per-window latency, s; CPU = 20 cores)\n")
        lines.append("| Model | CPU annotate | CPU Model-Actor | CPU speedup [95% CI] | GPU annotate | GPU Model-Actor |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODELS:
            def g(dev, meth):
                return rows.get((model, st, dev, meth))
            ca, cm = g("cpu", "stream_annotate"), g("cpu", "stream_modelactor")
            ga, gm = g("gpu", "stream_annotate"), g("gpu", "stream_modelactor")

            def f(v):
                c = ci95(v)
                return f"{c[0]:.2f} ± {c[1]:.2f}" if c else "–"
            sp = "–"
            if ca and cm:
                r, lo, hi = boot_ratio(ca, cm)
                sp = f"**{r:.1f}×** [{lo:.1f}–{hi:.1f}]"
            lines.append(f"| {model} | {f(ca)} | {f(cm)} | {sp} | {f(ga)} | {f(gm)} |")
    lines.append("\nWarm = mean of feeds 1–7 (steady state); cold feed 0 excluded. Model-Actor uses "
                 "native classify() inside the persistent pool. Lower is better. CPU speedup = "
                 "CPU annotate ÷ CPU Model-Actor.\n")
    (OUT / "table_h2h.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    D = load()
    fig(D, 580)
    table(load_repeats())
    print(f"\nWrote {OUT/'fig_head_to_head.png'} and {OUT/'table_h2h.md'}")
