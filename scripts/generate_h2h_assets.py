#!/usr/bin/env python3
"""Head-to-head: warm Model-Actor vs warm batched annotate(), CPU and GPU.

Reads the isolated head-to-head run (results/fair_benchmark_h2h) and writes the
warm per-window latency figure + table that anchors the paper's central claim:
on CPU, persistent-actor orchestration beats warm annotate() for the heavy
(accurate) models; on GPU, annotate() wins. -> docs/figures_v3/.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fair_benchmark_h2h"
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


def table(D):
    lines = ["## T5. Head-to-head: warm per-window latency, Model-Actor vs annotate() (measured)\n"]
    for st in (580, 250):
        lines.append(f"\n### {st} stations (warm feed, s; CPU = 20 cores)\n")
        lines.append("| Model | CPU annotate | CPU Model-Actor | CPU speedup | GPU annotate | GPU Model-Actor |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for model in MODELS:
            ca = best(D, model, st, "cpu", "stream_annotate"); cm = best(D, model, st, "cpu", "stream_modelactor")
            ga = best(D, model, st, "gpu", "stream_annotate"); gm = best(D, model, st, "gpu", "stream_modelactor")
            sp = f"{ca[1]/cm[1]:.1f}x" if (ca and cm) else "–"
            f = lambda x: f"{x[1]:.2f}" if x else "–"
            lines.append(f"| {model} | {f(ca)} | {f(cm)} | **{sp}** | {f(ga)} | {f(gm)} |")
    lines.append("\nWarm = mean of feeds 1–7 (steady state); cold feed 0 excluded. Model-Actor uses "
                 "native classify() inside the persistent pool. Lower is better.\n")
    (OUT / "table_h2h.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    D = load()
    fig(D, 580)
    table(D)
    print(f"\nWrote {OUT/'fig_head_to_head.png'} and {OUT/'table_h2h.md'}")
