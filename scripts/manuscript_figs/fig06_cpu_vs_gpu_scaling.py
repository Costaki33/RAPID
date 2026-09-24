#!/usr/bin/env python3
"""Fig 7 - CPU vs GPU winners and network-size scaling.

Left: quality-safe merged winner at C=20. CPU solid circles, GPU dashed
squares. Right: 580/250 mean-runtime ratio against the station-count ratio.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import common as c

STEM = "fig07_winners_cpu_vs_gpu_scaling"
XS = c.STATIONS
STATION_RATIO = 580 / 250


def winner_repeats(rows, w):
    sub = c.select(
        rows, model=w["model"], n_stations=w["n_stations"], device=w["device"], n_cpus=w["n_cpus"],
        dtype=w["dtype"], batch_size=w["batch_size"], thread_mult=w["thread_mult"], packaging="merged",
    )
    return [r["inference_s"] for r in sub if r["ok"]]


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()
    c.configure_style()
    fig, axes = c.panel_grid([0], ["time", "ratio"], width=6.9, row_height=3.15, sharey=False)
    ax, axr = axes[0]
    pos = c.x_positions(XS)
    csv_rows = []
    n_pts = 0
    ymax = 0.0
    ymin = None
    for mi, model in enumerate(c.MODELS):
        ratios = {}
        for dev, ls, mk in (("cpu", "-", "o"), ("gpu", "--", "s")):
            locked = {}
            for n_st in XS:
                w = winners[c.nwo.winner_key(model, n_st, dev, c.REF_CPUS)]
                vals = winner_repeats(rows, w)
                locked[n_st] = vals
                n_pts += len(vals)
                for v in vals:
                    ymin = v if ymin is None else min(ymin, v)
                for i, v in enumerate(vals):
                    csv_rows.append({"model": model, "device": dev, "n_stations": n_st, "n_cpus": c.REF_CPUS,
                                     "dtype": w["dtype"], "batch_size": w["batch_size"], "thread_mult": w["thread_mult"],
                                     "repeat_index": i, "inference_s": v})
            ymax = max(ymax, c.plot_series(
                ax, xs=XS, pos=pos, locked=locked, color=c.MODEL_COLOR[model],
                model_index=mi, n_models=len(c.MODELS), marker=mk, linestyle=ls, align_x=True,
            ))
            if locked[250] and locked[580]:
                ratios[dev] = float(np.mean(locked[580]) / np.mean(locked[250]))
        off = np.linspace(-0.22, 0.22, len(c.MODELS))[mi]
        for dev, mk in (("cpu", "o"), ("gpu", "s")):
            if dev in ratios:
                axr.scatter(off + (0 if dev == "cpu" else 1.0), ratios[dev], s=34, marker=mk,
                            color=c.MODEL_COLOR[model], edgecolors=c.INK, linewidths=0.5, zorder=4)
                csv_rows.append({"model": model, "device": dev, "n_stations": "580/250", "n_cpus": c.REF_CPUS,
                                 "dtype": "", "batch_size": "", "thread_mult": "", "repeat_index": -1,
                                 "inference_s": ratios[dev]})
    c.set_categorical_x(ax, XS, pos, [f"{n} stations" for n in XS])
    ax.set_yscale("log")
    if ymin and ymax:
        ax.set_ylim(ymin * 0.75, ymax * 1.45)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Inference Time (s)")
    ax.set_xlabel("Network Size")
    ax.set_title("Quality-safe Merged winners · C=20")
    c.tidy_axis(ax)
    ax.text(-0.16, 1.08, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)

    axr.axhline(STATION_RATIO, color=c.MUTED, linestyle=":", linewidth=1.0)
    axr.text(1.42, STATION_RATIO, f"Network Size Ratio  {STATION_RATIO:.2f}", va="bottom", ha="right", fontsize=7, color=c.MUTED)
    axr.set_xticks([0, 1.0])
    axr.set_xticklabels(["CPU", "GPU"])
    axr.set_xlim(-0.55, 1.55)
    axr.set_ylabel("Runtime-Network size ratio (580 / 250)")
    axr.set_title("Runtime Scaling Behavior Across Network Sizes")
    c.tidy_axis(axr)
    axr.text(-0.18, 1.08, "B", transform=axr.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="left", clip_on=False)

    c.model_legend(
        fig,
        extra=[
            Line2D([0], [0], color=c.INK, linewidth=1.6, marker="o", markersize=4.5, label="CPU"),
            Line2D([0], [0], color=c.INK, linewidth=1.1, linestyle="--", marker="s", markersize=4.5, label="GPU"),
        ],
        ncol=4,
        y=-0.02,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows, ["model", "device", "n_stations", "n_cpus", "dtype", "batch_size", "thread_mult", "repeat_index", "inference_s"])
    c.report(STEM, n_pts, len(c.MODELS) * 2 * 2 * 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
