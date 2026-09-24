#!/usr/bin/env python3
"""Fig 15 - One thread versus the optimized thread count, one model instance.

y = Stage B makespan (K=1, T=1) / Stage A makespan (K=1, optimized T).
Diamonds: 580 stations.  Circles: 250 stations.  Line at 1.0.
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

import common as c

STEM = "fig15_one_thread_over_optimized"


def main() -> int:
    rows = c.load_orch_rows(stages=("A", "B"))
    c.configure_style()
    fig, axes = c.panel_grid([0], c.DEVICES, width=6.9, row_height=2.7, sharey=True)
    pos = c.x_positions(c.CPUS)
    csv_rows = []
    n_pts = 0
    for ci, dev in enumerate(c.DEVICES):
        ax = axes[0][ci]
        ymax = 0.0
        for mi, model in enumerate(c.MODELS):
            for n_st, ls, mk in ((250, "--", "o"), (580, "-", "D")):
                locked = {}
                for n_cpus in c.CPUS:
                    stage_a = c.select(rows, stage="A", model=model, n_stations=n_st, device=dev, n_cpus=n_cpus)
                    stage_b = c.select(rows, stage="B", composition="ma", k=1, model=model, n_stations=n_st, device=dev, n_cpus=n_cpus)
                    if not stage_a or not stage_b:
                        continue
                    a_mean = float(np.mean([r["makespan_s"] for r in stage_a]))
                    vals = [r["makespan_s"] / a_mean for r in stage_b]
                    locked[n_cpus] = vals
                    for r in stage_b:
                        csv_rows.append({
                            "model": model, "n_stations": n_st, "device": dev, "n_cpus": n_cpus, "k": 1,
                            "torch_threads": r["torch_threads"], "repeat_index": r["repeat_index"],
                            "makespan_s": r["makespan_s"], "stage_a_makespan_s_mean": a_mean,
                            "ratio": r["makespan_s"] / a_mean,
                        })
                        n_pts += 1
                ymax = max(ymax, c.plot_series(
                    ax, xs=c.CPUS, pos=pos, locked=locked, color=c.MODEL_COLOR[model],
                    model_index=mi, n_models=len(c.MODELS), marker=mk, linestyle=ls,
                    align_x=True, repeat_marker=mk,
                ))
        ax.axhline(1.0, color=c.MUTED, linestyle=":", linewidth=1.0, zorder=1)
        c.set_categorical_x(ax, c.CPUS, pos)
        ax.set_xlabel(r"Host Logical CPU Allocation $C$")
        ax.set_title(f"{c.DEVICE_LABEL[dev]} · Single instance")
        if ci == 0:
            ax.set_ylabel("Makespan Ratio\n(One Thread / Optimized Thread)")
            ax.set_ylim(0, ymax * 1.12)
        c.tidy_axis(ax)
    extra = [
        Line2D([0], [0], color=c.INK, marker="D", linewidth=1.5, markersize=4.5, label="580 stations"),
        Line2D([0], [0], color=c.INK, marker="o", linewidth=1.2, linestyle="--", markersize=4.5, label="250 stations"),
    ]
    c.model_legend(fig, extra=extra, ncol=4, y=-0.04)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, len(c.MODELS) * len(c.STATIONS) * len(c.DEVICES) * len(c.CPUS) * 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
