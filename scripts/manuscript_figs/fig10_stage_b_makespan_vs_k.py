#!/usr/bin/env python3
"""Fig 11 - Stage B: all-ready playback makespan vs instance count K (580 st).

Rows = host allocation C, columns = device.  Model-Actor solid circles,
Ripper dashed squares.  Dotted line = optimized inference pipeline (the
single-process winner at that C).  Open ring = smallest K within 10% of
that model's fastest pool, carried into Stage C.  Log y.
"""
from __future__ import annotations

import numpy as np

import common as c

STEM = "fig11_stage_b_makespan_vs_k_580st"
N_ST = 580


def main() -> int:
    rows = c.load_orch_rows(stages=("B",))
    c.configure_style()
    fig, axes = c.panel_grid(c.CPUS, c.DEVICES, width=6.9, row_height=2.3, sharey="row")
    csv_rows = []
    n_pts = 0
    for ri, C in enumerate(c.CPUS):
        for ci, dev in enumerate(c.DEVICES):
            ax = axes[ri][ci]
            ks = [k for k in (c.K_TICKS_CPU if dev == "cpu" else c.K_TICKS_GPU) if k <= C]
            pos = c.x_positions(ks)
            ymax = 0.0
            for mi, model in enumerate(c.MODELS):
                col = c.MODEL_COLOR[model]
                for comp, ls, mk in (("ma", "-", "o"), ("ripper", "--", "s")):
                    locked = {}
                    nat = None
                    for k in ks:
                        sub = c.select(rows, stage="B", composition=comp, model=model, n_stations=N_ST, device=dev, n_cpus=C, k=k)
                        if not sub:
                            continue
                        nat = sub[0]["native_inference_s_mean"]
                        locked[k] = [r["makespan_s"] for r in sub if r["makespan_s"] is not None]
                        for r in sub:
                            csv_rows.append({"composition": comp, "model": model, "n_stations": N_ST, "device": dev, "n_cpus": C,
                                             "k": k, "dtype": r["dtype"], "batch_size": r["batch_size"], "torch_threads": r["torch_threads"],
                                             "repeat_index": r["repeat_index"], "makespan_s": r["makespan_s"],
                                             "native_inference_s_mean": nat})
                            n_pts += 1
                    if not locked:
                        continue
                    ymax = max(ymax, c.plot_series(ax, xs=ks, pos=pos, locked=locked, color=col, model_index=mi,
                                                   n_models=len(c.MODELS), marker=mk, linestyle=ls, align_x=True))
                    if comp == "ma":
                        means = {k: float(np.mean(v)) for k, v in locked.items()}
                        ke = c.elbow_k(means)
                        ax.scatter(pos[ke], means[ke], s=80, facecolors="none", edgecolors=col, linewidths=1.3, zorder=6)
                        if nat:
                            ax.axhline(nat, color=col, linestyle=":", linewidth=0.8, alpha=0.8, zorder=1)
            ax.set_yscale("log")
            c.set_categorical_x(ax, ks, pos, [str(k) for k in ks], pad=0.55)
            ax.set_title(f"{c.DEVICE_LABEL[dev]} · 580 st · C={C} · Stage B playback")
            if ri == len(c.CPUS) - 1:
                ax.set_xlabel(r"Model Instances $K$")
            if ci == 0:
                ax.set_ylabel("Makespan (s)")
            c.tidy_axis(ax)
    from matplotlib.lines import Line2D
    extra = [Line2D([0], [0], color=c.INK, marker="o", linewidth=1.5, markersize=4.5, label="Model-Actor"),
             Line2D([0], [0], color=c.INK, marker="s", linewidth=1.2, linestyle="--", markersize=4.5, label="Ripper"),
             Line2D([0], [0], color=c.INK, linestyle=":", linewidth=0.9, label="Optimized Inference Pipeline"),
             Line2D([0], [0], color=c.INK, marker="o", linewidth=0, markerfacecolor="none", markersize=9, label="Smallest $K$ within 10% of the fastest")]
    c.model_legend(fig, extra=extra, ncol=5, y=-0.01)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    # 580 st: MA cpu 6+6+6+... K<=C: C5:{1,2,4,5}=4, C10:5, C15:5, C20:6 = 20; ripper 4*4=16; gpu MA 3*4=12; ripper 2*4=8 -> 56 cells
    c.report(STEM, n_pts, 56 * len(c.MODELS) * 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
