#!/usr/bin/env python3
"""Fig 12 - Ripper versus Model-Actor at matched K (Stage B playback).

x = Model Instances K.  y = Ripper makespan / Model-Actor makespan at the
same model, network, device, C, and K.  One point per host allocation C.
Line = mean across C.  Line at 1.0.  Log y.
"""
from __future__ import annotations

import numpy as np

import common as c

STEM = "fig12_ripper_over_model_actor_vs_k"


def main() -> int:
    rows = c.load_orch_rows(stages=("B",))
    cells = c.cell_means(c.select(rows, stage="B"), "makespan_s", c.ORCH_CELL_KEYS)
    mean_of = {k: d["mean"] for k, d in cells.items()}
    c.configure_style()
    fig, axes = c.panel_grid(c.STATIONS, c.DEVICES, width=6.9, row_height=2.4, sharey="row")
    csv_rows = []
    n_pts = 0
    for ri, n_st in enumerate(c.STATIONS):
        for ci, dev in enumerate(c.DEVICES):
            ax = axes[ri][ci]
            ks_all = (1, 2) if n_st == 250 else ((1, 2, 4, 5) if dev == "cpu" else (1, 2))
            pos = c.x_positions(ks_all)
            for mi, model in enumerate(c.MODELS):
                locked = {}
                for k in ks_all:
                    vals = []
                    for C in c.CPUS:
                        ma = mean_of.get(("B", "ma", model, n_st, dev, C, k))
                        rp = mean_of.get(("B", "ripper", model, n_st, dev, C, k))
                        if ma and rp:
                            vals.append(rp / ma)
                            csv_rows.append({"model": model, "n_stations": n_st, "device": dev, "n_cpus": C, "k": k,
                                             "ma_makespan_s_mean": ma, "ripper_makespan_s_mean": rp, "ripper_over_ma": rp / ma})
                            n_pts += 1
                    if vals:
                        locked[k] = vals
                c.plot_series(ax, xs=ks_all, pos=pos, locked=locked, color=c.MODEL_COLOR[model],
                              model_index=mi, n_models=len(c.MODELS), align_x=True)
            ax.axhline(1.0, color=c.MUTED, linestyle=":", linewidth=1.0, zorder=1)
            ax.set_yscale("log")
            c.set_categorical_x(ax, ks_all, pos, pad=0.55)
            ax.set_title(f"{c.DEVICE_LABEL[dev]} · {n_st} st · Stage B playback")
            if ri == len(c.STATIONS) - 1:
                ax.set_xlabel(r"Model Instances $K$")
            if ci == 0:
                ax.set_ylabel("Makespan Ratio\n(Ripper / Model-Actor)")
            c.tidy_axis(ax)
    c.model_legend(fig, ncol=6, y=-0.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, 240)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
