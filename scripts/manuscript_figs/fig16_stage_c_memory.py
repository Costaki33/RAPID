#!/usr/bin/env python3
"""Fig 16 - Peak memory of the Stage C pools.

The pools are the Stage B instance counts carried into the staggered-arrival
trials.  Rows are network size.  Columns are CPU peak host PSS, GPU peak host
PSS, and GPU peak VRAM.  Model-Actor solid circles, Ripper dashed squares.
Faint markers are the five isolated repeats.  Values are MB/1000.
"""
from __future__ import annotations

import common as c

STEM = "fig16_stage_c_peak_memory"
PANELS = (
    ("cpu", "peak_pss_mb", "Peak Host PSS (GB)", "CPU · Host PSS"),
    ("gpu", "peak_pss_mb", "Peak Host PSS (GB)", "GPU · Host PSS"),
    ("gpu", "peak_vram_mb", "Peak VRAM (GB)", "GPU · VRAM"),
)


def main() -> int:
    rows = c.load_orch_rows(stages=("C",))
    c.configure_style()
    fig, axes = c.panel_grid(c.STATIONS, [p[3] for p in PANELS], width=7.2, row_height=2.45, sharey="col")
    pos = c.x_positions(c.CPUS)
    csv_rows = []
    n_pts = 0
    for ci, (dev, key, ylabel, title) in enumerate(PANELS):
        col_ymax = 0.0
        for ri, n_st in enumerate(c.STATIONS):
            ax = axes[ri][ci]
            for mi, model in enumerate(c.MODELS):
                col = c.MODEL_COLOR[model]
                for comp, ls, mk in (("ma", "-", "o"), ("ripper", "--", "s")):
                    locked = {}
                    for C in c.CPUS:
                        sub = c.select(
                            rows, stage="C", composition=comp, model=model,
                            n_stations=n_st, device=dev, n_cpus=C,
                        )
                        vals = [r[key] / 1000.0 for r in sub if r.get(key) is not None]
                        if not vals:
                            continue
                        locked[C] = vals
                        for i, v in enumerate(vals):
                            n_pts += 1
                            csv_rows.append({
                                "composition": comp, "model": model, "n_stations": n_st,
                                "device": dev, "n_cpus": C, "k": sub[0]["k"],
                                "metric": key, "repeat_index": i, "value_gb": v,
                            })
                    col_ymax = max(col_ymax, c.plot_series(
                        ax, xs=c.CPUS, pos=pos, locked=locked, color=col,
                        model_index=mi, n_models=len(c.MODELS), align_x=True,
                        marker=mk, linestyle=ls, repeat_marker=mk,
                    ))
            c.set_categorical_x(ax, c.CPUS, pos)
            ax.set_title(f"{title} · {n_st} st · Stage C")
            ax.set_ylabel(ylabel)
            if ri == len(c.STATIONS) - 1:
                ax.set_xlabel(r"Host Logical CPU Allocation $C$")
            c.tidy_axis(ax)
        for ri in range(len(c.STATIONS)):
            axes[ri][ci].set_ylim(0, col_ymax * 1.12 if col_ymax else 1)
    from matplotlib.lines import Line2D
    extra = [
        Line2D([0], [0], color=c.INK, marker="o", linewidth=1.5, markersize=4.5, label="Model-Actor"),
        Line2D([0], [0], color=c.INK, marker="s", linewidth=1.0, linestyle="--", markersize=4.5, label="Ripper"),
    ]
    c.model_legend(fig, extra=extra, ncol=4, y=-0.02)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, 1440)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
