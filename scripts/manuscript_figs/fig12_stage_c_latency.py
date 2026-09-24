#!/usr/bin/env python3
"""Fig 12 - Stage C: staggered-arrival pick latency.

x = host allocation C; y = pooled 95th-percentile (finish - ready) across all
stations and 5 repeats (1,250 / 2,900 samples).  Model-Actor solid circles,
Ripper dashed squares; the promoted K is printed next to each Model-Actor
point.  Log y.  Footnote: makespan range of every Stage C cell (~90 s ceiling).

"""
from __future__ import annotations

import common as c

STEM = "fig13_stage_c_p95_latency_vs_host_cpus"


def main() -> int:
    rows = c.load_orch_rows(stages=("C",))
    cells = c.cell_means(rows, "makespan_s", c.ORCH_CELL_KEYS)
    c.configure_style()
    fig, axes = c.panel_grid(c.STATIONS, c.DEVICES, width=6.9, row_height=2.5, sharey="row")
    pos = c.x_positions(c.CPUS)
    csv_rows = []
    n_pts = 0
    all_ms = []
    for ri, n_st in enumerate(c.STATIONS):
        for ci, dev in enumerate(c.DEVICES):
            ax = axes[ri][ci]
            for mi, model in enumerate(c.MODELS):
                col = c.MODEL_COLOR[model]
                for comp, ls, mk in (("ma", "-", "o"), ("ripper", "--", "s")):
                    xs, ys = [], []
                    for C in c.CPUS:
                        sub = c.select(rows, stage="C", composition=comp, model=model, n_stations=n_st, device=dev, n_cpus=C)
                        if not sub:
                            continue
                        r = sub[0]
                        ms_vals = [x["makespan_s"] for x in sub if x["makespan_s"] is not None]
                        all_ms.extend(ms_vals)
                        xs.append(pos[C])
                        ys.append(r["p95_e2e_s"])
                        n_pts += 1
                        csv_rows.append({"composition": comp, "model": model, "n_stations": n_st, "device": dev, "n_cpus": C,
                                         "k": r["k"], "p95_e2e_s": r["p95_e2e_s"], "p50_e2e_s": r["p50_e2e_s"],
                                         "makespan_s_mean": r["makespan_s_mean"], "n_latency_samples": n_st * 5})
                        if comp == "ma":
                            ax.annotate(f"{r['k']}", (pos[C], r["p95_e2e_s"]), textcoords="offset points",
                                        xytext=(0, 4), ha="center", fontsize=5.2, color=col, zorder=7)
                    ax.plot(xs, ys, color=col, linestyle=ls, linewidth=1.4 if comp == "ma" else 1.0, zorder=4)
                    ax.scatter(xs, ys, s=22, marker=mk, color=col, edgecolors=col, linewidths=1.0, zorder=5)
            ax.set_yscale("log")
            c.set_categorical_x(ax, c.CPUS, pos)
            ax.set_title(f"{c.DEVICE_LABEL[dev]} · {n_st} st · Stage C staggered arrivals")
            if ri == len(c.STATIONS) - 1:
                ax.set_xlabel(r"Host Logical CPU Allocation $C$")
            if ci == 0:
                ax.set_ylabel("p95 (finish − ready) (s)")
            c.tidy_axis(ax)
    from matplotlib.lines import Line2D
    extra = [Line2D([0], [0], color=c.INK, marker="o", linewidth=1.5, markersize=4.5, label="Model-Actor (num = $K$ carried from Stage B)"),
             Line2D([0], [0], color=c.INK, marker="s", linewidth=1.0, linestyle="--", markersize=4.5, label="Ripper")]
    c.model_legend(fig, extra=extra, ncol=4, y=-0.02)
    fig.text(0.5, -0.06, f"Stage C makespan is not plotted: every cell finished in {min(all_ms):.1f}-{max(all_ms):.1f} s ({len(all_ms)} repeats), set by the latest 90 s station delay.", ha="center", va="top", fontsize=6.8, color=c.MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, 192)
    print(f"Stage C makespan range {min(all_ms):.2f}-{max(all_ms):.2f} s over {len(all_ms)} repeats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
