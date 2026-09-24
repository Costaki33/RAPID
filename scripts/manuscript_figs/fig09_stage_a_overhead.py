#!/usr/bin/env python3
"""Fig 10 - Stage A: one RAPID actor versus optimized inference time.

x = host allocation C.  y = RAPID makespan (K=1, optimized thread count)
/ single-process inference time.  Dots = 5 repeats, line at 1.0.
"""
from __future__ import annotations

import common as c

STEM = "fig10_stage_a_rapid_over_native"


def main() -> int:
    rows = c.load_orch_rows(stages=("A",))
    c.configure_style()
    fig, axes = c.panel_grid(c.STATIONS, c.DEVICES, width=6.9, row_height=2.35, sharey="row")
    pos = c.x_positions(c.CPUS)
    csv_rows = []
    n_pts = 0
    for ri, n_st in enumerate(c.STATIONS):
        for ci, dev in enumerate(c.DEVICES):
            ax = axes[ri][ci]
            ymax = 0.0
            for mi, model in enumerate(c.MODELS):
                locked = {}
                for n_cpus in c.CPUS:
                    sub = c.select(rows, stage="A", model=model, n_stations=n_st, device=dev, n_cpus=n_cpus)
                    if not sub:
                        continue
                    inference_s = sub[0]["native_inference_s_mean"]
                    vals = [r["makespan_s"] / inference_s for r in sub if r["makespan_s"] is not None]
                    locked[n_cpus] = vals
                    for r in sub:
                        csv_rows.append({
                            "model": model, "n_stations": n_st, "device": dev, "n_cpus": n_cpus, "k": 1,
                            "torch_threads": r["torch_threads"], "repeat_index": r["repeat_index"],
                            "makespan_s": r["makespan_s"], "inference_s_mean": inference_s,
                            "ratio": r["makespan_s"] / inference_s,
                        })
                        n_pts += 1
                ymax = max(ymax, c.plot_series(
                    ax, xs=c.CPUS, pos=pos, locked=locked, color=c.MODEL_COLOR[model],
                    model_index=mi, n_models=len(c.MODELS), align_x=True,
                ))
            ax.axhline(1.0, color=c.MUTED, linestyle=":", linewidth=1.0, zorder=1)
            c.set_categorical_x(ax, c.CPUS, pos)
            ax.set_title(f"{c.DEVICE_LABEL[dev]} · {n_st} st · Stage A (K=1, optimized $T$)")
            if ri == len(c.STATIONS) - 1:
                ax.set_xlabel(r"Host Logical CPU Allocation $C$")
            if ci == 0:
                ax.set_ylabel("Makespan Ratio\n(RAPID Makespan / Inference Time)")
                ax.set_ylim(0, ymax * 1.12)
            c.tidy_axis(ax)
    c.model_legend(fig, ncol=6, y=-0.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, len(c.MODELS) * len(c.STATIONS) * len(c.DEVICES) * len(c.CPUS) * 5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
