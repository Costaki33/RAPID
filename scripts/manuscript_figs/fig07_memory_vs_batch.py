#!/usr/bin/env python3
"""Fig 8 - Peak memory vs batch size (580 stations).

x = Batch Size B {64,128,256,512}.  Left/centre: Peak Host PSS (GB) for
CPU and GPU trials.  Right: Peak VRAM (GB) for GPU trials.  Locked:
Merged, C=20, thread_mult=1.0, winner dtype.  Dots: individual repeats.
"""
from __future__ import annotations

import common as c

STEM = "fig08_peak_memory_vs_batch_size_580st"
N_ST = 580
PANELS = (("cpu", "peak_pss_mb", "Peak Host PSS (GB)", "CPU trial"),
          ("gpu", "peak_pss_mb", "Peak Host PSS (GB)", "GPU trial"),
          ("gpu", "peak_vram_mb", "Peak VRAM (GB)", "GPU trial"))


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()
    c.configure_style()
    fig, axes = c.panel_grid([0], [p[3] for p in PANELS], width=7.0, row_height=2.6, sharey=False)
    pos = c.x_positions(c.BATCHES)
    csv_rows = []
    n_pts = 0
    for ci, (dev, key, ylabel, title) in enumerate(PANELS):
        ax = axes[0][ci]
        ymax = 0.0
        for mi, model in enumerate(c.MODELS):
            fixed = dict(model=model, n_stations=N_ST, device=dev, packaging=c.REF_PACKAGING,
                         n_cpus=c.REF_CPUS, thread_mult=c.REF_MULT, dtype=c.ref_dtype(winners, model, N_ST, dev))
            locked = c.locked_series(rows, x_key="batch_size", y_key=key, fixed=fixed)
            locked = {x: [v / 1000.0 for v in vals] for x, vals in locked.items()}
            for x, vals in locked.items():
                for i, v in enumerate(vals):
                    csv_rows.append({**fixed, "batch_size": x, "metric": key, "repeat_index": i, "value_gb": v})
                    n_pts += 1
            ymax = max(ymax, c.plot_series(ax, xs=c.BATCHES, pos=pos, locked=locked, color=c.MODEL_COLOR[model],
                                           model_index=mi, n_models=len(c.MODELS), align_x=True))
        c.set_categorical_x(ax, c.BATCHES, pos)
        ax.set_ylim(0, ymax * 1.12 if ymax else 1)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(r"Batch Size $B$")
        ax.set_title(f"{title}\n580 st · Merged · C=20 · T=C")
        c.tidy_axis(ax)
    c.model_legend(fig, ncol=6, y=-0.04)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, len(c.MODELS) * len(PANELS) * len(c.BATCHES) * 3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
