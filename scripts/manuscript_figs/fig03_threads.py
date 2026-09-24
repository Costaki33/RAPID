#!/usr/bin/env python3
"""Fig 3 - Intra-operation thread multiplier.

x = m {0.25, 0.5, 1.0, 1.5, 2.0} (T = round-half-up(m*C)); y = inference time.
580-station network only.  Panel rows = host allocation C, columns = device.
Locked: merged, batch=512, winner dtype.  Dashed = best over (batch, dtype).
"""
from __future__ import annotations

import common as c

STEM = "fig04_inference_vs_thread_mult_580st"
N_ST = 580


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()

    # Row variable is C here; ``native_runtime_figure`` passes it as ``n_st``.
    def locked(model, n_cpus, dev):
        return dict(model=model, n_stations=N_ST, device=dev, n_cpus=n_cpus,
                    packaging=c.REF_PACKAGING, batch_size=c.REF_BATCH,
                    dtype=c.ref_dtype(winners, model, N_ST, dev, n_cpus))

    def best(model, n_cpus, dev):
        return dict(model=model, n_stations=N_ST, device=dev, n_cpus=n_cpus, packaging=c.REF_PACKAGING)

    c.native_runtime_figure(
        stem=STEM, rows=rows, winners=winners, x_key="thread_mult", xs=c.MULTS,
        locked_fixed=locked, best_fixed=best, stations=c.CPUS, log_y=True,
        x_label=r"Thread Multiplier $m$ ($T$ = round-half-up($m \cdot C$))",
        y_label="Inference Time (s)",
        align_x=True,
        x_tick_labels=[f"{m:g}" for m in c.MULTS],
        title_fn=lambda C, d: f"{c.DEVICE_LABEL[d]} · 580 st · C={C} · Merged · B=512",
        series_labels=("Mean quality-safe config", "Minimum quality-safe config"),
        legend_y=-0.02,
        legend_ncol=4,
        dtype_markers=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
