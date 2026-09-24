#!/usr/bin/env python3
"""Fig 4 - Batch size.

x = B {64,128,256,512}; y = inference time.  Locked: merged, C=20,
thread_mult=1.0, winner dtype.  Dashed = best over (mult, dtype).
"""
from __future__ import annotations

import common as c

STEM = "fig05_inference_vs_batch_size"


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()

    def locked(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, packaging=c.REF_PACKAGING,
                    n_cpus=c.REF_CPUS, thread_mult=c.REF_MULT,
                    dtype=c.ref_dtype(winners, model, n_st, dev))

    def best(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, packaging=c.REF_PACKAGING, n_cpus=c.REF_CPUS)

    c.native_runtime_figure(
        stem=STEM, rows=rows, winners=winners, x_key="batch_size", xs=c.BATCHES,
        locked_fixed=locked, best_fixed=best, log_y=True,
        x_label=r"Batch Size $B$ (Waveform windows per forward pass)",
        y_label="Inference Time (s)",
        series_labels=("Mean quality-safe config", "Minimum quality-safe config"),
        align_x=True,
        title_fn=lambda n, d: f"{c.DEVICE_LABEL[d]} · {n} st · Merged · C=20 · T=C",
        legend_y=-0.02,
        legend_ncol=4,
        dtype_markers=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
