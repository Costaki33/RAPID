#!/usr/bin/env python3
"""Fig 5 - Numerical format.

x = dtype {FP32, FP16, BF16}; y = inference time.  Locked: merged, C=20,
batch=512, thread_mult=1.0.  Hollow markers = not quality-safe
(Versus-FP32 P F1 < 0.99).  EQTransformer / EQT-NC have no FP16 cell.
Dashed = best over (batch, mult) for that dtype.
"""
from __future__ import annotations

import common as c

STEM = "fig06_inference_vs_dtype"


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()

    def locked(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, packaging=c.REF_PACKAGING,
                    n_cpus=c.REF_CPUS, thread_mult=c.REF_MULT, batch_size=c.REF_BATCH)

    def best(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, packaging=c.REF_PACKAGING, n_cpus=c.REF_CPUS)

    c.native_runtime_figure(
        stem=STEM, rows=rows, winners=winners, x_key="dtype", xs=c.DTYPES,
        locked_fixed=locked, best_fixed=best, log_y=True,
        x_label="Numerical Precision",
        y_label="Inference Time (s)",
        x_tick_labels=[c.DTYPE_LABEL[d] for d in c.DTYPES],
        hollow_unsafe=True,
        align_x=True,
        unsafe_marker="x",
        unsafe_label="Mean, not quality-safe",
        best_linestyle=":",
        best_marker="o",
        series_labels=("Mean quality-safe config", "Minimum quality-safe config"),
        legend_ncol=3,
        dense_log_yticks=True,
        title_fn=lambda n, d: f"{c.DEVICE_LABEL[d]} · {n} st · Merged · C=20 · B=512 · T=C",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
