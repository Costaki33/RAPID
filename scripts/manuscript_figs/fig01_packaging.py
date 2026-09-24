#!/usr/bin/env python3
"""Fig 1 - Merged vs Sequential waveform packaging.

x = packaging {merged, sequential}; y = inference time (log).
Each point is one locked-config repeat (C=20, batch=512, thread_mult=1.0,
winner dtype). No connecting line and no best-over-other-settings series.
Timed-out sequential cells are up-arrows at the 1800 s ceiling.
"""
from __future__ import annotations

import common as c

STEM = "fig02_packaging_merged_vs_sequential"
XS = ("merged", "sequential")


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()

    def locked(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, n_cpus=c.REF_CPUS,
                    batch_size=c.REF_BATCH, thread_mult=c.REF_MULT,
                    dtype=c.ref_dtype(winners, model, n_st, dev))

    def best(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev)

    c.native_runtime_figure(
        stem=STEM, rows=rows, winners=winners, x_key="packaging", xs=XS,
        locked_fixed=locked, best_fixed=best,
        x_label="Waveform Packaging", x_tick_labels=("Merged-Network", "Sequential-Station"),
        y_label="Inference Time (s)",
        log_y=True, censor_timeouts=True, show_best=False, points_only=True,
        title_fn=lambda n, d: f"{c.DEVICE_LABEL[d]} · {n} st · C=20 · B=512 · T=C",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
