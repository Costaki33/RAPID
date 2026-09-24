#!/usr/bin/env python3
"""Fig 2 - Host-CPU allocation.

x = C {5,10,15,20}; y = inference time.  Mean line only: Merged, batch=512,
thread_mult=1.0 (T = C), winner dtype at C=20.
"""
from __future__ import annotations

import common as c

STEM = "fig03_inference_vs_host_cpus"


def main() -> int:
    rows = c.load_native_rows()
    winners = c.load_winners()

    def locked(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, packaging=c.REF_PACKAGING,
                    batch_size=c.REF_BATCH, thread_mult=c.REF_MULT,
                    dtype=c.ref_dtype(winners, model, n_st, dev))

    def best(model, n_st, dev):
        return dict(model=model, n_stations=n_st, device=dev, packaging=c.REF_PACKAGING)

    c.native_runtime_figure(
        stem=STEM, rows=rows, winners=winners, x_key="n_cpus", xs=c.CPUS,
        locked_fixed=locked, best_fixed=best, log_y=True,
        x_label="Host Logical CPU Allocation $C$",
        y_label="Inference Time (s)",
        show_best=False, lines_only=True,
        title_fn=lambda n, d: f"{c.DEVICE_LABEL[d]} · {n} st · Merged · B=512 · T=C",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
