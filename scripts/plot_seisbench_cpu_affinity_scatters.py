"""Scatter panels for CPU affinity SeisBench trials (baseline annotate + CPU lean).

Reads ``results/seisbench_matrix_lean_cpu_aff{12,16,20}.jsonl`` and writes:

- ``figures/affinity/cpu_baseline_annotate_by_affinity.{pdf,png}``
- ``figures/affinity/cpu_lean_bf16_by_affinity.{pdf,png}`` (BF16 + FP16 CPU lean only)

Run from ``RAPID/``::

    python scripts/plot_seisbench_cpu_affinity_scatters.py

No plot titles (paper adds captions). X = station count at measured points only;
series are drawn as straight segments between consecutive points.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from seisbench_plot_style import (
    AFF_CPUS,
    LS_ANNOTATE,
    MODEL_COLORS,
    MODEL_ORDER,
    NST_MARKERS,
    NST_ORDER,
    lean_linestyle,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_DIR = ROOT / "figures" / "affinity"


def _median(vals: list[float]) -> float:
    good = sorted(x for x in vals if x is not None and x > 0 and x == x)
    if not good:
        return float("nan")
    return float(statistics.median(good))


def _skipped(row: dict) -> bool:
    st = row.get("benchmark_status")
    if st and "skip" in str(st).lower():
        return True
    return bool(row.get("skip_reason") or row.get("is_skipped_incompatible"))


def _compile_on(row: dict) -> bool:
    be = row.get("backend_extra") or {}
    return (
        row.get("torch_compile") is True
        or row.get("compile") is True
        or be.get("torch_compile") is True
        or be.get("compile") is True
    )


def load_cpu_jsonl(n_cpus: int) -> list[dict]:
    path = RESULTS / f"seisbench_matrix_lean_cpu_aff{n_cpus}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "env":
                continue
            rows.append(r)
    return rows


def median_wall_by_model_nst(
    rows: list[dict],
    *,
    filt,
) -> dict[tuple[str, int], float]:
    buckets: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        if _skipped(r):
            continue
        if r.get("device") != "cpu":
            continue
        if not filt(r):
            continue
        wt = r.get("wall_time_s")
        if wt is None or wt <= 0:
            continue
        m = r.get("model_label")
        ns = int(r.get("n_stations") or 0)
        buckets[(str(m), ns)].append(float(wt))
    return {k: _median(v) for k, v in buckets.items()}


def plot_baseline_panels() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, ncpu in zip(axes, AFF_CPUS):
        rows = load_cpu_jsonl(ncpu)

        def flt(r):
            return r.get("runner") == "baseline_annotate" and r.get("device") == "cpu"

        med = median_wall_by_model_nst(rows, filt=flt)
        for model in MODEL_ORDER:
            c = MODEL_COLORS[model]
            xs: list[int] = []
            ys: list[float] = []
            for ns in NST_ORDER:
                y = med.get((model, ns))
                if y == y:
                    xs.append(ns)
                    ys.append(y)
            if len(xs) >= 2:
                ax.plot(xs, ys, linestyle=LS_ANNOTATE, color=c, lw=1.4)
            for ns in NST_ORDER:
                y = med.get((model, ns))
                if y == y:
                    ax.scatter(
                        [ns],
                        [y],
                        color=c,
                        marker=NST_MARKERS[ns],
                        s=44,
                        zorder=4,
                    )
        ax.set_xticks(NST_ORDER)
        ax.set_xlabel("Number of stations")
        ax.grid(True, ls=":", alpha=0.45)
    axes[0].set_ylabel("Wall time (s)")
    leg_b = [
        mlines.Line2D(
            [],
            [],
            color=MODEL_COLORS[m],
            ls=LS_ANNOTATE,
            lw=1.6,
            label=m,
        )
        for m in MODEL_ORDER
    ]
    for ns in NST_ORDER:
        leg_b.append(
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker=NST_MARKERS[ns],
                ls="",
                ms=8,
                label=f"N_st={ns}",
            )
        )
    leg_b.append(
        mlines.Line2D(
            [],
            [],
            color="black",
            ls=LS_ANNOTATE,
            lw=2.0,
            label="annotate()",
        )
    )
    fig.legend(handles=leg_b, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "cpu_baseline_annotate_by_affinity"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_lean_precision_panels() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    dtypes = ("fp16", "bf16")
    for ax, ncpu in zip(axes, AFF_CPUS):
        rows = load_cpu_jsonl(ncpu)

        def flt(r, dt: str):
            return (
                r.get("runner") == "lean_pytorch"
                and r.get("device") == "cpu"
                and not _compile_on(r)
                and str(r.get("dtype") or "").lower() == dt
            )

        for model in MODEL_ORDER:
            for dt in dtypes:
                if model.startswith("EQ") and dt == "fp16":
                    continue
                med = median_wall_by_model_nst(rows, filt=lambda r, dt=dt: flt(r, dt))
                ls = lean_linestyle(dt, False)
                c = MODEL_COLORS[model]
                xs: list[int] = []
                ys: list[float] = []
                for ns in NST_ORDER:
                    y = med.get((model, ns))
                    if y == y:
                        xs.append(ns)
                        ys.append(y)
                if len(xs) >= 2:
                    ax.plot(xs, ys, linestyle=ls, color=c, lw=1.2)
                for ns in NST_ORDER:
                    y = med.get((model, ns))
                    if y == y:
                        ax.scatter(
                            [ns],
                            [y],
                            color=c,
                            marker=NST_MARKERS[ns],
                            s=44,
                            zorder=3,
                        )
        ax.set_xticks(NST_ORDER)
        ax.set_xlabel("Number of stations")
        ax.grid(True, ls=":", alpha=0.45)
    axes[0].set_ylabel("Wall time (s)")

    leg = [
        mlines.Line2D(
            [],
            [],
            color="gray",
            ls=lean_linestyle("fp16", False),
            lw=1.5,
            label="FP16 lean",
        ),
        mlines.Line2D(
            [],
            [],
            color="gray",
            ls=lean_linestyle("bf16", False),
            lw=1.5,
            label="BF16 lean",
        ),
    ]
    for m in MODEL_ORDER:
        leg.append(mlines.Line2D([], [], color=MODEL_COLORS[m], marker="o", ls="", ms=7, label=m))
    for ns in NST_ORDER:
        leg.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                marker=NST_MARKERS[ns],
                ls="",
                ms=8,
                label=f"N_st={ns}",
            )
        )
    fig.legend(handles=leg, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    base = OUT_DIR / "cpu_lean_bf16_by_affinity"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    plot_baseline_panels()
    plot_cpu_lean_precision_panels()
    print("Wrote:", OUT_DIR / "cpu_baseline_annotate_by_affinity.pdf")
    print("Wrote:", OUT_DIR / "cpu_lean_bf16_by_affinity.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
