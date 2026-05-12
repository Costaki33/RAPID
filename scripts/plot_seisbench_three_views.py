"""Publication figures: CPU baseline (single axes), 3×2 affinity scaling, speedup bars.

Uses shared styling in :mod:`seisbench_plot_style` (model colors, N_st markers,
annotate dash pattern, lean linestyles by dtype + compile).

- **cpu_baseline_affinity_facets** — one axes: CPU ``annotate()`` only, x = pinned
  cores 12/16/20; color = model; marker = N_st; linestyle = annotate pattern.

- **gpu_scaling_log** — 3×2 grid: rows = N_st, cols = CPU | GPU; x = 12/16/20
  host CPUs; y = log median wall time. Same color/marker rules; lean linestyles
  from dtype + compile. GPU column uses ``aff12``, ``aff16``, and partial ``aff20``.

- **gpu16_speedup_vs_baseline_bars** — grouped bars per model at N_st=580,
  cuda:0, 16 host CPUs: annotate = 1.0 plus FP16/BF16 with and without
  compile (bars omitted when no timed rows, e.g. FP16+compile).

Run from ``RAPID/``::

    python scripts/plot_seisbench_three_views.py
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from seisbench_plot_style import (
    AFF_CPUS,
    BAR_ANNOTATE,
    BAR_BF16_OFF,
    BAR_BF16_ON,
    BAR_FP16_OFF,
    BAR_FP16_ON,
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

SPEEDUP_NST = 580

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


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("kind") == "env":
                continue
            rows.append(r)
    return rows


def agg_median_by(
    rows: list[dict],
    filt,
    keys: tuple[str, ...],
) -> dict[tuple, float]:
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if _skipped(r):
            continue
        if not filt(r):
            continue
        wt = r.get("wall_time_s")
        if wt is None or wt <= 0:
            continue
        key = tuple(r.get(k) for k in keys)
        buckets[key].append(float(wt))
    return {k: _median(v) for k, v in buckets.items()}


def _one_median_for(rows: list[dict], filt) -> float:
    vals: list[float] = []
    for r in rows:
        if _skipped(r):
            continue
        if not filt(r):
            continue
        wt = r.get("wall_time_s")
        if wt is None or wt <= 0:
            continue
        vals.append(float(wt))
    return _median(vals)


def _affinity_curve(
    by_aff: dict[int, list[dict]],
    affinities: list[int],
    filt,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for n in affinities:
        rows = by_aff.get(n)
        if not rows:
            continue
        v = _one_median_for(rows, filt)
        if v == v:
            xs.append(float(n))
            ys.append(v)
    return xs, ys


def plot_cpu_baseline_combined(by_cpu_aff: dict[int, list[dict]]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    mk = NST_MARKERS
    for model in MODEL_ORDER:
        c = MODEL_COLORS[model]
        for ns in NST_ORDER:
            xs, ys = _affinity_curve(
                by_cpu_aff,
                AFF_CPUS,
                lambda r, m=model, nst=ns: (
                    r.get("runner") == "baseline_annotate"
                    and r.get("device") == "cpu"
                    and str(r.get("model_label")) == m
                    and int(r.get("n_stations") or 0) == nst
                ),
            )
            if not xs:
                continue
            if len(xs) >= 2:
                ax.plot(
                    xs,
                    ys,
                    linestyle=LS_ANNOTATE,
                    color=c,
                    marker=mk[ns],
                    ms=7,
                    lw=1.6,
                )
            else:
                ax.scatter(xs, ys, s=55, marker=mk[ns], color=c, zorder=4)
    ax.set_xticks(AFF_CPUS)
    ax.set_xlabel("Pinned CPU cores")
    ax.set_ylabel("Median wall time (s)")
    ax.grid(True, ls=":", alpha=0.45)
    leg_handles = [
        mlines.Line2D([], [], color=MODEL_COLORS[m], marker="o", ls=LS_ANNOTATE, lw=1.6, label=m)
        for m in MODEL_ORDER
    ]
    for ns in NST_ORDER:
        leg_handles.append(
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker=NST_MARKERS[ns],
                ls="",
                markersize=8,
                label=f"N_st={ns}",
            )
        )
    leg_handles.append(mlines.Line2D([], [], color="black", ls=LS_ANNOTATE, lw=1.8, label="annotate()"))
    ax.legend(handles=leg_handles, fontsize=7, ncol=2, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_DIR / "cpu_baseline_affinity_facets"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_scaling_panel(
    ax,
    *,
    ns: int,
    device: str,
    by_aff: dict[int, list[dict]],
    affinities: list[int],
) -> None:
    mk = NST_MARKERS[ns]

    def ann_filt(r, m: str, nst: int = ns, dev: str = device):
        return (
            r.get("runner") == "baseline_annotate"
            and r.get("device") == dev
            and str(r.get("model_label")) == m
            and int(r.get("n_stations") or 0) == nst
        )

    for model in MODEL_ORDER:
        c = MODEL_COLORS[model]
        xs, ys = _affinity_curve(by_aff, affinities, lambda r, m=model: ann_filt(r, m))
        if len(xs) >= 2:
            ax.plot(xs, ys, ls=LS_ANNOTATE, color=c, marker=mk, ms=6, lw=1.5)
        elif xs:
            ax.scatter(xs, ys, s=40, marker=mk, color=c, zorder=4)

    lean_specs: list[tuple[str, bool]] = [
        ("fp16", False),
        ("fp16", True),
        ("bf16", False),
        ("bf16", True),
    ]

    for model in MODEL_ORDER:
        c = MODEL_COLORS[model]
        for dtype, compiled in lean_specs:
            if dtype == "fp16" and model.startswith("EQ"):
                continue
            if device == "cpu" and compiled:
                continue
            ls = lean_linestyle(dtype, compiled)

            def lf(
                r,
                m=model,
                nst=ns,
                dev=device,
                dt=dtype,
                comp=compiled,
            ):
                if r.get("runner") != "lean_pytorch" or r.get("device") != dev:
                    return False
                if str(r.get("model_label")) != m:
                    return False
                if int(r.get("n_stations") or 0) != nst:
                    return False
                if str(r.get("dtype") or "").lower() != dt:
                    return False
                return _compile_on(r) == comp

            xs, ys = _affinity_curve(by_aff, affinities, lf)
            if not xs:
                continue
            if len(xs) >= 2:
                ax.plot(xs, ys, ls=ls, color=c, marker=mk, ms=5, lw=1.3)
            else:
                ax.scatter(xs, ys, s=36, marker=mk, color=c, zorder=3)

    ax.set_xticks(affinities)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.grid(True, which="both", ls=":", alpha=0.4)


def plot_scaling_grid(by_cpu_aff: dict[int, list[dict]], by_gpu_aff: dict[int, list[dict]]) -> None:
    aff_cpu = [n for n in AFF_CPUS if by_cpu_aff.get(n)]
    aff_gpu = [n for n in AFF_CPUS if by_gpu_aff.get(n)]
    fig, axes = plt.subplots(3, 2, figsize=(9.8, 11), sharex=False)
    for ri, ns in enumerate(NST_ORDER):
        _plot_scaling_panel(
            axes[ri, 0], ns=ns, device="cpu", by_aff=by_cpu_aff, affinities=aff_cpu
        )
        _plot_scaling_panel(
            axes[ri, 1], ns=ns, device="cuda:0", by_aff=by_gpu_aff, affinities=aff_gpu
        )
        axes[ri, 0].set_ylabel(f"N_st={ns}\nlog time (s)")
    axes[2, 0].set_xlabel("Pinned host CPU cores (CPU trials)")
    axes[2, 1].set_xlabel("Pinned host CPU cores (GPU trials)")
    handles = [
        mlines.Line2D([], [], color="black", ls=LS_ANNOTATE, lw=1.8, label="annotate()"),
    ]
    for dtype, compiled in [("fp16", False), ("fp16", True), ("bf16", False), ("bf16", True)]:
        lab = f"{dtype.upper()}" + (" +compile" if compiled else "")
        handles.append(
            mlines.Line2D([], [], color="gray", ls=lean_linestyle(dtype, compiled), lw=1.6, label=lab)
        )
    for m in MODEL_ORDER:
        handles.append(mpatches.Patch(color=MODEL_COLORS[m], label=m))
    for ns in NST_ORDER:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="k",
                marker=NST_MARKERS[ns],
                ls="",
                markersize=7,
                label=f"markers: N_st={ns}",
            )
        )
    fig.legend(handles=handles, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = OUT_DIR / "gpu16_scaling_log"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_speedup_bars_grid(
    by_cpu_aff: dict[int, list[dict]],
    by_gpu_aff: dict[int, list[dict]],
) -> None:
    """3×3 panels: rows = CPU / 1-GPU / 2-GPU; cols = host affinity 12, 16, 20."""
    aff_cols = AFF_CPUS
    row_modes = ("cpu", "1gpu", "2gpu")
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 9.5), sharey="row")
    used_legend_keys: set[str] = set()

    def _base_median(rows: list[dict], mode: str, n_cpus: int, model: str) -> float:
        if mode == "cpu":

            def bf(r, m=model, nc=n_cpus):
                return (
                    r.get("runner") == "baseline_annotate"
                    and r.get("device") == "cpu"
                    and int(r.get("process_n_cpus") or 0) == nc
                    and int(r.get("n_stations") or 0) == SPEEDUP_NST
                    and str(r.get("model_label")) == m
                )

        elif mode == "1gpu":

            def bf(r, m=model, nc=n_cpus):
                return (
                    r.get("runner") == "baseline_annotate"
                    and r.get("device") == "cuda:0"
                    and int(r.get("process_n_cpus") or 0) == nc
                    and int(r.get("n_stations") or 0) == SPEEDUP_NST
                    and str(r.get("model_label")) == m
                )

        else:

            def bf(r, m=model, nc=n_cpus):
                return (
                    r.get("runner") == "baseline_annotate_dual"
                    and r.get("device") == "cuda:0+cuda:1"
                    and int(r.get("process_n_cpus") or 0) == nc
                    and int(r.get("n_stations") or 0) == SPEEDUP_NST
                    and str(r.get("model_label")) == m
                )

        return _one_median_for(rows, bf)

    def _lean_median(
        rows: list[dict],
        mode: str,
        n_cpus: int,
        model: str,
        dtype: str,
        compiled: bool,
    ) -> float:
        dt = dtype.lower()

        if mode == "cpu":

            def lf(r, m=model, nc=n_cpus, d=dt, c=compiled):
                return (
                    r.get("runner") == "lean_pytorch"
                    and r.get("device") == "cpu"
                    and int(r.get("process_n_cpus") or 0) == nc
                    and int(r.get("n_stations") or 0) == SPEEDUP_NST
                    and str(r.get("model_label")) == m
                    and str(r.get("dtype") or "").lower() == d
                    and _compile_on(r) == c
                )

        elif mode == "1gpu":

            def lf(r, m=model, nc=n_cpus, d=dt, c=compiled):
                return (
                    r.get("runner") == "lean_pytorch"
                    and r.get("device") == "cuda:0"
                    and int(r.get("process_n_cpus") or 0) == nc
                    and int(r.get("n_stations") or 0) == SPEEDUP_NST
                    and str(r.get("model_label")) == m
                    and str(r.get("dtype") or "").lower() == d
                    and _compile_on(r) == c
                )

        else:

            def lf(r, m=model, nc=n_cpus, d=dt, c=compiled):
                return (
                    r.get("runner") == "lean_pytorch_dual_pipelined"
                    and r.get("device") == "cuda:0+cuda:1"
                    and int(r.get("process_n_cpus") or 0) == nc
                    and int(r.get("n_stations") or 0) == SPEEDUP_NST
                    and str(r.get("model_label")) == m
                    and str(r.get("dtype") or "").lower() == d
                    and _compile_on(r) == c
                )

        return _one_median_for(rows, lf)

    def _spd(b: float, l: float) -> float:
        if b != b or l != l or l <= 0:
            return float("nan")
        return b / l

    for ri, mode in enumerate(row_modes):
        for ci, n_cpus in enumerate(aff_cols):
            ax = axes[ri, ci]
            rows = by_cpu_aff[n_cpus] if mode == "cpu" else by_gpu_aff.get(n_cpus) or []
            if not rows:
                ax.set_visible(False)
                continue

            x_models = np.arange(len(MODEL_ORDER), dtype=float)
            bar_w = 0.07

            for mi, model in enumerate(MODEL_ORDER):
                base = _base_median(rows, mode, n_cpus, model)
                if base != base:
                    continue
                eqt = model.startswith("EQ")
                center = x_models[mi]

                bars_plot: list[tuple[float, str, str]] = []
                bars_plot.append((1.0, BAR_ANNOTATE, "ann"))
                if not eqt:
                    v = _spd(base, _lean_median(rows, mode, n_cpus, model, "fp16", False))
                    if v == v:
                        bars_plot.append((v, BAR_FP16_OFF, "fp16"))
                    v = _spd(base, _lean_median(rows, mode, n_cpus, model, "fp16", True))
                    if v == v:
                        bars_plot.append((v, BAR_FP16_ON, "fp16_tc"))
                v = _spd(base, _lean_median(rows, mode, n_cpus, model, "bf16", False))
                if v == v:
                    bars_plot.append((v, BAR_BF16_OFF, "bf16"))
                v = _spd(base, _lean_median(rows, mode, n_cpus, model, "bf16", True))
                if v == v:
                    bars_plot.append((v, BAR_BF16_ON, "bf16_tc"))

                n_b = len(bars_plot)
                offsets = (np.arange(n_b) - (n_b - 1) / 2.0) * bar_w
                for j, (height, color, key) in enumerate(bars_plot):
                    if height != height or height <= 0:
                        continue
                    ax.bar(
                        center + offsets[j],
                        height,
                        bar_w * 0.92,
                        color=color,
                        edgecolor="black",
                        linewidth=0.3,
                    )
                    used_legend_keys.add(key)

            ax.axhline(1.0, color="#bbbbbb", linestyle=(0, (4, 3)), linewidth=0.85, zorder=0)
            ax.set_xticks(x_models)
            ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right", fontsize=8)
            ax.grid(axis="y", ls=":", alpha=0.4)
            if ri == 0:
                ax.set_title(f"{n_cpus} host CPUs", fontsize=10)
            if ci == 0:
                ylab = {"cpu": "CPU device", "1gpu": "1 GPU", "2gpu": "2 GPU"}[mode]
                ax.set_ylabel(f"{ylab}\nSpeedup", fontsize=9)

    proxy_map = {
        "ann": mpatches.Patch(facecolor=BAR_ANNOTATE, edgecolor="black", linewidth=0.35, label="annotate()"),
        "fp16": mpatches.Patch(facecolor=BAR_FP16_OFF, edgecolor="black", linewidth=0.35, label="FP16"),
        "fp16_tc": mpatches.Patch(facecolor=BAR_FP16_ON, edgecolor="black", linewidth=0.35, label="FP16+compile"),
        "bf16": mpatches.Patch(facecolor=BAR_BF16_OFF, edgecolor="black", linewidth=0.35, label="BF16"),
        "bf16_tc": mpatches.Patch(facecolor=BAR_BF16_ON, edgecolor="black", linewidth=0.35, label="BF16+compile"),
    }
    order_keys = ["ann", "fp16", "fp16_tc", "bf16", "bf16_tc"]
    handles = [proxy_map[k] for k in order_keys if k in used_legend_keys]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=8, frameon=False)
    fig.text(
        0.5,
        0.01,
        f"Median wall-time speedup (annotate / lean) at N_st={SPEEDUP_NST}. "
        "EQT-family FP16 omitted. Bars omitted when baseline or lean medians are missing.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    base_path = OUT_DIR / f"speedup_vs_baseline_bars_{SPEEDUP_NST}_stations"
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_cpu_aff = {n: load_jsonl(RESULTS / f"seisbench_matrix_lean_cpu_aff{n}.jsonl") for n in AFF_CPUS}
    by_gpu_aff = {n: load_jsonl(RESULTS / f"seisbench_matrix_lean_aff{n}.jsonl") for n in AFF_CPUS}

    missing = [n for n in AFF_CPUS if not by_cpu_aff[n]]
    if missing:
        raise SystemExit(f"Missing CPU affinity JSONL for: {missing}")

    plot_cpu_baseline_combined(by_cpu_aff)
    plot_scaling_grid(by_cpu_aff, by_gpu_aff)

    plot_speedup_bars_grid(by_cpu_aff, by_gpu_aff)

    # print("Wrote:", OUT_DIR / "cpu_baseline_affinity_facets.pdf")
    # print("Wrote:", OUT_DIR / "gpu16_scaling_log.pdf")
    # print("Wrote:", OUT_DIR / "speedup_vs_baseline_bars.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
