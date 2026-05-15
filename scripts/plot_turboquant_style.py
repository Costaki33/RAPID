"""TurboQuant-inspired comparison figures for SeisBench matrix results.

Visual analogues (not the same metrics as TurboQuant):
  Fig.~1 style  : overlaid error distributions (absolute P-onset offset, samples).
  Fig.~2 style  : quality vs workload scale (lines + markers).
  Fig.~3 style  : ``rate'' (dtype bits) vs wall time with marginal quality strip.
  Fig.~4/5 style: grouped bars -- relative latency and speed score by model.

Reads: results/seisbench_matrix_lean_aff16.jsonl (GPU, 16-CPU affinity by default).

Run from repo:  python RAPID/scripts/plot_turboquant_style.py [--model PhaseNet] [--skip-grid] [--only-grid]

  --model NAME    Model for the 3x3 wall-time IQR grid (PhaseNet, PhaseNetLight, EQTransformer, EQT-NC).
  --skip-grid     Do not emit grid figures (only the five plots from a single GPU JSONL).
  --only-grid     Only emit the grid (requires six affinity JSONL files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- paths -----------------------------------------------------------------
_RAPID = Path(__file__).resolve().parent.parent
if str(_RAPID) not in sys.path:
    sys.path.insert(0, str(_RAPID))

RESULTS_DIR = _RAPID / "results"
OUT_DIR = _RAPID / "figures" / "turboquant_style"

MODEL_ORDER = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
MODEL_COLORS = {
    "PhaseNet": "#2563eb",
    "PhaseNetLight": "#ea580c",
    "EQTransformer": "#16a34a",
    "EQT-NC": "#dc2626",
}

N_ST_LIST = [64, 256, 580]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def _q25_q75(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return float(np.percentile(xs, 25)), float(np.percentile(xs, 75))


def abs_p_onset_error(r: Dict[str, Any]) -> float | None:
    pq = r.get("pick_quality")
    if not pq or "onset_delta_p_vs_catalog" not in pq:
        return None
    return abs(float(pq["onset_delta_p_vs_catalog"]))


def collect_errors(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    runner: str,
    dtype: str,
    n_stations: int,
    dataset: str = "stead",
    repeat: int = 0,
    lean_batch: int = 256,
) -> List[float]:
    out: List[float] = []
    for r in rows:
        if r.get("model_label") != model:
            continue
        if r.get("runner") != runner:
            continue
        if r.get("dtype") != dtype:
            continue
        if r.get("n_stations") != n_stations:
            continue
        if r.get("dataset_label") != dataset:
            continue
        if r.get("repeat") != repeat:
            continue
        if runner == "lean_pytorch":
            if r.get("batch_size") != lean_batch:
                continue
            if (r.get("backend_extra") or {}).get("compile"):
                continue
        elif runner == "baseline_annotate":
            if r.get("batch_size") != -1:
                continue
        e = abs_p_onset_error(r)
        if e is not None:
            out.append(e)
    return out


def collect_lean_pick_errors_pooled(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    dtype: str,
    dataset: str,
    n_stations_list: List[int] | None = None,
    repeats: Tuple[int, ...] = (0, 1),
    lean_batch: int = 256,
) -> List[float]:
    """All absolute P-onset errors for lean path: every N_st and repeat in lists."""
    if n_stations_list is None:
        n_stations_list = list(N_ST_LIST)
    out: List[float] = []
    for n_st in n_stations_list:
        for rep in repeats:
            out.extend(
                collect_errors(
                    rows,
                    model=model,
                    runner="lean_pytorch",
                    dtype=dtype,
                    n_stations=n_st,
                    dataset=dataset,
                    repeat=rep,
                    lean_batch=lean_batch,
                )
            )
    return out


def collect_wall_times(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    runner: str,
    dtype: str,
    n_stations: int,
    dataset: str = "stead",
    repeat: int = 0,
    lean_batch: int = 256,
) -> List[float]:
    out: List[float] = []
    for r in rows:
        if r.get("model_label") != model:
            continue
        if r.get("runner") != runner:
            continue
        if r.get("dtype") != dtype:
            continue
        if r.get("n_stations") != n_stations:
            continue
        if r.get("dataset_label") != dataset:
            continue
        if r.get("repeat") != repeat:
            continue
        if r.get("wall_time_s") is None:
            continue
        if runner == "lean_pytorch":
            if r.get("batch_size") != lean_batch:
                continue
            if (r.get("backend_extra") or {}).get("compile"):
                continue
        elif runner == "baseline_annotate":
            if r.get("batch_size") != -1:
                continue
        t = float(r["wall_time_s"])
        out.append(t)
    return out


def median_abs_p_error_pooled(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    runner: str,
    dtype: str,
    n_stations: int,
    lean_batch: int = 256,
) -> float:
    """Median |P onset error| over STEAD and TXED, repeat 0 and 1."""
    vals: List[float] = []
    for ds in ("stead", "txed"):
        for rep in (0, 1):
            for r in rows:
                if r.get("model_label") != model:
                    continue
                if r.get("runner") != runner:
                    continue
                if r.get("dtype") != dtype:
                    continue
                if r.get("n_stations") != n_stations:
                    continue
                if r.get("dataset_label") != ds:
                    continue
                if r.get("repeat") != rep:
                    continue
                if runner == "lean_pytorch":
                    if r.get("batch_size") != lean_batch:
                        continue
                    if (r.get("backend_extra") or {}).get("compile"):
                        continue
                elif runner == "baseline_annotate":
                    if r.get("batch_size") != -1:
                        continue
                e = abs_p_onset_error(r)
                if e is not None:
                    vals.append(e)
    return _median(vals)


def _wall_median_all_datasets(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    runner: str,
    dtype: str,
    n_stations: int,
    lean_batch: int = 256,
    compile_ok: bool = False,
) -> float:
    """Median wall time pooling stead+txed, both repeats, all batch sizes for baseline (-1 only)."""
    xs: List[float] = []
    for r in rows:
        if r.get("model_label") != model:
            continue
        if r.get("runner") != runner:
            continue
        if r.get("dtype") != dtype:
            continue
        if r.get("n_stations") != n_stations:
            continue
        if r.get("device") != "cuda:0":
            continue
        if r.get("wall_time_s") is None:
            continue
        extra = r.get("backend_extra") or {}
        if runner == "lean_pytorch":
            if bool(extra.get("compile")) != compile_ok:
                continue
            if r.get("batch_size") != lean_batch:
                continue
        if runner == "baseline_annotate":
            if r.get("batch_size") not in (-1, None):
                continue
        xs.append(float(r["wall_time_s"]))
    return _median(xs)


def figure_error_distributions(rows: List[Dict[str, Any]], out_dir: Path) -> List[Path]:
    """TurboQuant Fig.~1 analogue: overlaid error histograms (BF16 vs FP16 lean), per dataset.

    Pools all PhaseNet lean trials: every $N_{\\mathrm{st}}$ in ``N_ST_LIST``, both repeats,
    lean batch_size 256, no compile.
    """
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    model = "PhaseNet"
    paths: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    pool_note = (
        rf"all $N_{{\mathrm{{st}}}}\in$ {{{', '.join(str(x) for x in N_ST_LIST)}}}, "
        "repeats 0--1, lean batch size 256, no compile"
    )

    for dataset, ds_label in [("stead", "STEAD"), ("txed", "TXED")]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        bf = collect_lean_pick_errors_pooled(rows, model=model, dtype="bf16", dataset=dataset)
        fp = collect_lean_pick_errors_pooled(rows, model=model, dtype="fp16", dataset=dataset)

        hi = max(max(bf) if bf else [0], max(fp) if fp else [0], 1)
        bins = np.linspace(0, hi, 22)

        ax.hist(
            bf,
            bins=bins,
            alpha=0.55,
            label=f"Lean BF16 ($n={len(bf)}$)",
            color="#1d4ed8",
            density=True,
            histtype="stepfilled",
        )
        ax.hist(
            fp,
            bins=bins,
            alpha=0.5,
            label=f"Lean FP16 ($n={len(fp)}$)",
            color="#c2410c",
            density=True,
            histtype="stepfilled",
        )

        ax.set_xlabel(r"Absolute P-onset error $\|\Delta\|$ (samples @ 100 Hz)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Distribution of pick error ({model}, {ds_label}, {pool_note})"
        )
        ax.legend(frameon=True)
        fig.tight_layout()
        slug = dataset.lower()
        p = out_dir / f"tq_style_error_distribution_phasenet_{slug}.pdf"
        fig.savefig(p)
        fig.savefig(p.with_suffix(".png"), dpi=200)
        plt.close(fig)
        paths.append(p)

    sns.reset_defaults()
    return paths


def figure_quality_vs_scale(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """TurboQuant Fig.~2 analogue: lines with markers (quality vs station count)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    model = "PhaseNet"
    for dtype, color, mk in [("bf16", "#1d4ed8", "o"), ("fp16", "#c2410c", "s")]:
        ys, xs = [], []
        for n in N_ST_LIST:
            errs = collect_errors(
                rows, model=model, runner="lean_pytorch", dtype=dtype, n_stations=n
            )
            if errs:
                xs.append(n)
                ys.append(_median(errs))
        ax.plot(xs, ys, marker=mk, color=color, lw=2.2, label=f"Lean {dtype.upper()}")

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_ST_LIST)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"Stations $N_{\mathrm{st}}$")
    ax.set_ylabel(r"Median $|\Delta_{\mathrm{P}}|$ (samples)")
    ax.set_title(f"P-onset error vs workload scale ({model}, STEAD, lean)")
    ax.legend()
    fig.tight_layout()
    p = out_dir / "tq_style_quality_vs_scale.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    sns.reset_defaults()
    return p


def figure_rate_latency_tradeoff(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """TurboQuant Fig.~3 analogue: dtype / path vs wall time with matching quality strip."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.12},
        constrained_layout=True,
    )

    model = "PhaseNet"
    n_st = 580
    xs_labels: List[str] = []
    wall: List[float] = []
    qual: List[float] = []

    w_ann = _wall_median_all_datasets(
        rows, model=model, runner="baseline_annotate", dtype="fp32", n_stations=n_st
    )
    q_ann = median_abs_p_error_pooled(
        rows, model=model, runner="baseline_annotate", dtype="fp32", n_stations=n_st
    )
    xs_labels.append("FP32\nannotate()")
    wall.append(w_ann)
    qual.append(q_ann)

    for dtype in ("bf16", "fp16"):
        w = _wall_median_all_datasets(
            rows, model=model, runner="lean_pytorch", dtype=dtype, n_stations=n_st
        )
        q = median_abs_p_error_pooled(
            rows, model=model, runner="lean_pytorch", dtype=dtype, n_stations=n_st
        )
        xs_labels.append(f"Lean\n{dtype.upper()}")
        wall.append(w)
        qual.append(q)

    x_plot = np.arange(len(xs_labels))
    colors = ["#64748b", "#1d4ed8", "#c2410c"]
    ax_top.bar(x_plot, wall, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax_top.set_ylabel("Median wall time (s)")
    ax_top.set_title(
        rf"Format / path vs latency ({model}, $N_{{\mathrm{{st}}}}={n_st}$, GPU, median over STEAD+TXED)"
    )
    ax_top.text(
        0.02,
        0.97,
        "Lower panel: median |ΔP| pooled over STEAD/TXED and both repeats. "
        "annotate() vs lean may differ in how picks are surfaced in the harness.",
        transform=ax_top.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        color="#475569",
    )

    ax_bot.bar(x_plot, qual, color=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    ax_bot.set_ylabel(r"Median $|\Delta_{\mathrm{P}}|$ (samples)")
    ax_bot.set_xticks(x_plot)
    ax_bot.set_xticklabels(xs_labels, fontsize=9)
    ax_bot.set_xlabel("Configuration")

    p = out_dir / "tq_style_rate_latency_tradeoff.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    sns.reset_defaults()
    return p


def figure_walltime_bands(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    r"""Lines with shaded IQR: wall time vs N_st (TurboQuant-style bound band)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    model = "PhaseNet"

    def series(runner: str, dtype: str) -> Tuple[List[int], List[float], List[float], List[float]]:
        xs, meds, lo, hi = [], [], [], []
        for n in N_ST_LIST:
            ts = []
            for r in rows:
                if (
                    r.get("model_label") == model
                    and r.get("runner") == runner
                    and r.get("dtype") == dtype
                    and r.get("n_stations") == n
                    and r.get("device") == "cuda:0"
                    and r.get("wall_time_s") is not None
                ):
                    if runner == "lean_pytorch":
                        if r.get("batch_size") != 256:
                            continue
                        if (r.get("backend_extra") or {}).get("compile"):
                            continue
                    if runner == "baseline_annotate" and r.get("batch_size") != -1:
                        continue
                    ts.append(float(r["wall_time_s"]))
            if ts:
                q1, q3 = _q25_q75(ts)
                xs.append(n)
                meds.append(_median(ts))
                lo.append(q1)
                hi.append(q3)
        return xs, meds, lo, hi

    x_b, m_b, lo_b, hi_b = series("baseline_annotate", "fp32")
    x_l, m_l, lo_l, hi_l = series("lean_pytorch", "bf16")

    ax.fill_between(x_b, lo_b, hi_b, alpha=0.22, color="#64748b")
    ax.plot(x_b, m_b, "s-", color="#334155", lw=2.2, ms=7, label="FP32 annotate() (IQR band)")

    ax.fill_between(x_l, lo_l, hi_l, alpha=0.22, color="#1d4ed8")
    ax.plot(x_l, m_l, "o-", color="#1e3a8a", lw=2.2, ms=7, label="Lean BF16 (IQR band)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_ST_LIST)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"Stations $N_{\mathrm{st}}$")
    ax.set_ylabel("Wall time (s)")
    ax.set_yscale("log")
    ax.set_title(f"Wall time vs scale with dispersion bands ({model}, GPU cuda:0)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = out_dir / "tq_style_walltime_bands.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    sns.reset_defaults()
    return p


def _gather_curve_times(
    rows: List[Dict[str, Any]],
    *,
    model: str,
    n_stations: int,
    runner: str,
    device: str,
    dtype: str,
    compile_mode: str,
    batch_size: int,
) -> List[float]:
    """compile_mode: 'ignore' | 'off' | 'on' for backend_extra.compile."""
    out: List[float] = []
    for r in rows:
        if r.get("model_label") != model:
            continue
        if r.get("runner") != runner:
            continue
        if r.get("device") != device:
            continue
        if r.get("dtype") != dtype:
            continue
        if r.get("n_stations") != n_stations:
            continue
        if r.get("wall_time_s") is None:
            continue
        if r.get("batch_size") != batch_size:
            continue
        extra = r.get("backend_extra") or {}
        c = bool(extra.get("compile"))
        if compile_mode == "ignore":
            pass
        elif compile_mode == "off" and c:
            continue
        elif compile_mode == "on" and not c:
            continue
        out.append(float(r["wall_time_s"]))
    return out


def _curve_specs_for_panel(model: str, panel: str) -> List[Dict[str, Any]]:
    """panel: 'cpu' | 'single_gpu' | 'dual_gpu'."""
    has_fp16 = model not in ("EQTransformer", "EQT-NC")
    has_compile = model in ("PhaseNet", "PhaseNetLight")
    specs: List[Dict[str, Any]] = []

    if panel == "dual_gpu":
        specs.append(
            {
                "label": "annotate() FP32",
                "runner": "baseline_annotate_dual",
                "device": "cuda:0+cuda:1",
                "dtype": "fp32",
                "compile_mode": "ignore",
                "batch_size": -1,
                "color": "#334155",
                "fill": "#94a3b8",
                "marker": "s",
                "ls": "-",
            }
        )
        specs.append(
            {
                "label": "Lean BF16",
                "runner": "lean_pytorch_dual_pipelined",
                "device": "cuda:0+cuda:1",
                "dtype": "bf16",
                "compile_mode": "off",
                "batch_size": 256,
                "color": "#1e3a8a",
                "fill": "#1d4ed8",
                "marker": "o",
                "ls": "-",
            }
        )
        if has_fp16:
            specs.append(
                {
                    "label": "Lean FP16",
                    "runner": "lean_pytorch_dual_pipelined",
                    "device": "cuda:0+cuda:1",
                    "dtype": "fp16",
                    "compile_mode": "off",
                    "batch_size": 256,
                    "color": "#9a3412",
                    "fill": "#ea580c",
                    "marker": "^",
                    "ls": "-",
                }
            )
        return specs

    device = "cpu" if panel == "cpu" else "cuda:0"
    runner_baseline = "baseline_annotate"
    runner_lean = "lean_pytorch"

    specs.append(
        {
            "label": "annotate() FP32",
            "runner": runner_baseline,
            "device": device,
            "dtype": "fp32",
            "compile_mode": "ignore",
            "batch_size": -1,
            "color": "#334155",
            "fill": "#94a3b8",
            "marker": "s",
            "ls": "-",
        }
    )
    specs.append(
        {
            "label": "Lean BF16",
            "runner": runner_lean,
            "device": device,
            "dtype": "bf16",
            "compile_mode": "off",
            "batch_size": 256,
            "color": "#1e3a8a",
            "fill": "#1d4ed8",
            "marker": "o",
            "ls": "-",
        }
    )
    if has_compile:
        specs.append(
            {
                "label": "Lean BF16 + compile",
                "runner": runner_lean,
                "device": device,
                "dtype": "bf16",
                "compile_mode": "on",
                "batch_size": 256,
                "color": "#0f766e",
                "fill": "#5eead4",
                "marker": "o",
                "ls": "--",
            }
        )
    if has_fp16:
        specs.append(
            {
                "label": "Lean FP16",
                "runner": runner_lean,
                "device": device,
                "dtype": "fp16",
                "compile_mode": "off",
                "batch_size": 256,
                "color": "#9a3412",
                "fill": "#ea580c",
                "marker": "^",
                "ls": "-",
            }
        )
        if has_compile:
            specs.append(
                {
                    "label": "Lean FP16 + compile",
                    "runner": runner_lean,
                    "device": device,
                    "dtype": "fp16",
                    "compile_mode": "on",
                    "batch_size": 256,
                    "color": "#831843",
                    "fill": "#f472b6",
                    "marker": "^",
                    "ls": "--",
                }
            )
    return specs


def _plot_iqr_bands_on_ax(
    ax: plt.Axes,
    rows: List[Dict[str, Any]],
    *,
    model: str,
    specs: List[Dict[str, Any]],
    show_ylabel: bool = True,
) -> None:
    for sp in specs:
        xs: List[int] = []
        meds: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for n in N_ST_LIST:
            ts = _gather_curve_times(
                rows,
                model=model,
                n_stations=n,
                runner=sp["runner"],
                device=sp["device"],
                dtype=sp["dtype"],
                compile_mode=sp["compile_mode"],
                batch_size=sp["batch_size"],
            )
            if not ts:
                continue
            q1, q3 = _q25_q75(ts)
            xs.append(n)
            meds.append(_median(ts))
            lo.append(q1)
            hi.append(q3)
        if not xs:
            continue
        ax.fill_between(xs, lo, hi, alpha=0.2, color=sp["fill"])
        ax.plot(
            xs,
            meds,
            marker=sp["marker"],
            ls=sp["ls"],
            color=sp["color"],
            lw=2.0,
            ms=6,
            label=sp["label"],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(N_ST_LIST)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$N_{\mathrm{st}}$")
    if show_ylabel:
        ax.set_ylabel("Wall time (s)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.45)


def figure_walltime_bands_affinity_device_grid(model: str, out_dir: Path) -> Path | None:
    """3×3 grid: rows = pinned CPU count (12,16,20); cols = CPU | 1×GPU | 2×GPU pipelined."""

    aff_levels = [12, 16, 20]
    gpu_paths = {n: RESULTS_DIR / f"seisbench_matrix_lean_aff{n}.jsonl" for n in aff_levels}
    cpu_paths = {n: RESULTS_DIR / f"seisbench_matrix_lean_cpu_aff{n}.jsonl" for n in aff_levels}

    for p in list(gpu_paths.values()) + list(cpu_paths.values()):
        if not p.is_file():
            print("Missing required result file:", p, file=sys.stderr)
            return None

    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(13.5, 10.5),
        constrained_layout=False,
        sharey="row",
    )
    col_titles = [
        "CPU-only (device = cpu)",
        r"1$\times$ GPU (cuda:0)",
        r"2$\times$ GPU (pipelined lean, annotate dual baseline)",
    ]
    row_labels = [f"{n} CPUs pinned" for n in aff_levels]

    handles, labels = [], []
    for ri, n_cpu in enumerate(aff_levels):
        cpu_rows = load_jsonl(cpu_paths[n_cpu])
        gpu_rows = load_jsonl(gpu_paths[n_cpu])
        panels: List[Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]] = [
            ("cpu", cpu_rows, _curve_specs_for_panel(model, "cpu")),
            ("single_gpu", gpu_rows, _curve_specs_for_panel(model, "single_gpu")),
            ("dual_gpu", gpu_rows, _curve_specs_for_panel(model, "dual_gpu")),
        ]
        for ci, (_pid, rows_panel, specs) in enumerate(panels):
            ax = axes[ri][ci]
            _plot_iqr_bands_on_ax(ax, rows_panel, model=model, specs=specs, show_ylabel=(ci == 0))
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=11)
            if ci == 0:
                ax.annotate(
                    row_labels[ri],
                    xy=(-0.28, 0.5),
                    xycoords="axes fraction",
                    fontsize=11,
                    rotation=90,
                    va="center",
                    ha="right",
                )
            if ri == 0 and ci == 0:
                handles, labels = ax.get_legend_handles_labels()

    fig.suptitle(
        f"Wall time vs station count with IQR bands — {model}\n"
        r"Band = middle 50\% of trial wall times (Q1--Q3) at each $N_{\mathrm{st}}$; "
        r"line = median. Lean single-GPU/CPU uses batch\_size = 256; dual lean pipelined uses batch\_size = 256.",
        fontsize=11,
        y=1.01,
    )
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.995),
            frameon=True,
            fontsize=8,
        )
    plt.tight_layout(rect=[0.03, 0.0, 1.0, 0.92])
    slug = model.replace(" ", "_").replace("/", "-")
    p = out_dir / f"tq_style_walltime_bands_affinity_grid_{slug}.pdf"
    fig.savefig(p)
    fig.savefig(p.with_suffix(".png"), dpi=200)
    plt.close(fig)
    sns.reset_defaults()
    return p


def figure_grouped_speed_score(rows: List[Dict[str, Any]], out_dir: Path) -> Path:
    """TurboQuant Fig.~4/5 analogue: grouped bars (relative latency + ``throughput score'')."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    n_st = 580
    n_models = len(MODEL_ORDER)
    w_bar = 0.34
    x = np.arange(n_models)

    rel_lat: List[float] = []
    tput: List[float] = []
    for m in MODEL_ORDER:
        wb = _wall_median_all_datasets(
            rows, model=m, runner="baseline_annotate", dtype="fp32", n_stations=n_st
        )
        wl = _wall_median_all_datasets(
            rows, model=m, runner="lean_pytorch", dtype="bf16", n_stations=n_st
        )
        rel_lat.append(wl / wb if wb > 0 else float("nan"))
        tput.append(wb / wl if wl > 0 else float("nan"))

    ax1.bar(
        x - w_bar / 2,
        [1.0] * n_models,
        w_bar,
        label="annotate() (norm.)",
        color="#94a3b8",
        edgecolor="black",
        linewidth=0.55,
    )
    ax1.bar(
        x + w_bar / 2,
        rel_lat,
        w_bar,
        label="Lean BF16 / annotate()",
        color="#1d4ed8",
        edgecolor="black",
        linewidth=0.55,
    )
    ax1.axhline(1.0, color="black", lw=0.8, linestyle="--")
    ax1.set_xticks(x)
    ax1.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    ax1.set_ylabel("Relative median wall time")
    ax1.set_title(rf"Latency ratio @ $N_{{\mathrm{{st}}}}={n_st}$ (lower is better for lean)")
    ax1.legend(fontsize=8)

    ax2.bar(
        x,
        tput,
        color=[MODEL_COLORS[m] for m in MODEL_ORDER],
        edgecolor="black",
        linewidth=0.55,
    )
    ax2.axhline(1.0, color="black", lw=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(MODEL_ORDER, rotation=12, ha="right")
    ax2.set_ylabel(r"Speed factor (annotate / lean BF16)")
    ax2.set_title(rf"Effective speedup @ $N_{{\mathrm{{st}}}}={n_st}$")
    fig.suptitle("Cross-model comparison (GPU, pooled datasets, 16-CPU host affinity)", y=1.02)
    fig.tight_layout()
    p = out_dir / "tq_style_grouped_speed_score.pdf"
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    sns.reset_defaults()
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description="TurboQuant-style SeisBench figures.")
    ap.add_argument(
        "--model",
        default="PhaseNet",
        help="Model for wall-time IQR grid (PhaseNet, PhaseNetLight, EQTransformer, EQT-NC).",
    )
    ap.add_argument(
        "--skip-grid",
        action="store_true",
        help="Do not build the 3x3 CPU/1-GPU/2-GPU IQR grid (requires six affinity JSONL files).",
    )
    ap.add_argument(
        "--only-grid",
        action="store_true",
        help="Only build the affinity IQR grid (skip the five standalone figures).",
    )
    args = ap.parse_args()

    if args.only_grid:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        g = figure_walltime_bands_affinity_device_grid(args.model, OUT_DIR)
        if g is None:
            return 1
        print(g)
        return 0

    jsonl = RESULTS_DIR / "seisbench_matrix_lean_aff16.jsonl"
    if not jsonl.is_file():
        print("Missing", jsonl, file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(jsonl)
    if not rows:
        print("No rows in", jsonl, file=sys.stderr)
        return 1

    paths = []
    paths.extend(figure_error_distributions(rows, OUT_DIR))
    paths.append(figure_quality_vs_scale(rows, OUT_DIR))
    paths.append(figure_rate_latency_tradeoff(rows, OUT_DIR))
    paths.append(figure_walltime_bands(rows, OUT_DIR))
    paths.append(figure_grouped_speed_score(rows, OUT_DIR))
    if not args.skip_grid:
        g = figure_walltime_bands_affinity_device_grid(args.model, OUT_DIR)
        if g is not None:
            paths.append(g)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
