"""Generate figures for the affinity-controlled benchmark results."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex, to_rgb

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11

MODELS: List[str] = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
MODEL_ABBREV: Dict[str, str] = {
    "PhaseNet": "PN",
    "PhaseNetLight": "PNL",
    "EQTransformer": "EQT",
    "EQT-NC": "EQT-NC",
}
MODEL_COLORS = {
    "PhaseNet": "#1f77b4",
    "PhaseNetLight": "#ff7f0e",
    "EQTransformer": "#2ca02c",
    "EQT-NC": "#d62728",
}

N_STATIONS = [64, 256, 580]

LS_ANNOT = "-"
LS_BF16 = ":"
LS_FP16 = "--"
MK_ANNOT = "s"
MK_BF16 = "o"
MK_FP16 = "^"
MS = 6
FIG_DPI = 150


def _mix_with_white(hex_color: str, white_frac: float) -> str:
    """Blend base color toward white (0 = pure base, 1 = white)."""
    r, g, b = to_rgb(hex_color)
    r = r * (1 - white_frac) + white_frac
    g = g * (1 - white_frac) + white_frac
    b = b * (1 - white_frac) + white_frac
    return to_hex((r, g, b))


def trace_colors_for_model(model: str) -> Tuple[str, str, str]:
    """annotate (darkest), BF16 lean, FP16 lean — same hue, lighter shades."""
    base = MODEL_COLORS[model]
    return base, _mix_with_white(base, 0.25), _mix_with_white(base, 0.45)


def _legend_model_and_fp_types(ax: plt.Axes) -> None:
    """Single legend: model colors (line swatches) + FP path markers/linestyles."""
    handles: List[mlines.Line2D] = [
        mlines.Line2D(
            [],
            [],
            color=MODEL_COLORS[m],
            linestyle="-",
            linewidth=3,
            label=MODEL_ABBREV[m],
        )
        for m in MODELS
    ]
    handles.extend(
        [
            mlines.Line2D(
                [],
                [],
                color="0.35",
                linestyle=LS_ANNOT,
                marker=MK_ANNOT,
                markersize=MS,
                label="annotate",
            ),
            mlines.Line2D(
                [],
                [],
                color="0.35",
                linestyle=LS_BF16,
                marker=MK_BF16,
                markersize=MS,
                label="BF16",
            ),
            mlines.Line2D(
                [],
                [],
                color="0.35",
                linestyle=LS_FP16,
                marker=MK_FP16,
                markersize=MS,
                label="FP16",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        fontsize=8,
        title="Model & FP Types",
        loc="best",
        framealpha=0.95,
    )


def load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def median(xs: List[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def aggregate_by(
    rows: List[Dict],
    key_fields: Tuple[str, ...],
    value_field: str = "wall_time_s",
    **filters,
) -> Dict[Tuple, List[float]]:
    groups: Dict[Tuple, List[float]] = defaultdict(list)
    for r in rows:
        if any(r.get(k) != v for k, v in filters.items()):
            continue
        val = r.get(value_field)
        if val is None:
            continue
        key = tuple(r.get(k) for k in key_fields)
        groups[key].append(val)
    return groups


def _lean_nc_pred(r: Dict) -> bool:
    return r.get("runner") == "lean_pytorch" and not (r.get("backend_extra") or {}).get(
        "compile"
    )


def panel_note_gpu(aff: int, n_gpu: int) -> str:
    return f"{aff} CPUs, {n_gpu} GPU" + ("s" if n_gpu > 1 else "")


def _gpu_line_row_groups(rows: List[Dict], n_gpu: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    if n_gpu == 1:
        base = [
            r
            for r in rows
            if r.get("runner") == "baseline_annotate"
            and r.get("device") == "cuda:0"
            and r.get("batch_size") == -1
        ]
        lean_bf = [
            r
            for r in rows
            if _lean_nc_pred(r)
            and r.get("device") == "cuda:0"
            and r.get("dtype") == "bf16"
            and r.get("batch_size") == 256
        ]
        lean_fp = [
            r
            for r in rows
            if _lean_nc_pred(r)
            and r.get("device") == "cuda:0"
            and r.get("dtype") == "fp16"
            and r.get("batch_size") == 256
        ]
    else:
        base = [
            r
            for r in rows
            if r.get("runner") == "baseline_annotate_dual"
            and r.get("device") == "cuda:0+cuda:1"
            and r.get("batch_size") == -1
        ]
        lean_bf = [
            r
            for r in rows
            if r.get("runner") == "lean_pytorch_dual_pipelined"
            and r.get("dtype") == "bf16"
            and r.get("batch_size") == 256
        ]
        lean_fp = [
            r
            for r in rows
            if r.get("runner") == "lean_pytorch_dual_pipelined"
            and r.get("dtype") == "fp16"
            and r.get("batch_size") == 256
        ]
    return base, lean_bf, lean_fp


def _cpu_line_row_groups(rows: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """CPU annotate vs lean (same batch_size=256 filter as GPU line plots)."""
    base = [
        r
        for r in rows
        if r.get("runner") == "baseline_annotate"
        and r.get("device") == "cpu"
        and r.get("batch_size") == -1
    ]
    lean_bf = [
        r
        for r in rows
        if _lean_nc_pred(r)
        and r.get("device") == "cpu"
        and r.get("dtype") == "bf16"
        and r.get("batch_size") == 256
    ]
    lean_fp = [
        r
        for r in rows
        if _lean_nc_pred(r)
        and r.get("device") == "cpu"
        and r.get("dtype") == "fp16"
        and r.get("batch_size") == 256
    ]
    return base, lean_bf, lean_fp


def _plot_annotate_lean_traces(
    ax: plt.Axes,
    *,
    line_groups: Tuple[List[Dict], List[Dict], List[Dict]],
    idx: int,
    panel_title: str,
) -> None:
    base, lean_bf, lean_fp = line_groups
    gb = aggregate_by(base, ("model_label", "n_stations"))
    g_bf = aggregate_by(lean_bf, ("model_label", "n_stations"))
    g_fp = aggregate_by(lean_fp, ("model_label", "n_stations"))

    for model in MODELS:
        c_ann, c_bf, c_fp = trace_colors_for_model(model)
        xs = [n for n in N_STATIONS if (model, n) in gb]
        ys = [median(gb[(model, n)]) for n in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                linestyle=LS_ANNOT,
                marker=MK_ANNOT,
                color=c_ann,
                label=None,
                markersize=6,
            )
        xs = [n for n in N_STATIONS if (model, n) in g_bf]
        ys = [median(g_bf[(model, n)]) for n in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                linestyle=LS_BF16,
                marker=MK_BF16,
                color=c_bf,
                label=None,
                markersize=6,
            )
        if model in ("EQTransformer", "EQT-NC"):
            continue
        xs = [n for n in N_STATIONS if (model, n) in g_fp]
        ys = [median(g_fp[(model, n)]) for n in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                linestyle=LS_FP16,
                marker=MK_FP16,
                color=c_fp,
                label=None,
                markersize=6,
            )

    ax.set_xlabel("Station count")
    ax.text(
        0.5,
        1.02,
        panel_title,
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    ax.grid(True, ls=":", alpha=0.5)
    if idx == 0:
        ax.set_ylabel("Median wall time (s)")
        _legend_model_and_fp_types(ax)


def plot_cpu_annotate_lean_lines_by_affinity(
    cpu_files: Dict[int, Path], out_dir: Path
) -> Path:
    """Three-panel grid matching GPU layout: CPU-only annotate + lean BF16/FP16."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for idx, (aff, fpath) in enumerate(sorted(cpu_files.items())):
        rows = load_jsonl(fpath)
        _plot_annotate_lean_traces(
            axes[idx],
            line_groups=_cpu_line_row_groups(rows),
            idx=idx,
            panel_title=f"{aff} CPUs",
        )
    fig.tight_layout()
    out = out_dir / "cpu_0gpu_annotate_lean_by_affinity.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gpu_annotate_lean_lines_by_affinity(
    gpu_files: Dict[int, Path],
    out_dir: Path,
    n_gpu: int,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for idx, (aff, fpath) in enumerate(sorted(gpu_files.items())):
        rows = load_jsonl(fpath)
        _plot_annotate_lean_traces(
            axes[idx],
            line_groups=_gpu_line_row_groups(rows, n_gpu),
            idx=idx,
            panel_title=panel_note_gpu(aff, n_gpu),
        )
    fig.tight_layout()
    out = out_dir / f"gpu_{n_gpu}gpu_annotate_lean_by_affinity.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _draw_scaling_log_on_ax(
    ax: plt.Axes,
    rows: List[Dict],
    n_gpu: int,
    panel_title: str,
    show_legend: bool = False,
) -> None:
    """Helper to draw the scaling log traces on a given axis."""
    if n_gpu == 0:
        base, lean_bf, lean_fp = _cpu_line_row_groups(rows)
    else:
        base, lean_bf, lean_fp = _gpu_line_row_groups(rows, n_gpu)

    gb = aggregate_by(base, ("model_label", "n_stations"))
    g_bf = aggregate_by(lean_bf, ("model_label", "n_stations"))
    g_fp = aggregate_by(lean_fp, ("model_label", "n_stations"))

    for model in MODELS:
        c_ann, c_bf, c_fp = trace_colors_for_model(model)
        xs = [n for n in N_STATIONS if (model, n) in gb]
        ys = [median(gb[(model, n)]) for n in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                linestyle=LS_ANNOT,
                marker=MK_ANNOT,
                color=c_ann,
                label=None,
                markersize=7,
            )
        xs = [n for n in N_STATIONS if (model, n) in g_bf]
        ys = [median(g_bf[(model, n)]) for n in xs]
        if xs:
            ax.plot(
                xs,
                ys,
                linestyle=LS_BF16,
                marker=MK_BF16,
                color=c_bf,
                label=None,
                markersize=7,
            )
        if model not in ("EQTransformer", "EQT-NC"):
            xs = [n for n in N_STATIONS if (model, n) in g_fp]
            ys = [median(g_fp[(model, n)]) for n in xs]
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linestyle=LS_FP16,
                    marker=MK_FP16,
                    color=c_fp,
                    label=None,
                    markersize=7,
                )

    ax.set_yscale("log")
    ax.set_xlabel("Station count")
    ax.set_title(panel_title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_xticks(N_STATIONS)
    if show_legend:
        _legend_model_and_fp_types(ax)


def plot_scaling_log_combined(
    cpu_files: Dict[int, Path], gpu_files: Dict[int, Path], out_dir: Path, aff: int
) -> Path:
    """1x3 grid: CPU-only, 1 GPU, 2 GPUs for a fixed CPU affinity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

    # Left: CPU only (0 GPUs)
    rows_cpu = load_jsonl(cpu_files[aff])
    _draw_scaling_log_on_ax(axes[0], rows_cpu, 0, f"{aff} CPUs, 0 GPUs")
    axes[0].set_ylabel("Median wall time (s, log)")

    # Middle: 1 GPU
    rows_gpu = load_jsonl(gpu_files[aff])
    _draw_scaling_log_on_ax(axes[1], rows_gpu, 1, f"{aff} CPUs, 1 GPU")

    # Right: 2 GPUs
    _draw_scaling_log_on_ax(axes[2], rows_gpu, 2, f"{aff} CPUs, 2 GPUs", show_legend=True)

    fig.tight_layout()
    out = out_dir / f"scaling_log_combined_{aff}cpus.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scaling_log_cpu_affinity(cpu_files: Dict[int, Path], out_dir: Path, aff: int) -> Path:
    fpath = cpu_files[aff]
    rows = load_jsonl(fpath)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_scaling_log_on_ax(ax, rows, 0, f"{aff} CPUs, 0 GPUs", show_legend=True)
    ax.set_ylabel("Median wall time (s, log)")
    fig.tight_layout()
    out = out_dir / f"scaling_log_{aff}cpus_0gpus.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scaling_log_gpu_affinity(
    gpu_files: Dict[int, Path], out_dir: Path, aff: int, n_gpu: int
) -> Path:
    rows = load_jsonl(gpu_files[aff])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _draw_scaling_log_on_ax(ax, rows, n_gpu, panel_note_gpu(aff, n_gpu), show_legend=True)
    ax.set_ylabel("Median wall time (s, log)")
    fig.tight_layout()
    out = out_dir / f"scaling_log_{aff}cpus_{n_gpu}gpus.png"
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    results_dir = Path(__file__).resolve().parent.parent / "results"
    out_dir = Path(__file__).resolve().parent.parent / "figures" / "affinity"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpu_files: Dict[int, Path] = {
        12: results_dir / "seisbench_matrix_lean_cpu_aff12.jsonl",
        16: results_dir / "seisbench_matrix_lean_cpu_aff16.jsonl",
        20: results_dir / "seisbench_matrix_lean_cpu_aff20.jsonl",
    }
    gpu_files: Dict[int, Path] = {
        12: results_dir / "seisbench_matrix_lean_aff12.jsonl",
        16: results_dir / "seisbench_matrix_lean_aff16.jsonl",
        20: results_dir / "seisbench_matrix_lean_aff20.jsonl",
    }

    outputs: List[Path] = []

    print("Generating CPU annotate/lean line grid...")
    outputs.append(plot_cpu_annotate_lean_lines_by_affinity(cpu_files, out_dir))
    print("Generating GPU annotate/lean line grids...")
    outputs.append(plot_gpu_annotate_lean_lines_by_affinity(gpu_files, out_dir, n_gpu=1))
    outputs.append(plot_gpu_annotate_lean_lines_by_affinity(gpu_files, out_dir, n_gpu=2))
    print("Generating scaling log (CPU, per affinity)...")
    for aff in (12, 16, 20):
        outputs.append(plot_scaling_log_cpu_affinity(cpu_files, out_dir, aff))
    print("Generating scaling log (GPU, per affinity)...")
    for aff in (12, 16, 20):
        outputs.append(plot_scaling_log_gpu_affinity(gpu_files, out_dir, aff, n_gpu=1))
        outputs.append(plot_scaling_log_gpu_affinity(gpu_files, out_dir, aff, n_gpu=2))

    print("Generating combined scaling log (CPU + GPU grid)...")
    for aff in (12, 16, 20):
        outputs.append(plot_scaling_log_combined(cpu_files, gpu_files, out_dir, aff))

    print(f"\nGenerated {len(outputs)} PNG figures:")
    for p in outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
