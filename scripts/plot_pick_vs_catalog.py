#!/usr/bin/env python3
"""Pick residual histograms vs CATALOG P arrival (true ground truth).

Shows distributions of (onset_p - p_catalog_in_window) for:
  - baseline_annotate (SeisBench annotate())
  - lean BF16
  - lean FP16
  - lean BF16 + torch.compile

All relative to the **catalog P position** in the window, which is the true
ground truth from the seismic catalog.

Generates faceted figures by N_st (64, 256, 580) for each combination of:
  - Device: cpu, cuda:0 (1 GPU), cuda:0+cuda:1 (2 GPUs)
  - Model: PhaseNet, PhaseNetLight, EQTransformer, EQT-NC

Usage:
  python plot_pick_vs_catalog.py --jsonl-dir RAPID/results/ --out-dir RAPID/figures/pick_vs_catalog/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Station-count tiers
N_STATION_TIERS: Tuple[int, ...] = (64, 256, 580)

# Colors for each distribution
COLORS = {
    "annotate": "#7c3aed",      # purple
    "bf16": "#2563eb",          # blue
    "fp16": "#ea580c",          # orange
    "bf16_compile": "#16a34a",  # green
}

LABELS = {
    "annotate": "annotate()",
    "bf16": "lean BF16",
    "fp16": "lean FP16",
    "bf16_compile": "lean BF16 + compile",
}

# Legend row order (top → bottom): catalog line, then model paths
LEGEND_LABEL_ORDER: Tuple[str, ...] = (
    "Catalog P",
    LABELS["annotate"],
    LABELS["fp16"],
    LABELS["bf16"],
    LABELS["bf16_compile"],
)

# Device configurations
DEVICE_CONFIGS = {
    "cpu": {"device": "cpu", "label": "CPU only", "suffix": "cpu"},
    "1gpu": {"device": "cuda:0", "label": "1 GPU", "suffix": "1gpu"},
    "2gpu": {"device": "cuda:0+cuda:1", "label": "2 GPUs", "suffix": "2gpu"},
}

# Models
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]


def load_all_jsonl(jsonl_dir: Path) -> List[Dict]:
    """Load all JSONL files from directory."""
    rows: List[Dict] = []
    for p in sorted(jsonl_dir.glob("seisbench_matrix_lean*.jsonl")):
        if p.suffix == ".bak":
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("kind") == "env":
                    continue
                rows.append(r)
    return rows


def collect_deltas_vs_catalog(
    rows: List[Dict],
    *,
    device: str,
    model: str,
    n_stations: int,
    fs_hz: float = 100.0,
) -> Tuple[Dict[str, List[float]], int, int]:
    """Collect onset_delta_p_vs_catalog for each distribution type.
    Returns (results, cpu_count, gpu_count).
    """
    ms_per_sample = 1000.0 / fs_hz
    
    results = {
        "annotate": [],
        "bf16": [],
        "fp16": [],
        "bf16_compile": [],
    }
    
    cpu_counts = []
    gpu_count = 0
    if "cuda:0+cuda:1" in device:
        gpu_count = 2
    elif "cuda:" in device:
        gpu_count = 1
    
    for r in rows:
        if r.get("benchmark_status") == "skipped_incompatible":
            continue
        if r.get("model_label") != model:
            continue
        if r.get("device") != device:
            continue
        if int(r.get("n_stations") or 0) != n_stations:
            continue
        
        # Track CPU count (use process_n_cpus or n_cpus_pinned)
        nc = r.get("process_n_cpus") or r.get("n_cpus_pinned")
        if nc is not None:
            cpu_counts.append(int(nc))

        pq = r.get("pick_quality") or {}
        delta = pq.get("onset_delta_p_vs_catalog")
        if delta is None:
            continue
        
        delta_ms = float(delta) * ms_per_sample
        
        runner = r.get("runner", "")
        dtype = r.get("dtype", "")
        compile_flag = bool((r.get("backend_extra") or {}).get("compile"))
        
        # Classify the row
        if runner in ("baseline_annotate", "baseline_annotate_dual"):
            results["annotate"].append(delta_ms)
        elif runner in ("lean_pytorch", "lean_pytorch_dual_pipelined"):
            if dtype == "bf16":
                if compile_flag:
                    results["bf16_compile"].append(delta_ms)
                else:
                    results["bf16"].append(delta_ms)
            elif dtype == "fp16":
                if not compile_flag:  # Skip fp16+compile for simplicity
                    results["fp16"].append(delta_ms)
    
    # Use the most frequent CPU count found
    final_cpu = 0
    if cpu_counts:
        from collections import Counter
        final_cpu = Counter(cpu_counts).most_common(1)[0][0]
    
    return results, final_cpu, gpu_count


def _auto_symmetric_xlim(
    all_data: List[float], fs_hz: float = 100.0, percentile: float = 95.0
) -> Tuple[float, float]:
    """Compute symmetric x-limits from data using percentile to ignore outliers."""
    ms_per_sample = 1000.0 / fs_hz
    if not all_data:
        return (-500.0, 500.0)
    
    # Use percentile to ignore extreme outliers
    abs_vals = [abs(x) for x in all_data]
    mx = float(np.percentile(abs_vals, percentile))
    
    # Ensure at least ±500ms range, cap at ±3000ms for readability
    mx = max(mx, 500.0)
    mx = min(mx, 3000.0)
    
    pad = max(mx * 0.15, ms_per_sample * 2, 100.0)
    half = mx + pad
    return (-half, half)


def plot_model_combined(
    rows: List[Dict],
    *,
    model: str,
    out_dir: Path,
    fs_hz: float = 100.0,
    n_bins: int = 50,
) -> Optional[Path]:
    """Create a 3x3 faceted figure for one model (CPU, 1GPU, 2GPU rows)."""
    # Collect all data first to determine shared x-limits
    all_deltas: List[float] = []
    data_grid: Dict[str, Dict[int, Dict[str, List[float]]]] = {}
    hw_info: Dict[str, Dict[int, Tuple[int, int]]] = {}
    
    device_keys = ["cpu", "1gpu", "2gpu"]
    
    for dk in device_keys:
        device = DEVICE_CONFIGS[dk]["device"]
        data_grid[dk] = {}
        hw_info[dk] = {}
        for nst in N_STATION_TIERS:
            data, n_cpu, n_gpu = collect_deltas_vs_catalog(
                rows, device=device, model=model, n_stations=nst, fs_hz=fs_hz
            )
            data_grid[dk][nst] = data
            hw_info[dk][nst] = (n_cpu, n_gpu)
            for vals in data.values():
                all_deltas.extend(vals)
    
    if not all_deltas:
        return None
    
    # Compute shared x-limits across the entire 3x3 grid
    xlim = _auto_symmetric_xlim(all_deltas, fs_hz)
    
    fig, axes_grid = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    plot_order = ["bf16_compile", "fp16", "bf16", "annotate"]
    
    for row_idx, dk in enumerate(device_keys):
        for col_idx, nst in enumerate(N_STATION_TIERS):
            ax = axes_grid[row_idx, col_idx]
            data = data_grid[dk][nst]
            n_cpu, n_gpu = hw_info[dk][nst]
            
            # Plot each distribution
            for key in plot_order:
                vals = data[key]
                if not vals:
                    continue
                vals_in = [v for v in vals if xlim[0] <= v <= xlim[1]]
                if not vals_in:
                    continue
                
                ax.hist(
                    vals_in,
                    bins=bins,
                    alpha=0.35,
                    label=LABELS[key],
                    color=COLORS[key],
                    edgecolor=COLORS[key],
                    linewidth=1.2,
                    histtype="stepfilled",
                )
            
            # Reference line at 0
            ax.axvline(0, color="black", lw=2.5, ls="-", label="Catalog P", zorder=6)
            
            ax.set_xlim(xlim)
            if row_idx == 2:
                ax.set_xlabel(r"$\Delta t$ vs catalog (ms)", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel("Count", fontsize=10)
            
            # Panel label in top right: N_st on first line, CPUs/GPUs on second line (same math style)
            hw_parts = [rf"{n_cpu} \mathrm{{CPUs}}"]
            if n_gpu > 0:
                gpu_suffix = "s" if n_gpu > 1 else ""
                hw_parts.append(rf"{n_gpu} \mathrm{{GPU{gpu_suffix}}}")
            hw_str = ", ".join(hw_parts)
            
            label_text = rf"$N_{{\mathrm{{st}}}} = {nst}$" + f"\n${hw_str}$"
            
            ax.text(
                0.98,
                0.98,
                label_text,
                transform=ax.transAxes,
                fontsize=9,
                fontweight="bold",
                ha="right",
                va="top",
                linespacing=1.2
            )
            
            ax.grid(True, axis="y", ls=":", alpha=0.5)
            ax.grid(True, axis="x", ls=":", alpha=0.3)
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ord_handles: List[Any] = []
            ord_labels: List[str] = []
            for lab in LEGEND_LABEL_ORDER:
                h = by_label.get(lab)
                if h is not None:
                    ord_handles.append(h)
                    ord_labels.append(lab)
            
            if ord_handles:
                ax.legend(
                    ord_handles,
                    ord_labels,
                    fontsize=7,
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.86),
                    borderaxespad=0,
                    labelspacing=0.3,
                    handlelength=1.6,
                    handletextpad=0.5,
                    borderpad=0.3,
                    framealpha=0.92,
                )

    fig.tight_layout()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = model.replace("-", "").replace(" ", "_")
    out_path = out_dir / f"pick_vs_catalog_{model_slug}_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Pick residual vs catalog histograms")
    p.add_argument(
        "--jsonl-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
        help="Directory containing JSONL files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "figures" / "pick_vs_catalog",
        help="Output directory for figures",
    )
    p.add_argument("--fs-hz", type=float, default=100.0, help="Sample rate (Hz)")
    p.add_argument("--bins", type=int, default=50, help="Number of histogram bins")
    p.add_argument("--model", type=str, default=None, help="Single model to plot (default: all)")
    args = p.parse_args()
    
    if not args.jsonl_dir.exists():
        print(f"Error: {args.jsonl_dir} not found", file=sys.stderr)
        return 1
    
    print(f"Loading JSONL files from {args.jsonl_dir}...")
    rows = load_all_jsonl(args.jsonl_dir)
    print(f"Loaded {len(rows)} rows")
    
    models = [args.model] if args.model else MODELS
    
    generated = []
    for model in models:
        print(f"Generating combined plot for {model}...", end=" ")
        out_path = plot_model_combined(
            rows,
            model=model,
            out_dir=args.out_dir,
            fs_hz=args.fs_hz,
            n_bins=args.bins,
        )
        if out_path:
            print(f"-> {out_path.name}")
            generated.append(out_path)
        else:
            print("(no data)")
    
    print(f"\nGenerated {len(generated)} combined figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
