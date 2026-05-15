"""Overlapping pick residual histograms vs annotate() (geophysics-style).

For each matched trial, Δt_ms = (t_pick,lean − t_pick,annotate) with picks taken
as simple-threshold P onsets in samples (``pick_quality.onset_p``), converted
using ``--fs-hz`` (STEAD windows default to 100 Hz in SeisBench).

Onsets are **integer samples**, so Δt is a multiple of (1000/fs) ms (e.g. **10 ms**
at 100 Hz). **Fair pairing:** each residual pairs **one lean row with the annotate
row for the same SeisBench trial** — same ``dataset_label``, same trace identity
(``sb_trace_row`` / ``sb_trace_rows``), same ``n_stations``, ``repeat``, ``device``,
and the same window geometry (``n_samples``, ``in_samples``, ``overlap_samples``)
and catalog P index in-window (``p_catalog_in_window``). That is the **same raw
segment and catalog anchor** the matrix ran for both paths (see
``rapid/seisbench_matrix.py``). Pooling many trials in one histogram **does not**
mix unrelated traces; it averages Δt across many correctly matched pairs.

**Why lean can still differ from annotate:** even on an identical waveform segment,
``annotate()`` vs batched lean can produce **different P probability traces**
(filters, STFT, merge/cut). The histogram is therefore “threshold-onset shift”
between those two inference paths, not a claim about sub-sample clock accuracy.
When ``p_catalog_in_window`` matches, ``onset_delta_p_vs_catalog(lean) −
onset_delta_p_vs_catalog(annotate)`` equals ``onset_p(lean) − onset_p(annotate)``
(sample identity from algebra).

**Default** pools **all** matrix rows for the model (all ``n_stations`` tiers and
datasets). Use ``--per-n-stations`` for one column per ``N_st``.

**X-limits:** symmetric limits are **computed from the pooled data** unless
``--manual-xlim``. A **±0.5 ms** window usually shows **no bars** because |Δt| is
often ≫ 1 ms here.

Reads one matrix JSONL and plots up to three translucent distributions: BF16 lean,
FP16 lean, and BF16 + torch.compile (PhaseNet / PhaseNetLight only, when compile
trials exist).

Usage:
  python RAPID/scripts/plot_pick_residual_histogram.py \\
    --jsonl RAPID/results/seisbench_matrix_lean_aff16.jsonl \\
    --device cuda:0 --model PhaseNet \\
    --out RAPID/figures/affinity/pick_residual_phasenet_cuda0.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator

# Station-count tiers in the SeisBench matrix JSONL
N_STATION_TIERS: Tuple[int, ...] = (64, 256, 580)

LEAN_BATCH = 256
DEFAULT_FS_HZ = 100.0
DEFAULT_XMIN_MS = -0.5
DEFAULT_XMAX_MS = 0.5
DEFAULT_BINS = 80


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, encoding="utf-8") as f:
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


def _trace_key_part(r: Dict) -> Tuple[int, ...]:
    """Trace identity aligned with ``rapid.seisbench_matrix._trace_slot`` (hashable)."""
    trs = r.get("sb_trace_rows")
    if isinstance(trs, (list, tuple)) and len(trs) > 0:
        return tuple(sorted(int(x) for x in trs))
    if r.get("sb_trace_row") is not None:
        return (int(r["sb_trace_row"]),)
    return tuple()


def trial_key(r: Dict) -> Tuple:
    """Keys one matrix trial for annotate↔lean pairing (same trace + window + repeat)."""
    pc = r.get("p_catalog_in_window")
    pc_key = int(pc) if pc is not None else None
    return (
        str(r.get("model_label") or ""),
        str(r.get("dataset_label") or ""),
        _trace_key_part(r),
        int(r.get("n_stations") or 0),
        int(r.get("repeat") or 0),
        str(r.get("device") or ""),
        int(r.get("n_samples") or -1),
        int(r.get("in_samples") or -1),
        int(r.get("overlap_samples") or 0),
        pc_key,
    )


def _is_annotate_row(r: Dict, device: str) -> bool:
    if r.get("batch_size") != -1:
        return False
    if device == "cuda:0+cuda:1":
        return r.get("runner") == "baseline_annotate_dual" and r.get("device") == device
    return r.get("runner") == "baseline_annotate" and r.get("device") == device


def _is_lean_row(r: Dict, device: str, dtype: str, compiled: bool) -> bool:
    if r.get("batch_size") != LEAN_BATCH:
        return False
    if r.get("device") != device:
        return False
    if r.get("dtype") != dtype:
        return False
    comp = bool((r.get("backend_extra") or {}).get("compile"))
    if comp != compiled:
        return False
    if device == "cuda:0+cuda:1":
        return r.get("runner") == "lean_pytorch_dual_pipelined"
    return r.get("runner") == "lean_pytorch"


def annotate_onset_index(rows: List[Dict], device: str) -> Dict[Tuple, int]:
    """Map ``trial_key`` → annotate ``onset_p`` (samples). Last row wins on duplicate keys."""
    out: Dict[Tuple, int] = {}
    dup = 0
    for r in rows:
        if r.get("benchmark_status") == "skipped_incompatible":
            continue
        if not _is_annotate_row(r, device):
            continue
        pq = r.get("pick_quality") or {}
        if pq.get("onset_p") is None:
            continue
        k = trial_key(r)
        if k in out:
            dup += 1
        out[k] = int(pq["onset_p"])
    if dup:
        print(
            f"Note: {dup} duplicate annotate trial_keys were merged (last wins); "
            "check JSONL for conflicting rows.",
            file=sys.stderr,
        )
    return out


def collect_delta_ms(
    rows: List[Dict],
    *,
    device: str,
    ann: Dict[Tuple, int],
    dtype: str,
    compiled: bool,
    model: str | None,
    fs_hz: float,
    n_stations: Optional[int] = None,
) -> List[float]:
    ms_per_sample = 1000.0 / fs_hz
    deltas: List[float] = []
    for r in rows:
        if r.get("benchmark_status") == "skipped_incompatible":
            continue
        if not _is_lean_row(r, device, dtype, compiled):
            continue
        if model and r.get("model_label") != model:
            continue
        if n_stations is not None and int(r.get("n_stations") or 0) != n_stations:
            continue
        k = trial_key(r)
        if k not in ann:
            continue
        pq = r.get("pick_quality") or {}
        if pq.get("onset_p") is None:
            continue
        d_samp = int(pq["onset_p"]) - ann[k]
        deltas.append(float(d_samp * ms_per_sample))
    return deltas


def _apply_xaxis_format(ax: plt.Axes, xlim: Tuple[float, float]) -> None:
    span = float(xlim[1] - xlim[0])
    if span > 0 and span <= 2.0:
        major = max(0.05, span / 10.0)
        minor = major / 5.0
        ax.xaxis.set_major_locator(MultipleLocator(major))
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
    ax.tick_params(axis="x", which="major", labelsize=9)
    ax.tick_params(axis="x", which="minor", length=3, labelsize=7)


def _draw_picks_on_ax(
    ax: plt.Axes,
    series: List[Tuple[str, List[float], str]],
    *,
    xlim: Tuple[float, float],
    n_bins: int,
    density: bool,
    panel_title: str = "",
) -> None:
    ax.set_axisbelow(True)
    bins = np.linspace(xlim[0], xlim[1], n_bins + 1)
    for label, data, color in series:
        if not data:
            continue
        data_sel = [x for x in data if xlim[0] <= x <= xlim[1]]
        if not data_sel:
            continue
        leg = f"{label} (n={len(data_sel)}"
        if len(data_sel) != len(data):
            leg += f", {len(data) - len(data_sel)} outside x-range"
        leg += ")"
        ax.hist(
            data_sel,
            bins=bins,
            density=density,
            alpha=0.42,
            label=leg,
            color=color,
            edgecolor=color,
            linewidth=0.25,
        )
    ax.axvline(
        0.0, color="0.15", lw=1.4, ls="-", zorder=5, label="annotate() (reference index)"
    )
    ax.set_xlim(xlim)
    _apply_xaxis_format(ax, xlim)
    if panel_title:
        ax.set_title(panel_title, fontsize=10)
    ax.grid(True, axis="y", ls=":", alpha=0.55)
    ax.grid(True, axis="x", which="major", ls=":", alpha=0.35)
    ax.grid(True, axis="x", which="minor", ls=":", alpha=0.18)
    ax.legend(fontsize=7, loc="upper right")


def _auto_symmetric_xlim(
    pooled: List[float],
    *,
    fs_hz: float,
) -> Tuple[Tuple[float, float], str]:
    """Return (xlim, note_suffix) for symmetric limits about 0."""
    ms_per_sample = 1000.0 / fs_hz
    if not pooled:
        half = max(0.5 * ms_per_sample, abs(DEFAULT_XMAX_MS))
        return (
            (-float(half), float(half)),
            f" (x ∈ [{-half:.1f}, {half:.1f}] ms, auto)",
        )
    mx = max(abs(x) for x in pooled)
    pad = max(mx * 0.06, 0.5 * ms_per_sample, 1e-9)
    if mx == 0.0:
        half = max(0.5 * ms_per_sample, abs(DEFAULT_XMAX_MS))
    else:
        half = mx + pad
    xlim = (-float(half), float(half))
    return xlim, f" (x ∈ [{xlim[0]:.1f}, {xlim[1]:.1f}] ms, auto)"


def build_series(
    rows: List[Dict],
    ann: Dict[Tuple, int],
    *,
    device: str,
    model: str | None,
    fs_hz: float,
    n_stations: Optional[int] = None,
) -> List[Tuple[str, List[float], str]]:
    series: List[Tuple[str, List[float], str]] = [
        (
            "BF16 lean",
            collect_delta_ms(
                rows,
                device=device,
                ann=ann,
                dtype="bf16",
                compiled=False,
                model=model,
                fs_hz=fs_hz,
                n_stations=n_stations,
            ),
            "#2563eb",
        ),
        (
            "FP16 lean",
            collect_delta_ms(
                rows,
                device=device,
                ann=ann,
                dtype="fp16",
                compiled=False,
                model=model,
                fs_hz=fs_hz,
                n_stations=n_stations,
            ),
            "#ea580c",
        ),
    ]
    compiled_bf16 = collect_delta_ms(
        rows,
        device=device,
        ann=ann,
        dtype="bf16",
        compiled=True,
        model=model,
        fs_hz=fs_hz,
        n_stations=n_stations,
    )
    if compiled_bf16:
        series.append(("BF16 + torch.compile", compiled_bf16, "#16a34a"))
    return series


def plot_faceted_by_n_stations(
    rows: List[Dict],
    ann: Dict[Tuple, int],
    *,
    out: Path,
    device: str,
    model: str | None,
    fs_hz: float,
    n_bins: int,
    density: bool,
    title_suffix: str,
    manual_xlim: Optional[Tuple[float, float]],
) -> None:
    series_by_n: Dict[int, List[Tuple[str, List[float], str]]] = {}
    pooled_all: List[float] = []
    for n in N_STATION_TIERS:
        ser = build_series(rows, ann, device=device, model=model, fs_hz=fs_hz, n_stations=n)
        series_by_n[n] = ser
        for _, d, _ in ser:
            pooled_all.extend(d)

    if not pooled_all:
        raise ValueError("no residuals for faceted plot")

    if manual_xlim is not None:
        xlim = manual_xlim
        xlim_note = f" (x ∈ [{manual_xlim[0]}, {manual_xlim[1]}] ms, manual)"
    else:
        xlim, xlim_note = _auto_symmetric_xlim(pooled_all, fs_hz=fs_hz)

    fig, axes = plt.subplots(1, len(N_STATION_TIERS), figsize=(14, 4.6), sharey=True)
    if len(N_STATION_TIERS) == 1:
        axes = [axes]
    ylbl = "Density" if density else "Event count"
    for ax, n in zip(axes, N_STATION_TIERS):
        _draw_picks_on_ax(
            ax,
            series_by_n[n],
            xlim=xlim,
            n_bins=n_bins,
            density=density,
            panel_title=rf"$N_{{\mathrm{{st}}}} = {n}$",
        )
        ax.set_xlabel(r"$\Delta t$ (ms)")
    axes[0].set_ylabel(ylbl)
    fig.suptitle(
        "Pick residual vs SeisBench " + title_suffix + xlim_note,
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(
    series: List[Tuple[str, List[float], str]],
    *,
    out: Path,
    xlim: Tuple[float, float],
    n_bins: int,
    density: bool,
    title_suffix: str,
    xlim_note: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    _draw_picks_on_ax(ax, series, xlim=xlim, n_bins=n_bins, density=density)
    ylbl = "Density" if density else "Event count"
    ax.set_xlabel(r"Pick error $\Delta t = t_{\mathrm{pick}} - t_{\mathrm{annotate}}$ (ms)")
    ax.set_ylabel(ylbl)
    ax.set_title(
        "Pick residual vs SeisBench " + title_suffix + xlim_note,
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Pick residual histograms vs annotate()")
    p.add_argument(
        "--jsonl",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "seisbench_matrix_lean_aff16.jsonl",
    )
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0, cpu, or cuda:0+cuda:1")
    p.add_argument("--model", type=str, default=None, help="Filter to one model label (optional)")
    p.add_argument("--fs-hz", type=float, default=DEFAULT_FS_HZ)
    p.add_argument(
        "--manual-xlim",
        action="store_true",
        help=(
            "Use fixed --xmin/--xmax (default ±0.5 ms). Without this flag, x limits "
            "are chosen automatically (symmetric about 0) from the pooled residuals "
            "so BF16/FP16 histograms are visible."
        ),
    )
    p.add_argument(
        "--xmin",
        type=float,
        default=DEFAULT_XMIN_MS,
        help=f"ms, only with --manual-xlim (default {DEFAULT_XMIN_MS})",
    )
    p.add_argument(
        "--xmax",
        type=float,
        default=DEFAULT_XMAX_MS,
        help=f"ms, only with --manual-xlim (default {DEFAULT_XMAX_MS})",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help=f"Histogram bins across [xmin, xmax] (default {DEFAULT_BINS})",
    )
    p.add_argument(
        "--counts",
        action="store_true",
        help="Y-axis event counts (default: density)",
    )
    p.add_argument(
        "--per-n-stations",
        action="store_true",
        help="One panel per station-count tier (64 / 256 / 580); shared x-limits across panels.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "figures"
        / "affinity"
        / "pick_residual_histogram.png",
    )
    args = p.parse_args()

    rows = load_jsonl(args.jsonl)
    ann = annotate_onset_index(rows, args.device)
    if not ann:
        print("No annotate() rows with onset_p found for device", args.device)
        return 1

    mod_slug = (args.model or "all_models").replace(" ", "_")
    dev_slug = args.device.replace(":", "")

    series = build_series(
        rows, ann, device=args.device, model=args.model, fs_hz=args.fs_hz, n_stations=None
    )

    if not any(d for _, d, _ in series):
        print("No matched lean vs annotate pick pairs; check --device / --model / JSONL path.")
        return 1

    ms_per_sample = 1000.0 / args.fs_hz
    pooled: List[float] = []
    for _, data, _ in series:
        pooled.extend(data)

    manual_bounds: Optional[Tuple[float, float]] = None
    if args.manual_xlim:
        manual_bounds = (args.xmin, args.xmax)
        xlim = manual_bounds
        xlim_note = ""
        if pooled:
            n_out = sum(1 for x in pooled if x < xlim[0] or x > xlim[1])
            if n_out == len(pooled):
                print(
                    f"Warning: all {len(pooled)} residuals fall outside manual "
                    f"[{xlim[0]}, {xlim[1]}] ms (native step ≈ {ms_per_sample:g} ms/sample). "
                    "Omit --manual-xlim for automatic symmetric limits.",
                    file=sys.stderr,
                )
    else:
        xlim, xlim_note = _auto_symmetric_xlim(pooled, fs_hz=args.fs_hz)

    title_suffix = f"annotate() ({args.device})"
    if args.model:
        title_suffix = f"{args.model}, {title_suffix}"

    out_path = args.out
    if out_path.name == "pick_residual_histogram.png" and args.model:
        suffix = "_by_nst" if args.per_n_stations else ""
        out_path = out_path.parent / f"pick_residual_{mod_slug}_{dev_slug}{suffix}.png"

    if args.per_n_stations:
        try:
            plot_faceted_by_n_stations(
                rows,
                ann,
                out=out_path,
                device=args.device,
                model=args.model,
                fs_hz=args.fs_hz,
                n_bins=args.bins,
                density=not args.counts,
                title_suffix=title_suffix,
                manual_xlim=manual_bounds,
            )
        except ValueError as e:
            print(e, file=sys.stderr)
            return 1
    else:
        xlim_note = xlim_note + r", all $N_{\mathrm{st}}$ pooled"
        plot_histogram(
            series,
            out=out_path,
            xlim=xlim,
            n_bins=args.bins,
            density=not args.counts,
            title_suffix=title_suffix,
            xlim_note=xlim_note,
        )
    print("Wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
