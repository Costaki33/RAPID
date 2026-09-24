"""Shared loaders and plotting grammar for the manuscript figures.

Every figure follows one rule: y is the measured quantity, x is the single
parameter being varied, and every other parameter is held at a locked
reference config.  A faint dashed line shows the best value over all other
settings at each x (the number the manuscript text quotes).  Every isolated
repeat is drawn as a translucent dot so the spread is visible.

Native sweep root:  results/annotate_config/stead_iso_2026-09-08
Orch root:          results/orch_annotate/stead_iso_2026-09-19_native_winners
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAPID = Path(__file__).resolve().parents[2]
for p in (RAPID, RAPID / "benchmarks" / "fair"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_annotate_precision_trial import (  # noqa: E402
    CONFIG_BATCH_GRID,
    CONFIG_CPU_GRID,
    CONFIG_TAG,
    CONFIG_THREAD_MULTS,
    DTYPE_OF,
    MODELS as _MODEL_SPECS,
    config_result_relpath,
    iter_config_matrix,
)
import native_winner_orch as nwo  # noqa: E402

NATIVE_ROOT = RAPID / "results" / "annotate_config" / "stead_iso_2026-09-08"
ORCH_ROOT = RAPID / "results" / "orch_annotate" / "stead_iso_2026-09-19_native_winners"
FIG_DIR = NATIVE_ROOT / "figures"
CACHE_DIR = FIG_DIR / "cache"

MODELS: Tuple[str, ...] = tuple(_MODEL_SPECS)  # PhaseNet ... EQCCTP
CPUS: Tuple[int, ...] = tuple(CONFIG_CPU_GRID)
BATCHES: Tuple[int, ...] = tuple(CONFIG_BATCH_GRID)
MULTS: Tuple[float, ...] = tuple(CONFIG_THREAD_MULTS)
DTYPES: Tuple[str, ...] = ("fp32", "fp16", "bf16")
DEVICES: Tuple[str, ...] = ("cpu", "gpu")
STATIONS: Tuple[int, ...] = (250, 580)
QS_THRESH = nwo.QS_THRESH
TIMEOUT_S = 1800.0

# Locked reference config (everything that is not on the x-axis).
REF_BATCH = 512
REF_MULT = 1.0
REF_CPUS = 20
REF_PACKAGING = "merged"

MODEL_LABEL = {
    "PhaseNet": "PhaseNet",
    "PhaseNetLight": "PhaseNetLight",
    "EQTransformer": "EQTransformer",
    "EQT-NC": "EQTransformer-NonConservative",
    "EQCCT": "EQCCT (P+S)",
    "EQCCTP": "EQCCTP",
}
# Okabe-Ito.
MODEL_COLOR = {
    "PhaseNet": "#E69F00",
    "PhaseNetLight": "#009E73",
    "EQTransformer": "#CC79A7",
    "EQT-NC": "#56B4E9",
    "EQCCT": "#0072B2",
    "EQCCTP": "#D55E00",
}
DEVICE_LABEL = {"cpu": "CPU", "gpu": "GPU"}
DTYPE_LABEL = {"fp32": "FP32", "fp16": "FP16", "bf16": "BF16"}

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"


# ---------------------------------------------------------------------------
# style
# ---------------------------------------------------------------------------
def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.65,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def tidy_axis(ax, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.grid(True, axis="y", linewidth=0.4, color=GRID, alpha=0.9)


def panel_grid(
    rows: Sequence[Any],
    cols: Sequence[Any],
    *,
    width: float = 6.9,
    row_height: float = 2.35,
    sharey: str | bool = "row",
):
    fig, axes = plt.subplots(
        len(rows),
        len(cols),
        figsize=(width, row_height * len(rows)),
        squeeze=False,
        sharey=sharey,
    )
    return fig, axes


def model_legend(fig, *, models: Sequence[str] = MODELS, extra: Optional[List] = None, ncol: int = 6, y: float = -0.01, markers_only: bool = False):
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0], color=MODEL_COLOR[m], marker="o", markersize=5.0,
            linewidth=0 if markers_only else 1.6,
            markerfacecolor=MODEL_COLOR[m], markeredgecolor=MODEL_COLOR[m],
            label=MODEL_LABEL[m],
        )
        for m in models
    ]
    if extra:
        handles.extend(extra)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.1, handlelength=2.2)


DTYPE_MARKER = {"bf16": "o", "fp32": "D", "fp16": "^"}


def marker_for_dtype(dtype: Optional[str]) -> str:
    return DTYPE_MARKER.get(str(dtype or ""), "o")


def ref_dashed_handles(
    labels: Optional[Tuple[str, str]] = None,
    *,
    best_linestyle: str = "--",
    best_marker: str = "o",
    unsafe_label: Optional[str] = None,
    dtype_markers: bool = False,
) -> List:
    from matplotlib.lines import Line2D

    solid, minimum = labels or ("Mean quality-safe config", "Minimum quality-safe config")
    if dtype_markers:
        handles = [
            Line2D([0], [0], color=INK, linewidth=1.6, label=solid),
            Line2D([0], [0], color=INK, linewidth=1.0, linestyle=best_linestyle, label=minimum),
            Line2D([0], [0], color=INK, linewidth=0, marker="o", markersize=4.5, label="BF16"),
            Line2D([0], [0], color=INK, linewidth=0, marker="D", markersize=4.5, label="FP32"),
            Line2D([0], [0], color=INK, linewidth=0, marker="^", markersize=4.8, label="FP16"),
        ]
    else:
        handles = [
            Line2D([0], [0], color=INK, linewidth=1.6, marker="o", markersize=4.5, label=solid),
            Line2D(
                [0], [0], color=INK, linewidth=1.0, linestyle=best_linestyle, marker=best_marker,
                markerfacecolor=("white" if best_marker == "o" else INK), markersize=4.2, alpha=0.85, label=minimum,
            ),
        ]
    if unsafe_label:
        handles.append(Line2D([0], [0], color=INK, linewidth=0, marker="x", markersize=5.5, markeredgewidth=1.3, label=unsafe_label))
    return handles


# ---------------------------------------------------------------------------
# native sweep loader
# ---------------------------------------------------------------------------
NATIVE_FIELDS = [
    "model", "n_stations", "device", "n_cpus", "thread_mult", "torch_threads",
    "batch_size", "dtype", "packaging", "repeat_index", "inference_s",
    "peak_pss_mb", "peak_vram_mb", "p_f1", "s_f1", "p_f1_vs_fp32", "s_f1_vs_fp32",
    "quality_safe", "timed_out", "ok",
]


def _is_timed_out(result: dict) -> bool:
    timing = result.get("timing") or {}
    meta = result.get("meta") or {}
    return bool(timing.get("timed_out") or meta.get("timed_out") or meta.get("timed_out_by"))


def _fnum(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def load_native_rows(root: Path = NATIVE_ROOT, *, use_cache: bool = True) -> List[dict]:
    """One row per (cell, repeat).  Timed-out / failed cells yield a single
    row with ``ok=False`` so they can be drawn as censored markers."""
    cache = CACHE_DIR / f"native_rows_{root.name}.json"
    if use_cache and cache.is_file() and cache.stat().st_mtime > _newest_mtime(root):
        return json.loads(cache.read_text())
    rows: List[dict] = []
    for cell in iter_config_matrix():
        p = root / config_result_relpath(cell, tag=CONFIG_TAG)
        if not p.is_file():
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        timing = r.get("timing") or {}
        meta = r.get("meta") or {}
        dtype = meta.get("dtype") or DTYPE_OF[cell["method"]]
        vs = r.get("pick_quality_vs_fp32") or {}
        cat = r.get("pick_quality_vs_catalog") or {}
        p_vs = _fnum(vs.get("P.f1_mean"))
        qs = dtype == "fp32" or (p_vs is not None and p_vs >= QS_THRESH)
        timed = _is_timed_out(r)
        ok = (not timed) and _fnum(timing.get("success_rate")) == 1.0
        base = {
            "model": cell["model"],
            "n_stations": int(cell["n_stations"]),
            "device": cell["device"],
            "n_cpus": int(cell["n_cpus"]),
            "thread_mult": float(cell["thread_mult"]),
            "torch_threads": int(cell["torch_threads"]),
            "batch_size": int(cell["batch_size"]),
            "dtype": dtype,
            "packaging": cell["packaging"],
            "p_f1": _fnum(cat.get("P.f1_mean")),
            "s_f1": _fnum(cat.get("S.f1_mean")),
            "p_f1_vs_fp32": p_vs,
            "s_f1_vs_fp32": _fnum(vs.get("S.f1_mean")),
            "quality_safe": bool(qs),
            "timed_out": bool(timed),
            "ok": bool(ok),
        }
        reps = timing.get("repeats") or []
        emitted = 0
        if ok:
            for rec in reps:
                if not rec.get("success"):
                    continue
                inf = _fnum(rec.get("inference_s"))
                if inf is None or inf < 0:
                    continue
                rows.append({
                    **base,
                    "repeat_index": int(rec.get("repeat_index", emitted)),
                    "inference_s": inf,
                    "peak_pss_mb": _fnum(rec.get("peak_pss_mb")),
                    "peak_vram_mb": _fnum(rec.get("peak_vram_mb")),
                })
                emitted += 1
        if emitted == 0:
            rows.append({**base, "ok": False, "repeat_index": -1, "inference_s": None,
                         "peak_pss_mb": None, "peak_vram_mb": None})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def _newest_mtime(root: Path) -> float:
    # cheap proxy: the launcher log is appended for every finished cell
    cands = [root / "iso_annotate_config.log", root / "winners_merged_quality_safe.json"]
    return max((c.stat().st_mtime for c in cands if c.is_file()), default=0.0)


def load_winners(path: Path = NATIVE_ROOT / nwo.WINNERS_NAME) -> Dict[str, dict]:
    return nwo.load_winners(path)


def ref_dtype(winners: Dict[str, dict], model: str, n_stations: int, device: str, n_cpus: int = REF_CPUS) -> str:
    """Locked reference dtype = dtype of the quality-safe merged winner."""
    return winners[nwo.winner_key(model, n_stations, device, n_cpus)]["dtype"]


# ---------------------------------------------------------------------------
# generic selection helpers
# ---------------------------------------------------------------------------
CELL_KEYS = ("model", "n_stations", "device", "n_cpus", "thread_mult", "batch_size", "dtype", "packaging")


def _match(row: dict, fixed: dict) -> bool:
    for k, v in fixed.items():
        if callable(v):
            if not v(row.get(k)):
                return False
        elif isinstance(v, (set, frozenset, list, tuple)):
            if row.get(k) not in v:
                return False
        elif row.get(k) != v:
            return False
    return True


def select(rows: Iterable[dict], **fixed) -> List[dict]:
    return [r for r in rows if _match(r, fixed)]


def cell_means(rows: Iterable[dict], y_key: str, keys: Sequence[str] = CELL_KEYS) -> Dict[tuple, dict]:
    """Aggregate repeat rows into per-cell mean/min/values of ``y_key``."""
    acc: Dict[tuple, dict] = {}
    for r in rows:
        k = tuple(r.get(c) for c in keys)
        d = acc.setdefault(k, {"cell": {c: r.get(c) for c in keys}, "values": [], "ok": r.get("ok", True),
                               "timed_out": r.get("timed_out", False), "quality_safe": r.get("quality_safe", True)})
        y = r.get(y_key)
        if y is not None and r.get("ok", True):
            d["values"].append(float(y))
    for d in acc.values():
        d["mean"] = float(np.mean(d["values"])) if d["values"] else None
        d["min"] = float(min(d["values"])) if d["values"] else None
    return acc


def locked_series(rows: Iterable[dict], *, x_key: str, y_key: str, fixed: dict) -> Dict[Any, List[float]]:
    """x -> list of per-repeat y values for the locked reference config."""
    out: Dict[Any, List[float]] = defaultdict(list)
    for r in select(rows, **fixed):
        y = r.get(y_key)
        if y is not None and r.get("ok", True):
            out[r[x_key]].append(float(y))
    return dict(out)


def best_over_rest(
    rows: Iterable[dict],
    *,
    x_key: str,
    y_key: str,
    fixed: dict,
    quality_safe_only: bool = True,
    agg: str = "mean",
) -> Dict[Any, Tuple[float, dict]]:
    """x -> (best cell-mean y, cell) over every non-fixed parameter."""
    sub = select(rows, **fixed)
    if quality_safe_only:
        sub = [r for r in sub if r.get("quality_safe", True)]
    cells = cell_means(sub, y_key)
    best: Dict[Any, Tuple[float, dict]] = {}
    for d in cells.values():
        v = d[agg]
        if v is None:
            continue
        x = d["cell"][x_key]
        if x not in best or v < best[x][0]:
            best[x] = (v, d["cell"])
    return best


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def x_positions(xs: Sequence[Any], *, log: bool = False) -> Dict[Any, float]:
    if log:
        return {x: float(x) for x in xs}
    return {x: float(i) for i, x in enumerate(xs)}


def plot_series(
    ax,
    *,
    xs: Sequence[Any],
    pos: Dict[Any, float],
    locked: Dict[Any, List[float]],
    best: Optional[Dict[Any, Tuple[float, dict]]] = None,
    color: str,
    model_index: int,
    n_models: int,
    spread: float = 0.36,
    label: Optional[str] = None,
    marker: str = "o",
    linestyle: str = "-",
    hollow_x: Optional[set] = None,
    censored: Optional[Dict[Any, float]] = None,
    log_x: bool = False,
    points_only: bool = False,
    lines_only: bool = False,
    align_x: bool = False,
    unsafe_marker: Optional[str] = None,
    best_linestyle: str = "--",
    best_marker: str = "o",
    repeat_marker: Optional[str] = None,
    best_marker_for_x: Optional[Dict[Any, str]] = None,
) -> float:
    """Repeat dots + solid mean line (locked) + dashed best line.  Returns ymax.

    ``points_only`` draws each locked-config repeat as a filled marker and
    skips the mean line and the best-over-other-settings series.
    ``lines_only`` draws the locked mean on the category tick, with each
    repeat as a faint point on that same tick behind the mean marker.
    """
    # Points-only clusters stay close to the category tick so a point cannot
    # be read as belonging to the neighboring category.
    point_spread = 0.11 if points_only else spread
    model_off = 0.0 if lines_only or align_x else np.linspace(-point_spread, point_spread, n_models)[model_index]
    ymax = 0.0
    if points_only:
        for x in xs:
            vals = locked.get(x)
            if not vals:
                continue
            base = pos[x]
            off = model_off * (base * 0.08 if log_x else 1.0)
            rep_off = np.linspace(-0.018, 0.018, len(vals)) * (base * 0.08 if log_x else 1.0)
            for i, v in enumerate(vals):
                ax.scatter(
                    base + off + rep_off[i], v, s=28, marker=marker, color=color,
                    alpha=0.9, linewidths=0, zorder=4,
                )
                ymax = max(ymax, v)
        if censored:
            for x, y in censored.items():
                base = pos[x]
                off = model_off * (base * 0.08 if log_x else 1.0)
                ax.scatter(
                    [base + off], [y], marker="^", s=78, color=color,
                    edgecolors=color, linewidths=0.4, zorder=6, clip_on=False,
                )
                ymax = max(ymax, y)
        return ymax
    xm, ym = [], []
    for x in xs:
        vals = locked.get(x)
        if not vals:
            continue
        base = pos[x]
        off = model_off * (base * 0.08 if log_x else 1.0)
        rep_off = np.linspace(-0.045, 0.045, len(vals)) * (base * 0.08 if log_x else 1.0)
        for i, v in enumerate(vals):
            ax.scatter(base + off + rep_off[i], v, s=14, marker=("o" if repeat_marker is None else repeat_marker), color=color, alpha=0.35, linewidths=0, zorder=3)
            ymax = max(ymax, v)
        xm.append(base + off)
        ym.append(float(np.mean(vals)))
        ymax = max(ymax, float(np.mean(vals)))
    hollow_x = hollow_x or set()
    if xm:
        ax.plot(xm, ym, color=color, linestyle=linestyle, linewidth=1.5, zorder=4, label=label)
        for xx, yy, x in zip(xm, ym, [x for x in xs if locked.get(x)]):
            if unsafe_marker and x in hollow_x:
                ax.scatter(xx, yy, s=36, marker=unsafe_marker, color=color, linewidths=1.4, zorder=6)
            else:
                ax.scatter(xx, yy, s=(34 if marker in ("D", "^") else 22), marker=marker,
                           facecolors=("white" if x in hollow_x else color),
                           edgecolors=color, linewidths=1.1, zorder=5)
    if best:
        xb, yb = [], []
        for x in xs:
            if x in best:
                base = pos[x]
                off = model_off * (base * 0.08 if log_x else 1.0)
                xb.append(base + off)
                yb.append(best[x][0])
                ymax = max(ymax, best[x][0])
        if xb:
            ax.plot(xb, yb, color=color, linestyle=best_linestyle, linewidth=1.05, alpha=0.8, zorder=2)
            xs_best = [x for x in xs if x in best]
            for xx, yy, x in zip(xb, yb, xs_best):
                mk = best_marker_for_x.get(x, best_marker) if best_marker_for_x else best_marker
                ax.scatter(
                    [xx], [yy], s=(30 if mk in ("D", "^") else 22), marker=mk,
                    facecolors=("white" if (best_marker_for_x or mk == "o") else color),
                    edgecolors=color, linewidths=0.9, alpha=0.95, zorder=2,
                )
    if censored:
        for x, y in censored.items():
            base = pos[x]
            off = model_off * (base * 0.08 if log_x else 1.0)
            ax.annotate("", xy=(base + off, y * 1.25), xytext=(base + off, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0), zorder=6)
            ymax = max(ymax, y * 1.3)
    return ymax


def set_categorical_x(ax, xs: Sequence[Any], pos: Dict[Any, float], labels: Optional[Sequence[str]] = None, pad: float = 0.6) -> None:
    ax.set_xticks([pos[x] for x in xs])
    ax.set_xticklabels(labels if labels is not None else [str(x) for x in xs])
    ax.set_xlim(min(pos.values()) - pad, max(pos.values()) + pad)


def output_dir(stem: str) -> Path:
    """One subfolder per figure number, e.g. figures/fig02/."""
    prefix = stem.split("_", 1)[0]
    out = FIG_DIR / prefix if prefix.startswith("fig") else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    return out


def save(fig, stem: str) -> Tuple[Path, Path]:
    dest = output_dir(stem)
    png = dest / f"{stem}.png"
    pdf = dest / f"{stem}.pdf"
    fig.savefig(png, dpi=600)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"wrote {png}")
    return png, pdf


def write_csv(stem: str, rows: List[dict], fields: Optional[Sequence[str]] = None) -> Path:
    path = output_dir(stem) / f"{stem}.csv"
    if not rows:
        path.write_text("")
        return path
    fields = list(fields) if fields else sorted({k for r in rows for k in r})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")
    return path


def report(stem: str, n_points: int, n_expected: int, missing: Sequence[str] = ()) -> None:
    flag = "" if n_points == n_expected else "   <-- MISMATCH"
    print(f"[{stem}] plotted {n_points} points (expected {n_expected}){flag}")
    for m in missing:
        print(f"    missing/censored: {m}")


# ---------------------------------------------------------------------------
# standard native runtime figure: rows = stations, cols = device
# ---------------------------------------------------------------------------
def native_runtime_figure(
    *,
    stem: str,
    rows: List[dict],
    winners: Dict[str, dict],
    x_key: str,
    xs: Sequence[Any],
    locked_fixed: Callable[[str, int, str], dict],
    best_fixed: Callable[[str, int, str], dict],
    x_label: str,
    x_tick_labels: Optional[Sequence[str]] = None,
    y_key: str = "inference_s",
    y_label: str = "Inference Time (s)",
    log_y: bool = False,
    log_x: bool = False,
    title_fn: Optional[Callable[[int, str], str]] = None,
    stations: Sequence[int] = STATIONS,
    devices: Sequence[str] = DEVICES,
    show_best: bool = True,
    censor_timeouts: bool = False,
    hollow_unsafe: bool = False,
    sharey: str | bool = "row",
    legend_y: float = -0.02,
    note: Optional[str] = None,
    points_only: bool = False,
    lines_only: bool = False,
    series_labels: Optional[Tuple[str, str]] = None,
    align_x: bool = False,
    unsafe_marker: Optional[str] = None,
    unsafe_label: Optional[str] = None,
    best_linestyle: str = "--",
    best_marker: str = "o",
    legend_ncol: Optional[int] = None,
    dense_log_yticks: bool = False,
    dtype_markers: bool = False,
) -> None:
    configure_style()
    fig, axes = panel_grid(stations, devices, sharey=sharey)
    pos = x_positions(xs, log=log_x)
    csv_rows: List[dict] = []
    n_points = 0
    missing: List[str] = []
    censored_labels: List[str] = []
    n_censored = 0
    for ri, n_st in enumerate(stations):
        row_ymax = 0.0
        for ci, dev in enumerate(devices):
            ax = axes[ri][ci]
            ymax = 0.0
            n_timeout = 0
            for mi, model in enumerate(MODELS):
                lf = locked_fixed(model, n_st, dev)
                locked = locked_series(rows, x_key=x_key, y_key=y_key, fixed=lf)
                best = None if points_only or lines_only or not show_best else best_over_rest(
                    rows, x_key=x_key, y_key=y_key, fixed=best_fixed(model, n_st, dev)
                )
                censored: Dict[Any, float] = {}
                hollow: set = set()
                for x in xs:
                    sub = select(rows, **{**lf, x_key: x})
                    if not sub:
                        missing.append(f"{model} {n_st}st {dev} {x_key}={x}")
                        continue
                    if censor_timeouts and all(r["timed_out"] for r in sub):
                        censored[x] = TIMEOUT_S
                        n_censored += 1
                        n_timeout += 1
                        censored_labels.append(f"{model} {n_st}st {dev} {x_key}={x} timed out (drawn as triangle)")
                    if hollow_unsafe and not all(r.get("quality_safe", True) for r in sub):
                        hollow.add(x)
                    for r in sub:
                        if r["ok"] and r.get(y_key) is not None:
                            csv_rows.append({**{k: r[k] for k in CELL_KEYS}, "repeat_index": r["repeat_index"],
                                             y_key: r[y_key], "series": "locked", "quality_safe": r["quality_safe"]})
                            n_points += 1
                if best:
                    for x, (v, cell) in best.items():
                        csv_rows.append({**cell, "repeat_index": -1, y_key: v, "series": "best_over_rest", "quality_safe": True})
                locked_marker = marker_for_dtype(lf.get("dtype")) if dtype_markers else "o"
                best_markers = None
                if dtype_markers and best:
                    best_markers = {x: marker_for_dtype(cell.get("dtype")) for x, (_v, cell) in best.items()}
                ymax = max(ymax, plot_series(
                    ax, xs=xs, pos=pos, locked=locked, best=best, color=MODEL_COLOR[model],
                    model_index=mi, n_models=len(MODELS), hollow_x=hollow, censored=censored or None, log_x=log_x,
                    points_only=points_only, lines_only=lines_only, align_x=align_x,
                    unsafe_marker=unsafe_marker, best_linestyle=best_linestyle, best_marker=best_marker,
                    marker=locked_marker,
                    repeat_marker=locked_marker if dtype_markers else None,
                    best_marker_for_x=best_markers,
                ))
            if log_x:
                ax.set_xscale("log")
                ax.set_xticks([pos[x] for x in xs])
                ax.set_xticklabels(x_tick_labels if x_tick_labels else [str(x) for x in xs])
                ax.minorticks_off()
            else:
                set_categorical_x(ax, xs, pos, x_tick_labels)
            if n_timeout and points_only:
                ax.text(
                    max(pos.values()) + 0.16, TIMEOUT_S, "≥ 1800 s",
                    ha="left", va="center", fontsize=6.4, color=MUTED, clip_on=False,
                )
            row_ymax = max(row_ymax, ymax)
            title = title_fn(n_st, dev) if title_fn else f"{DEVICE_LABEL[dev]} · {n_st} stations"
            ax.set_title(title)
            if ri == len(stations) - 1:
                ax.set_xlabel(x_label)
            if ci == 0:
                ax.set_ylabel(y_label)
            tidy_axis(ax)
        for ci in range(len(devices)):
            ax = axes[ri][ci]
            if log_y:
                ax.set_yscale("log")
                if row_ymax > 0:
                    ax.set_ylim(top=row_ymax * 1.6)
                if dense_log_yticks:
                    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
                    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
                    ax.yaxis.set_minor_formatter(NullFormatter())
            elif row_ymax > 0:
                ax.set_ylim(0, row_ymax * 1.12)
    series = None if points_only or lines_only or not show_best else ref_dashed_handles(
        series_labels, best_linestyle=best_linestyle, best_marker=best_marker, unsafe_label=unsafe_label,
        dtype_markers=dtype_markers,
    )
    model_legend(
        fig,
        extra=None if points_only or lines_only or not show_best else series,
        ncol=legend_ncol if legend_ncol is not None else (6 if points_only or lines_only else 4),
        y=legend_y,
        markers_only=points_only,
    )
    if note:
        fig.text(0.5, legend_y - 0.035, note, ha="center", va="top", fontsize=6.8, color=MUTED)
    fig.tight_layout(rect=(0, 0.12 if dtype_markers else 0.08, 1, 1))
    save(fig, stem)
    write_csv(stem, csv_rows)
    n_expected = len(MODELS) * len(stations) * len(devices) * len(xs) * 3
    n_expected -= 3 * (len(missing) + n_censored)
    report(stem, n_points, n_expected, missing + censored_labels)


# ---------------------------------------------------------------------------
# orchestration loader
# ---------------------------------------------------------------------------
ORCH_FIELDS = [
    "stage", "composition", "model", "n_stations", "device", "n_cpus", "k", "dtype",
    "batch_size", "torch_threads", "arrival", "repeat_index", "makespan_s",
    "makespan_s_mean", "p95_e2e_s", "p50_e2e_s", "native_inference_s_mean", "promoted_k",
]


def load_orch_rows(root: Path = ORCH_ROOT, stages: Sequence[str] = ("A", "B", "C")) -> List[dict]:
    """One row per (cell, repeat) for Model-Actor / Ripper orch cells."""
    winners = load_winners()
    rows: List[dict] = []
    for stage in stages:
        for cell in nwo.iter_stage(stage, winners, results_root=root, tag=nwo.TAG):
            p = nwo.winner_cell_dir(root, cell, tag=nwo.TAG) / "result.json"
            r = nwo._result_ok(p)
            if r is None:
                continue
            k = int(cell["k_ma"] or cell["k_rp"])
            lat = ((r.get("latency") or {}).get("pooled_across_repeats") or {}).get("e2e_finish_minus_ready") or {}
            base = {
                "stage": stage,
                "composition": cell["composition"],
                "model": cell["model"],
                "n_stations": int(cell["n_stations"]),
                "device": cell["device"],
                "n_cpus": int(cell["n_cpus"]),
                "k": k,
                "dtype": cell["native_dtype"],
                "batch_size": int(cell["batch_size"]),
                "torch_threads": int(cell["torch_threads"]),
                "arrival": cell["arrival"],
                "makespan_s_mean": nwo._makespan(r),
                "p95_e2e_s": _fnum(lat.get("p95")),
                "p50_e2e_s": _fnum(lat.get("p50")),
                "native_inference_s_mean": _fnum(cell.get("native_inference_s_mean")),
                "promoted_k": cell.get("promoted_from_k"),
                "ok": True,
            }
            mem_by_rep = {
                int(rec.get("repeat_index", 0)): rec
                for rec in ((r.get("memory") or {}).get("repeats") or [])
            }
            for rec in (r.get("timing") or {}).get("repeats") or []:
                if not rec.get("success"):
                    continue
                i = int(rec.get("repeat_index", 0))
                mem = mem_by_rep.get(i) or {}
                rows.append({**base, "repeat_index": i,
                             "makespan_s": _fnum(rec.get("makespan_s")),
                             "peak_pss_mb": _fnum(mem.get("peak_pss_mb")),
                             "peak_vram_mb": _fnum(mem.get("peak_vram_mb"))})
    return rows


ORCH_CELL_KEYS = ("stage", "composition", "model", "n_stations", "device", "n_cpus", "k")


def orch_cells(rows: Iterable[dict], stage: str, **fixed) -> Dict[tuple, dict]:
    sub = select(rows, stage=stage, **fixed)
    return cell_means(sub, "makespan_s", ORCH_CELL_KEYS)


def elbow_k(ks_ms: Dict[int, float], rel: float = nwo.ELBOW_REL) -> Optional[int]:
    if not ks_ms:
        return None
    best = min(ks_ms.values())
    return min(k for k, v in ks_ms.items() if v <= best * (1.0 + rel))


K_TICKS_CPU = (1, 2, 4, 5, 10, 20)
K_TICKS_GPU = (1, 2, 4)
