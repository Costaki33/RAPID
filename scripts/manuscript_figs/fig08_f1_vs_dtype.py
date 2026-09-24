#!/usr/bin/env python3
"""Fig 9 - P-wave pick agreement by numerical format (580 stations).

Heatmap.  STEAD-Catalog columns: STEAD-Catalog P F1.  Versus-FP32 columns: F1 of
the reduced-precision picks against the FP32 picks of the same model, colored
by the 0.99 quality-safe threshold.  Locked: Merged, C=20, B=512, T=C.
Picks are deterministic per cell (one value, not per repeat).  EQTransformer
and EQTransformer-NonConservative have no FP16 cell.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle

import common as c

STEM = "fig09_p_f1_vs_dtype_580st"
N_ST = 580
COLS = (
    ("fp32", "catalog", "STEAD-\nCatalog"),
    ("fp16", "catalog", "STEAD-\nCatalog"),
    ("fp16", "vs", "vs FP32"),
    ("bf16", "catalog", "STEAD-\nCatalog"),
    ("bf16", "vs", "vs FP32"),
)
# Extra space after FP32 and after the FP16 pair so each format stays together.
XPOS = (0.0, 1.70, 3.25, 5.15, 6.70)
GROUPS = (("FP32", 0.0), ("FP16", 2.475), ("BF16", 5.925))
CAT_VMIN, CAT_VMAX = 0.64, 1.0
PASS_FACE, PASS_EDGE = "#D8EFE3", "#1B7F4E"
FAIL_FACE, FAIL_EDGE = "#F8D0BC", "#C4450A"
MISSING = "#F2F4F6"


def _ink_on(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.62 else c.INK


def _lookup(rows, model: str, dev: str):
    out = {}
    for dtype in c.DTYPES:
        sub = c.select(
            rows, model=model, n_stations=N_ST, device=dev, packaging=c.REF_PACKAGING,
            n_cpus=c.REF_CPUS, thread_mult=c.REF_MULT, batch_size=c.REF_BATCH, dtype=dtype,
        )
        out[dtype] = sub[0] if sub else None
    return out


def _draw(ax, rows, dev: str, cmap, norm) -> int:
    n_pts = 0
    for i, model in enumerate(c.MODELS):
        found = _lookup(rows, model, dev)
        for x, (dtype, kind, _label) in zip(XPOS, COLS):
            rec = found.get(dtype)
            value = None if rec is None else (rec["p_f1"] if kind == "catalog" else rec["p_f1_vs_fp32"])
            if kind == "vs" and dtype == "fp32":
                value = None
            if value is None:
                face, edge, text, tcol = MISSING, MISSING, "—", c.MUTED
            elif kind == "catalog":
                face = cmap(norm(value))
                edge = "white"
                text, tcol = f"{value:.3f}", _ink_on(face)
                n_pts += 1
            else:
                ok = value >= c.QS_THRESH
                face = PASS_FACE if ok else FAIL_FACE
                edge = PASS_EDGE if ok else FAIL_EDGE
                text, tcol = f"{value:.3f}", c.INK
                n_pts += 1
            ax.add_patch(Rectangle((x - 0.5, i - 0.5), 1.0, 1.0, facecolor=face, edgecolor=edge, linewidth=1.15, zorder=2))
            ax.text(x, i, text, ha="center", va="center", fontsize=7, color=tcol, zorder=3)
    ax.set_xlim(XPOS[0] - 0.62, XPOS[-1] + 0.62)
    ax.set_ylim(len(c.MODELS) - 0.55, -0.55)
    ax.set_xticks(list(XPOS))
    ax.set_xticklabels([])
    trans = ax.get_xaxis_transform()
    for name, x in GROUPS:
        ax.text(x, -0.04, name, transform=trans, ha="center", va="top", fontsize=8, fontweight="semibold", clip_on=False)
    for x, (_dtype, _kind, label) in zip(XPOS, COLS):
        ax.text(x, -0.15, label, transform=trans, ha="center", va="top", fontsize=6.2, color=c.INK, clip_on=False)
    ax.set_yticks(range(len(c.MODELS)))
    ax.set_yticklabels([c.MODEL_LABEL[m] for m in c.MODELS])
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{c.DEVICE_LABEL[dev]} · 580 st · Merged · C=20 · B=512 · T=C")
    return n_pts


def main() -> int:
    rows = c.load_native_rows()
    c.configure_style()
    fig, axes = c.panel_grid([0], c.DEVICES, width=8.4, row_height=3.05, sharey=True)
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(vmin=CAT_VMIN, vmax=CAT_VMAX)
    csv_rows = []
    n_pts = 0
    for ci, dev in enumerate(c.DEVICES):
        n_pts += _draw(axes[0][ci], rows, dev, cmap, norm)
        if ci:
            axes[0][ci].tick_params(labelleft=False)
        for model in c.MODELS:
            found = _lookup(rows, model, dev)
            for dtype in c.DTYPES:
                rec = found.get(dtype)
                if rec is None:
                    continue
                csv_rows.append({
                    "model": model, "device": dev, "n_stations": N_ST, "dtype": dtype,
                    "n_cpus": c.REF_CPUS, "batch_size": c.REF_BATCH, "thread_mult": c.REF_MULT,
                    "p_f1_catalog": rec["p_f1"], "s_f1_catalog": rec["s_f1"],
                    "p_f1_vs_fp32": rec["p_f1_vs_fp32"], "quality_safe": rec["quality_safe"],
                })
    fig.subplots_adjust(left=0.26, right=0.90, bottom=0.28, top=0.88, wspace=0.18)
    cax = fig.add_axes([0.915, 0.34, 0.014, 0.46])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("STEAD-Catalog P F1")
    cbar.ax.tick_params(length=2, width=0.4)
    cbar.outline.set_linewidth(0.4)
    fig.legend(
        handles=[
            Patch(facecolor=PASS_FACE, edgecolor=PASS_EDGE, label="Versus-FP32 ≥ 0.99"),
            Patch(facecolor=FAIL_FACE, edgecolor=FAIL_EDGE, label="Versus-FP32 < 0.99"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.55, 0.01), frameon=False,
    )
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, 52)  # 2 devices x (16 catalog + 10 vs-FP32; no EQT FP16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
