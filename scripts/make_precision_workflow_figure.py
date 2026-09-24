#!/usr/bin/env python3
"""Methods diagram: Annotate-FP32 vs Annotate-BF16 / Annotate-FP16.

One shared path. Numerical format changes only inside the model forward
pass. classify_aggregate turns the probability traces into picks and is
outside the timed inference interval.

Output: docs/figures/paper_png/annotate_precision_workflow.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


RAPID = Path(__file__).resolve().parents[1]
OUT = RAPID / "docs" / "figures" / "paper_png"

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"
LIGHT = "#F4F7F9"
NAVY = "#24557A"
TEAL = "#2A8C82"
TEAL_LIGHT = "#E6F2F0"
ORANGE = "#D97745"
ORANGE_LIGHT = "#F9ECE6"
GOLD_LIGHT = "#F7F1E3"
SIGN = "#485766"
EXPONENT = "#3D8E88"
FRACTION = "#D7834F"
P_COLOR = "#24557A"
S_COLOR = "#B4532A"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
        }
    )


def rounded_box(ax, x, y, w, h, *, face="white", edge=GRID, lw=1.0, radius=0.08, z=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        mutation_aspect=None,
        zorder=z,
    )
    ax.add_patch(box)
    return box


def arrow(ax, start, end, *, color=MUTED, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            color=color,
            linewidth=lw,
            shrinkA=2,
            shrinkB=2,
            zorder=6,
            clip_on=False,
        )
    )


def label(ax, x, y, text, *, size=8, weight="normal", color=INK, ha="center", va="center", z=7):
    ax.text(x, y, text, ha=ha, va=va, fontsize=size, fontweight=weight, color=color, zorder=z, linespacing=1.15)


def bit_bar(ax, x, y, w, h, exponent_bits, fraction_bits):
    """Sign, exponent, and fraction segments. Widths match the bit counts."""
    total = 1 + exponent_bits + fraction_bits
    sw, ew = w / total, w * exponent_bits / total
    fw = w - sw - ew
    for dx, dw, face in ((0, sw, SIGN), (sw, ew, EXPONENT), (sw + ew, fw, FRACTION)):
        ax.add_patch(Rectangle((x + dx, y), dw, h, facecolor=face, edgecolor="white", linewidth=0.35, zorder=4))


def draw_flow(ax) -> None:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    label(ax, 1.2, 59.2, "One Annotate call, three numerical formats", size=12, weight="semibold", ha="left", va="center")
    label(
        ax,
        1.2,
        55.6,
        "Windowing, batching, and pick extraction stay the same. Only the model forward pass changes format.",
        size=8,
        color=MUTED,
        ha="left",
    )

    # Timed band is the Annotate call, through the probability traces.
    # Untimed band is classify_aggregate and the discrete picks.
    rounded_box(ax, 0.7, 22.8, 78.4, 29.4, face="#F3F7FB", edge="#D5E2EE", lw=0.8, radius=0.25, z=0)
    rounded_box(ax, 80.0, 22.8, 19.2, 29.4, face="#FBF8F2", edge="#E7DCC4", lw=0.8, radius=0.25, z=0)
    label(ax, 2.0, 50.2, "Timed  ·  Annotate", size=7.2, weight="semibold", color=NAVY, ha="left")
    label(ax, 81.2, 50.2, "Untimed", size=7.2, weight="semibold", color="#8A6A2F", ha="left")

    # Shared steps
    specs = [
        (1.6, 28.0, 11.6, 19.2, "3-component\nStream", "60 s · 100 Hz\nSTEAD waveforms"),
        (15.2, 28.0, 12.2, 19.2, "Windows", "3001 samples\nPhaseNet family\n\n6000 samples\nEQTransformer, EQCCT"),
        (29.4, 28.0, 11.4, 19.2, "Batch of B\nwindows", "B = 64, 128,\n256, or 512"),
    ]
    for x, y, w, h, title, body in specs:
        rounded_box(ax, x, y, w, h, face="white", edge=GRID, lw=1.05)
        label(ax, x + w / 2, y + h - 3.5, title, size=8.0, weight="semibold")
        label(ax, x + w / 2, y + 6.4, body, size=6.2, color=MUTED)
    label(ax, 21.2, 25.4, "identical for FP32, BF16, and FP16", size=6.2, color=MUTED)

    arrow(ax, (13.2, 37.6), (15.2, 37.6))
    arrow(ax, (27.4, 37.6), (29.4, 37.6))
    arrow(ax, (40.8, 37.6), (42.8, 37.6))

    rounded_box(ax, 42.8, 26.2, 16.6, 22.8, face="white", edge=NAVY, lw=1.35)
    label(ax, 51.1, 47.2, "Model forward", size=7.8, weight="semibold", color=NAVY)
    inner = [
        (43.6, 42.15, "Annotate-FP32", "white", MUTED, 8, 23),
        (43.6, 37.55, "Annotate-BF16", TEAL_LIGHT, TEAL, 8, 7),
        (43.6, 32.95, "Annotate-FP16", ORANGE_LIGHT, ORANGE, 5, 10),
    ]
    for x, y, name, face, edge, exp, frac in inner:
        rounded_box(ax, x, y, 15.0, 4.25, face=face, edge=edge, lw=1.0, radius=0.05)
        label(ax, x + 0.45, y + 3.05, name, size=6.4, weight="semibold", color=edge if edge != MUTED else INK, ha="left")
        label(ax, x + 14.55, y + 3.05, f"1+{exp}+{frac}", size=5.6, color=MUTED, ha="right")
        bit_bar(ax, x + 0.45, y + 0.42, 14.1, 1.2, exp, frac)
    label(ax, 51.1, 31.15, "EQCCT: EQCCTP then EQCCTS", size=5.3, color=MUTED)
    label(ax, 51.1, 29.45, "no FP16 cell for EQTransformer", size=5.3, color=ORANGE)
    label(ax, 51.1, 27.35, "only this step changes format", size=5.9, weight="semibold", color=NAVY)

    arrow(ax, (59.4, 37.6), (61.2, 37.6))

    rounded_box(ax, 61.2, 28.0, 16.4, 19.2, face="white", edge=GRID, lw=1.05)
    label(ax, 69.4, 44.2, "Continuous\nP and S traces", size=7.6, weight="semibold")
    _mini_probs(ax, 62.6, 29.6, 13.6, 10.2)
    label(ax, 69.4, 25.4, "Annotate output", size=6.2, color=MUTED)

    arrow(ax, (77.6, 42.6), (81.2, 42.6))

    rounded_box(ax, 81.2, 37.4, 16.8, 10.4, face="white", edge="#C4A15A", lw=1.1)
    label(ax, 89.6, 44.2, "classify_aggregate", size=7.4, weight="semibold")
    label(ax, 89.6, 40.4, "thresholds 0.3\ndoes not rerun the network", size=5.8, color=MUTED)

    arrow(ax, (89.6, 37.4), (89.6, 35.6))

    rounded_box(ax, 81.2, 26.6, 16.8, 8.2, face="white", edge="#C4A15A", lw=1.1)
    label(ax, 89.6, 32.0, "Discrete P and S picks", size=7.3, weight="semibold")
    label(ax, 89.6, 28.8, "excluded from inference time", size=5.7, color="#8A6A2F")


def _mini_probs(ax, x, y, w, h):
    t = np.linspace(0, 1, 160)
    p = np.exp(-((t - 0.32) / 0.07) ** 2)
    s = 0.82 * np.exp(-((t - 0.68) / 0.08) ** 2)
    x0 = x + 1.15
    span = w - 1.3
    ax.plot(x0 + span * t, y + 0.55 * h + 0.38 * h * p, color=P_COLOR, lw=1.15, zorder=5, clip_on=False)
    ax.plot(x0 + span * t, y + 0.10 * h + 0.38 * h * s, color=S_COLOR, lw=1.15, zorder=5, clip_on=False)
    label(ax, x + 0.05, y + h - 0.15, "P", size=6.0, color=P_COLOR, ha="left", va="top", weight="semibold")
    label(ax, x + 0.05, y + 0.42 * h, "S", size=6.0, color=S_COLOR, ha="left", va="bottom", weight="semibold")


def draw_bottom(ax_trace, ax_cards) -> None:
    t = np.linspace(0, 6, 600)
    fp32 = np.exp(-((t - 2.05) / 0.18) ** 2) + 0.72 * np.exp(-((t - 3.55) / 0.20) ** 2)
    bf16 = np.exp(-((t - 2.08) / 0.19) ** 2) + 0.70 * np.exp(-((t - 3.58) / 0.21) ** 2)
    fp16 = 0.78 * np.exp(-((t - 2.28) / 0.22) ** 2) + 0.55 * np.exp(-((t - 3.85) / 0.24) ** 2)

    ax_trace.plot(t, fp32, color=NAVY, lw=1.6, label="FP32", zorder=3)
    ax_trace.plot(t, bf16, color=TEAL, lw=1.35, ls="--", label="BF16", zorder=3)
    ax_trace.plot(t, fp16, color=ORANGE, lw=1.35, ls=":", label="FP16", zorder=3)
    ax_trace.axhline(0.3, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax_trace.text(5.05, 0.33, "0.3", color=MUTED, fontsize=6.2, ha="right", va="bottom")
    ax_trace.text(2.05, 1.05, "P", color=NAVY, fontsize=7, ha="center", fontweight="semibold")
    ax_trace.text(3.55, 0.82, "S", color=S_COLOR, fontsize=7, ha="center", fontweight="semibold")

    for x, color in ((2.05, NAVY), (2.08, TEAL), (2.28, ORANGE)):
        ax_trace.plot([x], [0.02], marker="v", color=color, ms=6, clip_on=False)
    ax_trace.set_xlim(0.4, 5.2)
    ax_trace.set_ylim(-0.02, 1.12)
    ax_trace.set_xlabel("time along one station (schematic)", fontsize=7.2, color=INK)
    ax_trace.set_ylabel("probability", fontsize=7.2, color=INK)
    ax_trace.set_title("Same windows, different forward pass", fontsize=8.4, color=INK, loc="left", pad=4)
    ax_trace.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=3)
    ax_trace.spines["top"].set_visible(False)
    ax_trace.spines["right"].set_visible(False)
    ax_trace.tick_params(labelsize=6.5, length=2.5, colors=INK)
    ax_trace.set_yticks([0, 0.3, 0.6, 1.0])
    ax_trace.text(
        0.0,
        -0.22,
        "Schematic, not a measured station. Triangles are the resulting P picks. A pick matches FP32 if it falls within 50 samples (0.5 s).",
        transform=ax_trace.transAxes,
        fontsize=6.3,
        color=MUTED,
        va="top",
    )

    ax_cards.set_xlim(0, 10)
    ax_cards.set_ylim(0, 10)
    ax_cards.axis("off")
    rounded_box(ax_cards, 0.2, 5.35, 9.5, 4.2, face="white", edge=NAVY, lw=1.1, radius=0.15)
    label(ax_cards, 4.95, 8.7, "STEAD-catalog F1", size=8.2, weight="semibold", color=NAVY)
    label(ax_cards, 4.95, 6.7, "Picks compared with the labeled\nP and S arrivals in STEAD.\nGreedy match, 50 samples.", size=6.7, color=MUTED)

    rounded_box(ax_cards, 0.2, 0.35, 9.5, 4.5, face=TEAL_LIGHT, edge=TEAL, lw=1.1, radius=0.15)
    label(ax_cards, 4.95, 4.05, "Versus-FP32 F1", size=8.2, weight="semibold", color=TEAL)
    label(
        ax_cards,
        4.95,
        2.15,
        "FP16 and BF16 picks compared\nwith the matched FP32 picks.\nKept when P-wave F1  ≥  0.99.",
        size=6.7,
        color=INK,
    )


def main() -> int:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.4, 6.35))
    ax_flow = fig.add_axes([0.012, 0.40, 0.976, 0.58])
    ax_trace = fig.add_axes([0.055, 0.10, 0.56, 0.26])
    ax_cards = fig.add_axes([0.66, 0.07, 0.31, 0.30])
    draw_flow(ax_flow)
    draw_bottom(ax_trace, ax_cards)
    path = OUT / "annotate_precision_workflow.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
