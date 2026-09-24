#!/usr/bin/env python3
"""Create RAPID manuscript Figure 1 as a high-resolution PNG only."""

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
GOLD = "#B88A22"
GOLD_LIGHT = "#F7F1E3"
SIGN = "#485766"
EXPONENT = "#3D8E88"
FRACTION = "#D7834F"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def rounded_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str = "",
    *,
    face: str = "white",
    edge: str = GRID,
    linewidth: float = 1.0,
    text_color: str = INK,
    fontsize: float = 8.2,
    weight: str = "normal",
    radius: float = 0.10,
    zorder: int = 2,
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            color=text_color,
            fontsize=fontsize,
            fontweight=weight,
            zorder=zorder + 1,
        )
    return box


def arrow(ax, start, end, *, color=MUTED, linewidth=1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            color=color,
            linewidth=linewidth,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=5,
        )
    )


def waveform(ax, x: float, y: float, width: float, amplitude: float, phase: float) -> None:
    t = np.linspace(0, 1, 120)
    envelope = np.exp(-((t - 0.58) / 0.17) ** 2)
    trace = 0.22 * np.sin(8 * np.pi * t + phase) + envelope * np.sin(33 * np.pi * t + phase)
    ax.plot(x + width * t, y + amplitude * trace, color=NAVY, linewidth=0.65, clip_on=False, zorder=5)


def draw_stream(ax) -> None:
    x, y, width, height = 0.15, 1.75, 2.40, 3.45
    rounded_box(ax, x, y, width, height, face=LIGHT, edge=GRID, linewidth=1.1)
    ax.text(
        x + width / 2,
        y + height - 0.30,
        "ObsPy network Stream",
        ha="center",
        va="center",
        color=INK,
        fontsize=8.6,
        fontweight="semibold",
    )
    for i, station in enumerate(("STA 01", "STA 02", "STA 03")):
        card_y = y + height - 1.22 - i * 0.78
        card_x = x + 0.20 + i * 0.06
        rounded_box(
            ax,
            card_x,
            card_y,
            width - 0.40,
            0.62,
            face="white",
            edge=GRID,
            linewidth=0.8,
            radius=0.06,
        )
        ax.text(card_x + 0.08, card_y + 0.31, station, ha="left", va="center", fontsize=5.3, color=MUTED)
        for j, channel in enumerate(("Z", "N", "E")):
            trace_y = card_y + 0.45 - j * 0.14
            ax.text(card_x + 0.83, trace_y, channel, ha="center", va="center", fontsize=5.2, color=MUTED)
            waveform(ax, card_x + 0.94, trace_y, 0.78, 0.055, phase=0.8 * (i + j))
    ax.text(
        x + width / 2,
        y + 0.27,
        "Multiple stations · three components",
        ha="center",
        va="center",
        fontsize=6.7,
        color=MUTED,
    )


def draw_picker(ax) -> None:
    x, y, width, height = 7.20, 1.75, 2.55, 3.45
    rounded_box(ax, x, y, width, height, face=LIGHT, edge=NAVY, linewidth=1.35)
    ax.text(
        x + width / 2,
        y + height - 0.31,
        "Deep-learning\nphase picker",
        ha="center",
        va="center",
        color=NAVY,
        fontsize=7.5,
        fontweight="semibold",
    )
    variants = [
        ("Annotate-FP32", "reference", "white", MUTED),
        ("Annotate-BF16", "selected", TEAL_LIGHT, TEAL),
        ("Annotate-FP16", "limited", ORANGE_LIGHT, ORANGE),
    ]
    for i, (name, note, face, edge) in enumerate(variants):
        by = y + height - 1.18 - i * 0.73
        rounded_box(
            ax,
            x + 0.27,
            by,
            width - 0.54,
            0.53,
            face=face,
            edge=edge,
            linewidth=1.05,
            radius=0.06,
        )
        ax.text(x + 0.39, by + 0.32, name, ha="left", va="center", fontsize=6.6, color=INK, fontweight="semibold")
        ax.text(x + width - 0.35, by + 0.16, note, ha="right", va="center", fontsize=5.6, color=edge)


def draw_probabilities(ax) -> None:
    x, y, width, height = 10.20, 1.75, 2.20, 3.45
    rounded_box(ax, x, y, width, height, face="white", edge=GRID, linewidth=1.0)
    ax.text(
        x + width / 2,
        y + height - 0.39,
        "Continuous\nprobability traces",
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="semibold",
        color=INK,
    )
    t = np.linspace(0, 1, 180)
    p = 0.08 + 0.82 * np.exp(-((t - 0.34) / 0.07) ** 2)
    s = 0.08 + 0.76 * np.exp(-((t - 0.72) / 0.09) ** 2)
    for baseline, values, label, color in ((y + 2.0, p, "P", TEAL), (y + 0.95, s, "S", ORANGE)):
        ax.plot([x + 0.33, x + width - 0.22], [baseline, baseline], color=GRID, linewidth=0.6)
        ax.plot(x + 0.33 + 1.62 * t, baseline + 0.60 * values, color=color, linewidth=1.35)
        ax.text(x + 0.20, baseline + 0.18, label, color=color, fontsize=7.5, fontweight="semibold")


def draw_picks(ax) -> None:
    x, y, width, height = 14.15, 1.75, 1.70, 3.45
    rounded_box(ax, x, y, width, height, face=GOLD_LIGHT, edge=GOLD, linewidth=1.0)
    ax.text(
        x + width / 2,
        y + height - 0.42,
        "Discrete\nP/S picks",
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="semibold",
        color=INK,
    )
    ax.plot([x + 0.28, x + width - 0.25], [y + 1.72, y + 1.72], color=GRID, linewidth=0.8)
    for px, label, color in ((x + 0.58, "P", TEAL), (x + 1.17, "S", ORANGE)):
        ax.plot([px, px], [y + 1.25, y + 2.18], color=color, linewidth=1.8)
        ax.text(px, y + 2.34, label, ha="center", va="center", color=color, fontsize=8.5, fontweight="bold")
    ax.text(
        x + width / 2,
        y + 0.53,
        "pick attributes",
        ha="center",
        va="center",
        fontsize=5.7,
        color=MUTED,
    )


def draw_pipeline_panel(ax) -> None:
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.55)
    ax.axis("off")
    ax.text(0.0, 6.32, "A", fontsize=11.5, fontweight="bold", color=INK, va="top")
    ax.text(0.46, 6.32, "Annotate workflow", fontsize=10.5, fontweight="semibold", color=INK, va="top")
    ax.text(
        2.95,
        5.67,
        "Unchanged preprocessing and batching",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
    )
    ax.plot([2.80, 6.92], [5.53, 5.53], color=GRID, linewidth=1.0)
    ax.text(
        11.25,
        5.67,
        "Unchanged reconstruction and pick extraction",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color=MUTED,
    )
    ax.plot([9.98, 15.82], [5.53, 5.53], color=GRID, linewidth=1.0)

    draw_stream(ax)
    rounded_box(
        ax,
        3.00,
        2.35,
        1.65,
        2.15,
        "Preprocessing\n/\nwindowing",
        face="white",
        edge=GRID,
        fontsize=8.0,
        weight="semibold",
    )
    rounded_box(
        ax,
        5.15,
        2.35,
        1.55,
        2.15,
        "Batch\nconstruction",
        face="white",
        edge=GRID,
        fontsize=8.0,
        weight="semibold",
    )
    draw_picker(ax)
    draw_probabilities(ax)
    rounded_box(
        ax,
        12.87,
        2.35,
        0.82,
        2.15,
        face="white",
        edge=GRID,
    )
    ax.text(
        13.28,
        3.43,
        "Classify",
        ha="center",
        va="center",
        rotation=90,
        fontsize=6.9,
        fontweight="semibold",
        color=INK,
    )
    draw_picks(ax)

    center_y = 3.43
    for start, end in (
        ((2.56, center_y), (2.98, center_y)),
        ((4.67, center_y), (5.13, center_y)),
        ((6.72, center_y), (7.18, center_y)),
        ((9.77, center_y), (10.18, center_y)),
        ((12.42, center_y), (12.85, center_y)),
        ((13.71, center_y), (14.13, center_y)),
    ):
        arrow(ax, start, end)

    ax.text(
        8.48,
        0.78,
        "One workflow; precision changes only inside model execution",
        ha="center",
        va="center",
        color=NAVY,
        fontsize=8.2,
        fontweight="semibold",
    )


def draw_float_bar(ax, y, name, total_bits, exponent_bits, fraction_bits) -> None:
    x, width, height = 1.95, 8.98, 0.70
    sign_w = width * (1 / total_bits)
    exponent_w = width * (exponent_bits / total_bits)
    fraction_w = width * (fraction_bits / total_bits)

    ax.text(0.05, y + height / 2, name, ha="left", va="center", fontsize=9.2, color=INK, fontweight="semibold")
    ax.text(1.58, y + height / 2, f"{total_bits}-bit", ha="right", va="center", fontsize=6.5, color=MUTED)

    segments = [
        (x, sign_w, SIGN, "1\nsign"),
        (x + sign_w, exponent_w, EXPONENT, f"{exponent_bits}\nexponent"),
        (x + sign_w + exponent_w, fraction_w, FRACTION, f"{fraction_bits}\nfraction"),
    ]
    for sx, sw, color, label in segments:
        ax.add_patch(Rectangle((sx, y), sw, height, facecolor=color, edgecolor="white", linewidth=0.8))
        if sw > 0.55:
            ax.text(sx + sw / 2, y + height / 2, label, ha="center", va="center", fontsize=6.7, color="white")

    # Keep the one-bit FP32 sign segment readable without crowding the panel title.
    if sign_w <= 0.55:
        ax.text(x + sign_w / 2, y + height / 2, "1", ha="center", va="center", fontsize=6.5, color="white")
        ax.text(x + sign_w / 2, y - 0.10, "sign", ha="center", va="top", fontsize=5.3, color=SIGN)


def draw_representation_panel(ax) -> None:
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4.25)
    ax.axis("off")
    ax.text(0.0, 4.05, "B", fontsize=11.5, fontweight="bold", color=INK, va="top")
    ax.text(0.46, 4.05, "Floating-point representation", fontsize=10.5, fontweight="semibold", color=INK, va="top")

    draw_float_bar(ax, 2.83, "FP32", 32, 8, 23)
    draw_float_bar(ax, 1.75, "BF16", 16, 8, 7)
    draw_float_bar(ax, 0.67, "FP16", 16, 5, 10)

    # Dynamic-range callouts.
    rounded_box(
        ax,
        11.52,
        2.15,
        4.15,
        1.15,
        face=TEAL_LIGHT,
        edge=TEAL,
        linewidth=1.0,
    )
    ax.text(
        13.60,
        2.82,
        "FP32 and BF16",
        ha="center",
        va="center",
        fontsize=8.0,
        color=TEAL,
        fontweight="semibold",
    )
    ax.text(
        13.60,
        2.47,
        "same 8-bit exponent\n≈ same dynamic range",
        ha="center",
        va="center",
        fontsize=6.9,
        color=INK,
        linespacing=1.2,
    )
    rounded_box(
        ax,
        11.52,
        0.68,
        4.15,
        1.15,
        face=ORANGE_LIGHT,
        edge=ORANGE,
        linewidth=1.0,
    )
    ax.text(
        13.60,
        1.35,
        "FP16",
        ha="center",
        va="center",
        fontsize=8.0,
        color=ORANGE,
        fontweight="semibold",
    )
    ax.text(
        13.60,
        1.00,
        "more fraction bits than BF16,\nbut a narrower representable range",
        ha="center",
        va="center",
        fontsize=6.9,
        color=INK,
        linespacing=1.2,
    )

    ax.text(
        5.95,
        0.14,
        "Segment widths are proportional within each format.",
        ha="center",
        va="center",
        fontsize=6.4,
        color=MUTED,
    )


def main() -> int:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.25, 5.55))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.55, 1.0], hspace=0.05)
    draw_pipeline_panel(fig.add_subplot(grid[0]))
    draw_representation_panel(fig.add_subplot(grid[1]))

    path = OUT / "figure01_annotate_reduced_precision.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
