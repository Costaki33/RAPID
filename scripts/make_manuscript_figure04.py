#!/usr/bin/env python3
"""Create RAPID manuscript Figure 4: architecture and design dimensions.

The figure is conceptual and uses no experimental data. It exports PDF, SVG,
and a 600-dpi PNG.
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
BORDER = "#C9D2D9"
GRID = "#E3E8EC"
NEUTRAL = "#F1F3F5"
NEUTRAL_DARK = "#DDE2E6"

BLUE = "#24557A"
BLUE_LIGHT = "#E5EEF4"
TEAL = "#2A8C82"
TEAL_LIGHT = "#E4F1EF"
ORANGE = "#D97745"
ORANGE_LIGHT = "#F8ECE5"
GREEN = "#3F8068"
GREEN_LIGHT = "#E7F1ED"
RED = "#A84A42"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": BORDER,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def setup_axis(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def panel_frame(ax) -> None:
    ax.add_patch(
        Rectangle(
            (0.002, 0.005),
            0.996,
            0.94,
            facecolor="white",
            edgecolor=BORDER,
            linewidth=0.75,
            zorder=0,
        )
    )


def panel_title(ax, letter: str, title: str) -> None:
    ax.text(
        0.0,
        0.985,
        letter,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        0.038,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str = "",
    *,
    facecolor: str = "white",
    edgecolor: str = BORDER,
    textcolor: str = INK,
    fontsize: float = 7,
    fontweight: str = "normal",
    linewidth: float = 0.75,
    radius: float = 0.008,
    zorder: int = 2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=textcolor,
            fontweight=fontweight,
            linespacing=1.12,
            zorder=zorder + 1,
        )
    return patch


def arrow(
    ax,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = MUTED,
    linewidth: float = 0.8,
    mutation_scale: float = 7,
    zorder: int = 3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder,
        )
    )


def waveform(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    phase: float = 0.0,
    *,
    color: str = MUTED,
    linewidth: float = 0.45,
) -> None:
    samples = np.linspace(0, 1, 90)
    envelope = 0.16 + 0.84 * np.exp(-((samples - 0.62) / 0.16) ** 2)
    values = np.sin(2 * np.pi * (5.0 * samples + phase)) * envelope
    ax.plot(
        x + samples * width,
        y + values * height,
        color=color,
        linewidth=linewidth,
        clip_on=True,
        zorder=4,
    )


def station_stream(ax, x: float, y: float, label: str) -> None:
    width, height = 0.135, 0.105
    box(ax, x, y, width, height, facecolor=NEUTRAL, edgecolor=BORDER, radius=0.004)
    ax.text(
        x + 0.008,
        y + height / 2,
        label,
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=INK,
        zorder=5,
    )
    for index, channel in enumerate(("Z", "N", "E")):
        trace_y = y + 0.078 - index * 0.027
        ax.text(
            x + 0.050,
            trace_y,
            channel,
            ha="center",
            va="center",
            fontsize=4.6,
            color=MUTED,
            zorder=5,
        )
        waveform(
            ax,
            x + 0.060,
            trace_y,
            0.067,
            0.009,
            phase=0.22 * index,
            linewidth=0.38,
        )


def draw_panel_a(ax) -> None:
    setup_axis(ax)
    panel_frame(ax)
    panel_title(ax, "A", "Workload packaging")

    ax.text(
        0.092,
        0.865,
        "Network station Streams",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
    )
    station_stream(ax, 0.025, 0.665, "STA 01")
    station_stream(ax, 0.025, 0.535, "STA 02")
    station_stream(ax, 0.025, 0.405, "STA 03")
    ax.text(
        0.092,
        0.365,
        "·\n·\n·",
        ha="center",
        va="center",
        fontsize=6,
        color=MUTED,
        linespacing=0.45,
    )
    station_stream(ax, 0.025, 0.205, "STA N")

    arrow(ax, 0.170, 0.505, 0.215, 0.505)

    box(
        ax,
        0.218,
        0.235,
        0.190,
        0.540,
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        linewidth=0.9,
    )
    ax.text(
        0.313,
        0.725,
        "RAPID partitioning",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=TEAL,
    )
    ax.text(
        0.313,
        0.635,
        r"$G=\left\lceil \frac{N}{K}\right\rceil$",
        ha="center",
        va="center",
        fontsize=12,
        color=INK,
    )
    ax.text(
        0.313,
        0.567,
        "approximate stations\nper grouped task",
        ha="center",
        va="center",
        fontsize=6.2,
        color=MUTED,
        linespacing=1.15,
    )
    for index, text in enumerate(("Instance 1: 01, 1+K, …", "Instance 2: 02, 2+K, …", "Instance K: K, 2K, …")):
        box(
            ax,
            0.238,
            0.430 - index * 0.069,
            0.150,
            0.050,
            text,
            facecolor="white",
            edgecolor="#AFCDC8",
            fontsize=5.3,
            radius=0.003,
        )
    ax.text(
        0.313,
        0.272,
        "all-ready: round-robin assignment",
        ha="center",
        va="center",
        fontsize=5.7,
        color=MUTED,
    )

    ax.plot([0.408, 0.446], [0.505, 0.505], color=MUTED, linewidth=0.8)
    ax.plot([0.446, 0.446], [0.335, 0.685], color=MUTED, linewidth=0.8)
    arrow(ax, 0.446, 0.685, 0.483, 0.685)
    arrow(ax, 0.446, 0.335, 0.483, 0.335)

    # SG branch: one task contains several station Streams.
    box(
        ax,
        0.486,
        0.635,
        0.055,
        0.100,
        "SG",
        facecolor=TEAL,
        edgecolor=TEAL,
        textcolor="white",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        0.558,
        0.770,
        "Grouped multi-station task",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )
    box(
        ax,
        0.567,
        0.615,
        0.175,
        0.140,
        "STA 01 + STA 03 + …\nmerged Stream",
        facecolor=TEAL_LIGHT,
        edgecolor=TEAL,
        fontsize=6.5,
    )
    arrow(ax, 0.748, 0.685, 0.785, 0.685)
    box(
        ax,
        0.790,
        0.620,
        0.170,
        0.130,
        "one\nAnnotate-BF16 call",
        facecolor=BLUE,
        edgecolor=BLUE,
        textcolor="white",
        fontsize=6.8,
        fontweight="bold",
    )

    # S1 branch: each station remains a separate inference task.
    box(
        ax,
        0.486,
        0.285,
        0.055,
        0.100,
        "S1",
        facecolor=TEAL,
        edgecolor=TEAL,
        textcolor="white",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.text(
        0.558,
        0.420,
        "Single-station task",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
    )
    for index, station in enumerate(("STA 01 Stream", "STA 02 Stream", "STA 03 Stream")):
        center_y = 0.365 - index * 0.074
        box(
            ax,
            0.567,
            center_y - 0.027,
            0.145,
            0.054,
            station,
            facecolor=NEUTRAL,
            edgecolor=BORDER,
            fontsize=5.8,
            radius=0.003,
        )
        arrow(ax, 0.717, center_y, 0.755, center_y, mutation_scale=6)
        box(
            ax,
            0.760,
            center_y - 0.027,
            0.200,
            0.054,
            "one Annotate-BF16 call",
            facecolor=BLUE,
            edgecolor=BLUE,
            textcolor="white",
            fontsize=5.8,
            fontweight="bold",
            radius=0.003,
        )

    ax.plot([0.025, 0.975], [0.125, 0.125], color=GRID, linewidth=0.8)
    ax.text(
        0.5,
        0.072,
        "SG/S1 defines workload packaging — what each inference task contains.",
        ha="center",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=TEAL,
    )


def draw_ripper_row(ax, y: float, task_number: int) -> None:
    box(
        ax,
        0.035,
        y,
        0.120,
        0.105,
        f"Workload {task_number}",
        facecolor=NEUTRAL,
        edgecolor=BORDER,
        fontsize=6.2,
    )
    arrow(ax, 0.160, y + 0.052, 0.188, y + 0.052, mutation_scale=6)
    box(
        ax,
        0.193,
        y,
        0.130,
        0.105,
        "create\nworker",
        facecolor=ORANGE_LIGHT,
        edgecolor=ORANGE,
        fontsize=6.1,
        fontweight="bold",
    )
    arrow(ax, 0.328, y + 0.052, 0.352, y + 0.052, mutation_scale=6)
    box(
        ax,
        0.357,
        y,
        0.120,
        0.105,
        "load\nmodel",
        facecolor=ORANGE,
        edgecolor=ORANGE,
        textcolor="white",
        fontsize=6.1,
        fontweight="bold",
    )
    arrow(ax, 0.482, y + 0.052, 0.506, y + 0.052, mutation_scale=6)
    box(
        ax,
        0.511,
        y,
        0.155,
        0.105,
        "Annotate-BF16",
        facecolor=BLUE,
        edgecolor=BLUE,
        textcolor="white",
        fontsize=6.0,
        fontweight="bold",
    )
    arrow(ax, 0.671, y + 0.052, 0.695, y + 0.052, mutation_scale=6)
    box(
        ax,
        0.700,
        y,
        0.105,
        0.105,
        "return\npicks",
        facecolor=BLUE_LIGHT,
        edgecolor=BLUE,
        fontsize=6.0,
    )
    arrow(ax, 0.810, y + 0.052, 0.840, y + 0.052, mutation_scale=6)
    ax.plot(
        [0.858, 0.886],
        [y + 0.028, y + 0.076],
        color=RED,
        linewidth=1.15,
        zorder=4,
    )
    ax.plot(
        [0.858, 0.886],
        [y + 0.076, y + 0.028],
        color=RED,
        linewidth=1.15,
        zorder=4,
    )
    ax.text(
        0.920,
        y + 0.052,
        "worker\nterminates",
        ha="center",
        va="center",
        fontsize=5.5,
        color=MUTED,
        linespacing=1.05,
    )


def draw_panel_b(ax) -> None:
    setup_axis(ax)
    panel_frame(ax)
    panel_title(ax, "B", "Ripper")

    ax.text(
        0.5,
        0.875,
        "A new short-lived worker is created for every workload",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )
    draw_ripper_row(ax, 0.655, 1)
    draw_ripper_row(ax, 0.455, 2)
    draw_ripper_row(ax, 0.255, 3)

    ax.annotate(
        "",
        xy=(0.417, 0.760),
        xytext=(0.417, 0.255),
        arrowprops={
            "arrowstyle": "]-[",
            "color": ORANGE,
            "linewidth": 0.8,
            "mutation_scale": 7,
        },
    )
    ax.text(
        0.417,
        0.805,
        "repeated model initialization",
        ha="center",
        va="center",
        fontsize=5.8,
        color=ORANGE,
        fontweight="bold",
    )
    ax.plot([0.025, 0.975], [0.165, 0.165], color=GRID, linewidth=0.8)
    ax.text(
        0.5,
        0.085,
        "Ripper creates short-lived workers; model loading repeats for each task.",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=ORANGE,
    )


def draw_actor_lane(ax, y: float, actor_number: int) -> None:
    box(
        ax,
        0.035,
        y,
        0.930,
        0.135,
        facecolor=GREEN_LIGHT,
        edgecolor=GREEN,
        linewidth=0.8,
        radius=0.005,
    )
    ax.text(
        0.058,
        y + 0.068,
        f"Actor {actor_number}",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=GREEN,
    )
    box(
        ax,
        0.155,
        y + 0.025,
        0.170,
        0.085,
        "model resident",
        facecolor=GREEN,
        edgecolor=GREEN,
        textcolor="white",
        fontsize=5.8,
        fontweight="bold",
        radius=0.003,
    )
    arrow(ax, 0.330, y + 0.068, 0.360, y + 0.068, color=GREEN, mutation_scale=6)
    for x, workload in ((0.365, "W1"), (0.665, "W2")):
        box(
            ax,
            x,
            y + 0.032,
            0.070,
            0.072,
            workload,
            facecolor=NEUTRAL,
            edgecolor=BORDER,
            fontsize=5.7,
            radius=0.003,
        )
        arrow(ax, x + 0.075, y + 0.068, x + 0.098, y + 0.068, mutation_scale=5)
        box(
            ax,
            x + 0.103,
            y + 0.032,
            0.120,
            0.072,
            "Annotate",
            facecolor=BLUE,
            edgecolor=BLUE,
            textcolor="white",
            fontsize=5.5,
            fontweight="bold",
            radius=0.003,
        )
        arrow(ax, x + 0.228, y + 0.068, x + 0.248, y + 0.068, mutation_scale=5)
        ax.text(
            x + 0.260,
            y + 0.068,
            "ready",
            ha="left",
            va="center",
            fontsize=5.3,
            color=GREEN,
            fontweight="bold",
        )


def draw_panel_c(ax) -> None:
    setup_axis(ax)
    panel_frame(ax)
    panel_title(ax, "C", "Model-Actor")

    box(
        ax,
        0.185,
        0.795,
        0.150,
        0.095,
        "startup",
        facecolor=NEUTRAL,
        edgecolor=BORDER,
        fontsize=6.5,
        fontweight="bold",
    )
    arrow(ax, 0.340, 0.842, 0.390, 0.842)
    box(
        ax,
        0.395,
        0.780,
        0.275,
        0.125,
        "initialize K actors\nload + warm once per actor",
        facecolor=GREEN,
        edgecolor=GREEN,
        textcolor="white",
        fontsize=6.4,
        fontweight="bold",
    )
    ax.plot([0.532, 0.532], [0.780, 0.735], color=GREEN, linewidth=0.9)
    ax.plot([0.080, 0.532], [0.735, 0.735], color=GREEN, linewidth=0.9)
    ax.plot([0.080, 0.080], [0.690, 0.735], color=GREEN, linewidth=0.9)
    arrow(ax, 0.080, 0.690, 0.080, 0.660, color=GREEN, mutation_scale=6)

    draw_actor_lane(ax, 0.535, 1)
    draw_actor_lane(ax, 0.355, 2)
    draw_actor_lane(ax, 0.175, 3)

    ax.plot([0.025, 0.975], [0.135, 0.135], color=GRID, linewidth=0.8)
    ax.text(
        0.5,
        0.068,
        "Model-Actor loads once and reuses persistent model instances across workloads.",
        ha="center",
        va="center",
        fontsize=6.6,
        fontweight="bold",
        color=GREEN,
    )


def make_figure() -> plt.Figure:
    figure = plt.figure(figsize=(10.5, 8.0), facecolor="white")
    panel_a = figure.add_axes([0.045, 0.505, 0.910, 0.455])
    panel_b = figure.add_axes([0.045, 0.085, 0.440, 0.395])
    panel_c = figure.add_axes([0.515, 0.085, 0.440, 0.395])

    draw_panel_a(panel_a)
    draw_panel_b(panel_b)
    draw_panel_c(panel_c)

    figure.text(
        0.5,
        0.030,
        "Independent design choices: SG/S1 specifies task contents; "
        "Ripper/Model-Actor specifies how inference instances are managed.",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
    )
    figure.add_artist(
        plt.Line2D(
            [0.16, 0.84],
            [0.052, 0.052],
            transform=figure.transFigure,
            color=BORDER,
            linewidth=0.8,
        )
    )
    return figure


def main() -> int:
    configure_style()
    OUT.mkdir(parents=True, exist_ok=True)
    figure = make_figure()

    base = OUT / "figure04_rapid_architecture"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    figure.savefig(png_path, dpi=600, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(svg_path, facecolor="white")
    plt.close(figure)

    print(png_path)
    print(pdf_path)
    print(svg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
