#!/usr/bin/env python3
"""Fig 14 - Quality-safe Merged winners carried into RAPID.

One cell per model and host allocation.  Color is mean inference time.
Text is the configuration RAPID inherits: numerical format, batch size B,
and thread count T.  A red outline marks the lowest inference time in that
row.  Rows are network size; columns are device.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

import common as c

STEM = "fig14_winner_configs_carried_into_rapid"
BEST_EDGE = "#D62728"


def _ink_on(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else c.INK


def main() -> int:
    winners = c.load_winners()
    c.configure_style()
    fig, axes = c.panel_grid(c.STATIONS, c.DEVICES, width=7.6, row_height=3.15, sharey=True)
    cmap = plt.get_cmap("YlGnBu")
    times = [float(w["inference_s_mean"]) for w in winners.values()]
    norm = LogNorm(vmin=min(times) * 0.92, vmax=max(times) * 1.08)
    csv_rows = []
    n_pts = 0
    for ri, n_st in enumerate(c.STATIONS):
        for ci, dev in enumerate(c.DEVICES):
            ax = axes[ri][ci]
            for i, model in enumerate(c.MODELS):
                drawn = []
                for j, n_cpus in enumerate(c.CPUS):
                    rec = winners[c.nwo.winner_key(model, n_st, dev, n_cpus)]
                    val = float(rec["inference_s_mean"])
                    face = cmap(norm(val))
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", linewidth=1.4, zorder=2))
                    label = f"{c.DTYPE_LABEL[rec['dtype']]}\n$B$={int(rec['batch_size'])}  $T$={int(rec['torch_threads'])}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=6.2, color=_ink_on(face), zorder=3)
                    drawn.append((j, val, rec))
                    n_pts += 1
                fastest = min(val for _j, val, _rec in drawn)
                for j, val, rec in drawn:
                    is_fastest = val == fastest
                    if is_fastest:
                        ax.add_patch(Rectangle(
                            (j - 0.5, i - 0.5), 1, 1, facecolor="none",
                            edgecolor=BEST_EDGE, linewidth=2.3, zorder=5,
                        ))
                    csv_rows.append({
                        "model": model, "device": dev, "n_stations": n_st, "n_cpus": c.CPUS[j],
                        "dtype": rec["dtype"], "batch_size": int(rec["batch_size"]),
                        "thread_mult": rec["thread_mult"], "torch_threads": int(rec["torch_threads"]),
                        "inference_s_mean": val, "lowest_inference_time": is_fastest,
                    })
            ax.set_xlim(-0.5, len(c.CPUS) - 0.5)
            ax.set_ylim(len(c.MODELS) - 0.5, -0.5)
            ax.set_xticks(range(len(c.CPUS)))
            ax.set_xticklabels([str(x) for x in c.CPUS])
            ax.set_yticks(range(len(c.MODELS)))
            ax.set_yticklabels([c.MODEL_LABEL[m] for m in c.MODELS])
            ax.tick_params(length=0)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{c.DEVICE_LABEL[dev]} · {n_st} stations · Merged")
            if ri == len(c.STATIONS) - 1:
                ax.set_xlabel(r"Host Logical CPU Allocation $C$")
            if ci:
                ax.tick_params(labelleft=False)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.subplots_adjust(left=0.24, right=0.90, bottom=0.11, top=0.94, hspace=0.32, wspace=0.12)
    fig.legend(
        handles=[Patch(facecolor="white", edgecolor=BEST_EDGE, linewidth=1.8, label="Lowest inference time")],
        loc="lower center", bbox_to_anchor=(0.55, 0.0), frameon=False,
    )
    cax = fig.add_axes([0.915, 0.18, 0.016, 0.64])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Inference Time (s)")
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
    cbar.ax.yaxis.set_minor_formatter(NullFormatter())
    cbar.ax.tick_params(length=2, width=0.4, labelsize=7)
    cbar.outline.set_linewidth(0.4)
    c.save(fig, STEM)
    c.write_csv(STEM, csv_rows)
    c.report(STEM, n_pts, len(c.MODELS) * len(c.STATIONS) * len(c.DEVICES) * len(c.CPUS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
