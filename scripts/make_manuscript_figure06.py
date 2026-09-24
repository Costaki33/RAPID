#!/usr/bin/env python3
"""Create RAPID manuscript Figure 6: real-time streaming performance.

Pooled p95 finish-minus-ready latency is recalculated from every station event
across all five repeats. The script exports PDF, SVG, a 600-dpi PNG, the
processed plotting CSV, and a text validation report.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter


RAPID = Path(__file__).resolve().parents[1]
ROOT = RAPID / "results" / "orch_annotate" / "stead_2026-08-14"
OUT = RAPID / "docs" / "figures" / "paper_png"

MODELS = ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
SHORT = {
    "EQCCT": "EQCCT",
    "PhaseNet": "PN",
    "PhaseNetLight": "PNL",
    "EQTransformer": "EQT",
    "EQT-NC": "EQT-NC",
}
MODEL_COLORS = {
    "EQCCT": "#24557A",
    "PhaseNet": "#2A8C82",
    "PhaseNetLight": "#4C8BB8",
    "EQTransformer": "#B88A22",
    "EQT-NC": "#D97745",
}
MODEL_MARKERS = {
    "EQCCT": "o",
    "PhaseNet": "s",
    "PhaseNetLight": "^",
    "EQTransformer": "D",
    "EQT-NC": "P",
}

NETWORKS = [250, 580]
K_MA = {"cpu": [1, 2, 4, 5, 10, 20], "gpu": [1, 2, 4]}
K_WAIT = {"cpu": [5, 20], "gpu": [2, 4]}
HYBRID_SPLITS = {
    "cpu": [(5, 5), (4, 1)],  # total K=10 and K=5; both have matched MA cells
    "gpu": [(1, 1)],  # total K=2; matched MA cell
}
HYBRIDS = [
    ("ma_ontime_rp_delayed", "MA initial /\nRipper delayed"),
    ("rp_ontime_ma_delayed", "Ripper initial /\nMA delayed"),
]

INK = "#17212B"
MUTED = "#65727E"
GRID = "#D8E0E6"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.7,
            "axes.titleweight": "semibold",
            "axes.edgecolor": INK,
            "axes.linewidth": 0.75,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def result_dir(
    composition: str,
    network_size: int,
    model: str,
    device: str,
    k_ma: int,
    k_ripper: int,
    packaging: str,
    fill: str,
) -> Path:
    return (
        ROOT
        / composition
        / "annotate_bf16"
        / "stead"
        / f"{network_size}st"
        / model
        / device
        / f"kma{k_ma}_krp{k_ripper}"
        / packaging.lower()
        / "staggered"
        / fill
        / "bs512"
        / "orch_ann"
    )


@lru_cache(maxsize=None)
def pooled_p95(
    composition: str,
    network_size: int,
    model: str,
    device: str,
    k_ma: int,
    k_ripper: int,
    packaging: str,
    fill: str,
) -> dict:
    directory = result_dir(
        composition,
        network_size,
        model,
        device,
        k_ma,
        k_ripper,
        packaging,
        fill,
    )
    result_path = directory / "result.json"
    if not result_path.is_file():
        return {
            "status": "missing",
            "p95_s": None,
            "n_station_samples": 0,
            "source_file": str(result_path.relative_to(RAPID)),
        }

    result = json.loads(result_path.read_text())
    meta = result.get("meta") or {}
    expected = {
        "composition": composition,
        "n_stations": network_size,
        "model": model,
        "device": device,
        "k_ma": k_ma,
        "k_rp": k_ripper,
        "packaging": packaging.lower(),
        "arrival": "staggered",
        "fill": fill,
        "repeats": 5,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(
                f"Metadata mismatch in {result_path}: {key}={meta.get(key)!r}, "
                f"expected {value!r}"
            )

    latencies = []
    for repeat_index in range(5):
        repeat_path = directory / "repeats" / f"repeat_{repeat_index}.json"
        if not repeat_path.is_file():
            raise RuntimeError(f"Missing repeat file: {repeat_path}")
        repeat = json.loads(repeat_path.read_text())
        if repeat.get("success") is not True:
            raise RuntimeError(f"Unsuccessful repeat: {repeat_path}")
        events = repeat.get("station_events") or []
        if len(events) != network_size:
            raise RuntimeError(
                f"Expected {network_size} station events in {repeat_path}, "
                f"found {len(events)}"
            )
        for event in events:
            ready = float(event["ready_s"])
            finished = float(event["returned_s"])
            latency = finished - ready
            stored = float(event["e2e_s"])
            if not math.isclose(latency, stored, rel_tol=0.0, abs_tol=1e-9):
                raise RuntimeError(
                    f"Station latency mismatch in {repeat_path}: "
                    f"{latency} vs {stored}"
                )
            latencies.append(latency)

    expected_samples = network_size * 5
    if len(latencies) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} pooled samples in {result_path}, "
            f"found {len(latencies)}"
        )
    p95 = float(np.percentile(np.asarray(latencies, dtype=float), 95))
    stored_p95 = (
        (((result.get("latency") or {}).get("pooled_across_repeats") or {})
        .get("e2e_finish_minus_ready")
        or {})
        .get("p95")
    )
    if stored_p95 is None or not math.isclose(
        p95, float(stored_p95), rel_tol=0.0, abs_tol=1e-10
    ):
        raise RuntimeError(
            f"Recalculated pooled p95 disagrees with {result_path}: "
            f"{p95} vs {stored_p95}"
        )

    return {
        "status": "valid",
        "p95_s": p95,
        "n_station_samples": len(latencies),
        "source_file": str(result_path.relative_to(RAPID)),
    }


def base_row(
    panel: str,
    model: str,
    device: str,
    network_size: int,
    k_total: int,
    k_ma: int,
    k_ripper: int,
    packaging: str,
    composition: str,
    policy: str,
    condition: str,
    cell: dict,
    reference: dict | None,
    ratio: float | None,
) -> dict:
    return {
        "panel": panel,
        "model": model,
        "device": device,
        "network_size": network_size,
        "k_total": k_total,
        "k_ma": k_ma,
        "k_ripper": k_ripper,
        "packaging": packaging,
        "composition": composition,
        "policy": policy,
        "condition": condition,
        "status": cell["status"],
        "n_station_samples": cell["n_station_samples"],
        "p95_s": cell["p95_s"] if cell["p95_s"] is not None else "",
        "reference_p95_s": (
            reference["p95_s"]
            if reference is not None and reference["p95_s"] is not None
            else ""
        ),
        "ratio": ratio if ratio is not None else "",
        "source_file": cell["source_file"],
        "reference_source": reference["source_file"] if reference else "",
    }


def panel_a_rows() -> list[dict]:
    rows = []
    for network_size in NETWORKS:
        for model in MODELS:
            for device in ("cpu", "gpu"):
                for k in K_WAIT[device]:
                    eager = pooled_p95(
                        "ma", network_size, model, device, k, 0, "SG", "eager"
                    )
                    if eager["status"] != "valid":
                        raise RuntimeError(
                            f"Missing eager cell: {network_size}, {model}, "
                            f"{device}, K={k}"
                        )
                    for fill, condition in (
                        ("eager", "Eager"),
                        ("w5", "Wait-5"),
                        ("w10", "Wait-10"),
                    ):
                        cell = (
                            eager
                            if fill == "eager"
                            else pooled_p95(
                                "ma",
                                network_size,
                                model,
                                device,
                                k,
                                0,
                                "SG",
                                fill,
                            )
                        )
                        if cell["status"] != "valid":
                            raise RuntimeError(
                                f"Missing {fill} cell: {network_size}, {model}, "
                                f"{device}, K={k}"
                            )
                        rows.append(
                            base_row(
                                "A",
                                model,
                                device,
                                network_size,
                                k,
                                k,
                                0,
                                "SG",
                                "Model-Actor",
                                fill,
                                condition,
                                cell,
                                eager,
                                cell["p95_s"] / eager["p95_s"],
                            )
                        )
    return rows


def panel_b_rows() -> list[dict]:
    rows = []
    for network_size in NETWORKS:
        for model in MODELS:
            for device in ("cpu", "gpu"):
                for k in K_MA[device]:
                    sg = pooled_p95(
                        "ma", network_size, model, device, k, 0, "SG", "eager"
                    )
                    s1 = pooled_p95(
                        "ma", network_size, model, device, k, 0, "S1", "eager"
                    )
                    if sg["status"] != "valid" or s1["status"] != "valid":
                        raise RuntimeError(
                            f"Missing SG/S1 pair: {network_size}, {model}, "
                            f"{device}, K={k}"
                        )
                    rows.append(
                        base_row(
                            "B",
                            model,
                            device,
                            network_size,
                            k,
                            k,
                            0,
                            "SG_vs_S1",
                            "Model-Actor",
                            "eager",
                            "S1 / SG",
                            s1,
                            sg,
                            s1["p95_s"] / sg["p95_s"],
                        )
                    )
    return rows


def panel_c_rows() -> list[dict]:
    rows = []
    for network_size in NETWORKS:
        for model in MODELS:
            for device in ("cpu", "gpu"):
                for k_ma, k_ripper in HYBRID_SPLITS[device]:
                    k_total = k_ma + k_ripper
                    model_actor = pooled_p95(
                        "ma",
                        network_size,
                        model,
                        device,
                        k_total,
                        0,
                        "SG",
                        "eager",
                    )
                    if model_actor["status"] != "valid":
                        raise RuntimeError(
                            f"Missing total-K Model-Actor cell: {network_size}, "
                            f"{model}, {device}, K={k_total}"
                        )
                    rows.append(
                        base_row(
                            "C",
                            model,
                            device,
                            network_size,
                            k_total,
                            k_total,
                            0,
                            "SG",
                            "Model-Actor",
                            "eager",
                            "MA only",
                            model_actor,
                            model_actor,
                            1.0,
                        )
                    )
                    for composition, condition in HYBRIDS:
                        hybrid = pooled_p95(
                            composition,
                            network_size,
                            model,
                            device,
                            k_ma,
                            k_ripper,
                            "SG",
                            "eager",
                        )
                        if hybrid["status"] != "valid":
                            raise RuntimeError(
                                f"Missing hybrid cell: {composition}, "
                                f"{network_size}, {model}, {device}, "
                                f"Kma={k_ma}, Kripper={k_ripper}"
                            )
                        rows.append(
                            base_row(
                                "C",
                                model,
                                device,
                                network_size,
                                k_total,
                                k_ma,
                                k_ripper,
                                "SG",
                                composition,
                                "eager",
                                condition,
                                hybrid,
                                model_actor,
                                hybrid["p95_s"] / model_actor["p95_s"],
                            )
                        )
    return rows


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def describe_row(row: dict) -> str:
    return (
        f"{row['model']}, {row['network_size']} stations, {row['device']}, "
        f"K={row['k_total']}"
    )


def validate(rows: list[dict]) -> str:
    a_rows = [row for row in rows if row["panel"] == "A"]
    triplet_keys = {
        (row["model"], row["device"], row["network_size"], row["k_total"])
        for row in a_rows
    }
    if len(triplet_keys) != 40 or len(a_rows) != 120:
        raise RuntimeError("Panel A did not produce 40 matched policy triplets")
    eager_wins = 0
    for key in triplet_keys:
        matched = [
            row
            for row in a_rows
            if (
                row["model"],
                row["device"],
                row["network_size"],
                row["k_total"],
            )
            == key
        ]
        values = {row["condition"]: float(row["p95_s"]) for row in matched}
        eager_wins += values["Eager"] < min(values["Wait-5"], values["Wait-10"])
    if eager_wins != 40:
        raise RuntimeError(f"Expected 40 Eager wins, found {eager_wins}")

    wait5 = [row for row in a_rows if row["condition"] == "Wait-5"]
    wait10 = [row for row in a_rows if row["condition"] == "Wait-10"]
    eager = [row for row in a_rows if row["condition"] == "Eager"]
    wait5_ratios = [float(row["ratio"]) for row in wait5]
    wait10_ratios = [float(row["ratio"]) for row in wait10]

    b_rows = [row for row in rows if row["panel"] == "B"]
    b_ratios = [float(row["ratio"]) for row in b_rows]
    if len(b_rows) != 90 or sum(value > 1 for value in b_ratios) != 89:
        raise RuntimeError("Panel B did not reproduce 89 wins in 90 comparisons")

    c_rows = [row for row in rows if row["panel"] == "C"]
    lines = [
        "Figure 6 validation",
        "Every p95 is recalculated from pooled station events across five repeats.",
        "",
        f"Panel A matched policy triplets: n={len(triplet_keys)}",
        f"Panel A Eager lowest latency: n={eager_wins}",
        f"Panel A median Wait-5/Eager ratio: "
        f"{statistics.median(wait5_ratios):.12f}",
        f"Panel A median Wait-10/Eager ratio: "
        f"{statistics.median(wait10_ratios):.12f}",
        f"Panel A median Eager p95: "
        f"{statistics.median(float(row['p95_s']) for row in eager):.12f} s",
        f"Panel A median Wait-5 p95: "
        f"{statistics.median(float(row['p95_s']) for row in wait5):.12f} s",
        f"Panel A median Wait-10 p95: "
        f"{statistics.median(float(row['p95_s']) for row in wait10):.12f} s",
    ]
    for name, policy_rows in (("Wait-5", wait5), ("Wait-10", wait10)):
        smallest = min(policy_rows, key=lambda row: float(row["ratio"]))
        largest = max(policy_rows, key=lambda row: float(row["ratio"]))
        lines.extend(
            [
                f"Panel A smallest {name}/Eager ratio: "
                f"{float(smallest['ratio']):.12f} ({describe_row(smallest)})",
                f"Panel A largest {name}/Eager ratio: "
                f"{float(largest['ratio']):.12f} ({describe_row(largest)})",
            ]
        )

    lines.extend(
        [
            "",
            f"Panel B matched SG/S1 comparisons: n={len(b_rows)}",
            f"Panel B SG wins: n={sum(value > 1 for value in b_ratios)}",
            f"Panel B median SG speedup: {statistics.median(b_ratios):.12f}",
        ]
    )
    for device in ("cpu", "gpu"):
        values = [
            float(row["ratio"]) for row in b_rows if row["device"] == device
        ]
        lines.append(
            f"Panel B median SG speedup {device.upper()}: "
            f"{statistics.median(values):.12f}"
        )
    largest_b = max(b_rows, key=lambda row: float(row["ratio"]))
    lines.append(
        f"Panel B largest SG speedup: {float(largest_b['ratio']):.12f} "
        f"({describe_row(largest_b)})"
    )
    example = next(
        row
        for row in b_rows
        if row["model"] == "EQCCT"
        and row["network_size"] == 580
        and row["device"] == "gpu"
        and row["k_total"] == 4
    )
    lines.extend(
        [
            "Panel B 580-station GPU EQCCT K=4:",
            f"  SG p95: {float(example['reference_p95_s']):.12f} s",
            f"  S1 p95: {float(example['p95_s']):.12f} s",
            f"  speedup: {float(example['ratio']):.12f}",
            "",
            "Panel C uses strict total-K matching: "
            "K = K_ModelActor + K_Ripper for each hybrid.",
        ]
    )
    hybrid_summaries = []
    for composition, condition in HYBRIDS:
        hybrid_rows = [
            row for row in c_rows if row["composition"] == composition
        ]
        ratios = [float(row["ratio"]) for row in hybrid_rows]
        p95_values = [float(row["p95_s"]) for row in hybrid_rows]
        hybrid_summaries.append((condition, statistics.median(ratios)))
        lines.extend(
            [
                f"Panel C {condition.replace(chr(10), ' ')} comparisons: "
                f"n={len(hybrid_rows)}",
                f"Panel C {condition.replace(chr(10), ' ')} median ratio: "
                f"{statistics.median(ratios):.12f}",
                f"Panel C {condition.replace(chr(10), ' ')} ratio range: "
                f"{min(ratios):.12f}–{max(ratios):.12f}",
                f"Panel C {condition.replace(chr(10), ' ')} best p95: "
                f"{min(p95_values):.12f} s",
                f"Panel C {condition.replace(chr(10), ' ')} p95 range: "
                f"{min(p95_values):.12f}–{max(p95_values):.12f} s",
            ]
        )
    better = min(hybrid_summaries, key=lambda item: item[1])
    lines.extend(
        [
            f"Panel C lower-median hybrid: {better[0].replace(chr(10), ' ')}",
            "The ≈11× manuscript value is not used because it requires an "
            "unmatched K=20 CPU/K=4 GPU Model-Actor baseline.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_artifacts(rows: list[dict], validation: str) -> tuple[Path, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "figure06_streaming_derived.csv"
    fields = [
        "panel",
        "model",
        "device",
        "network_size",
        "k_total",
        "k_ma",
        "k_ripper",
        "packaging",
        "composition",
        "policy",
        "condition",
        "status",
        "n_station_samples",
        "p95_s",
        "reference_p95_s",
        "ratio",
        "source_file",
        "reference_source",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    validation_path = OUT / "figure06_streaming_validation.txt"
    validation_path.write_text(validation)
    return csv_path, validation_path


def clean_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def plain_log_axis(ax, ticks: list[float]) -> None:
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def header_axis(fig, spec, letter: str, title: str):
    ax = fig.add_subplot(spec)
    ax.axis("off")
    ax.text(
        0.0,
        0.68,
        letter,
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )
    if title:
        ax.text(
            0.065,
            0.68,
            title,
            ha="left",
            va="center",
            fontsize=9.1,
            fontweight="semibold",
            color=INK,
        )
    return ax


def dispatch_axis(ax, rows: list[dict], device: str, ylabel: bool) -> None:
    relevant = [
        row for row in rows if row["panel"] == "A" and row["device"] == device
    ]
    conditions = ["Eager", "Wait-5", "Wait-10"]
    x = np.arange(3)
    keys = sorted(
        {
            (row["model"], row["network_size"], row["k_total"])
            for row in relevant
        }
    )
    for model, network_size, k in keys:
        matched = [
            row
            for row in relevant
            if row["model"] == model
            and row["network_size"] == network_size
            and row["k_total"] == k
        ]
        values = [
            float(next(row for row in matched if row["condition"] == condition)["p95_s"])
            for condition in conditions
        ]
        ax.plot(
            x,
            values,
            color=MODEL_COLORS[model],
            linewidth=0.65,
            alpha=0.30,
            zorder=1,
        )
        ax.scatter(
            x,
            values,
            s=9,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.48,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    for index, condition in enumerate(conditions):
        values = [
            float(row["p95_s"]) for row in relevant if row["condition"] == condition
        ]
        q25, median, q75 = (
            percentile(values, 25),
            statistics.median(values),
            percentile(values, 75),
        )
        ax.errorbar(
            index,
            median,
            yerr=[[median - q25], [q75 - median]],
            fmt="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.0,
            ecolor=INK,
            elinewidth=1.15,
            capsize=2.4,
            zorder=5,
        )
    ax.set_yscale("log")
    ax.set_ylim(0.07, 14)
    plain_log_axis(ax, [0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_title(device.upper(), pad=4)
    if ylabel:
        ax.set_ylabel("p95 finish-minus-ready (s)")
    clean_axis(ax)


def grouping_axis(ax, rows: list[dict], device: str, ylabel: bool) -> None:
    relevant = [
        row for row in rows if row["panel"] == "B" and row["device"] == device
    ]
    rng = np.random.default_rng(52 if device == "cpu" else 53)
    for model_index, model in enumerate(MODELS):
        values = [
            float(row["ratio"]) for row in relevant if row["model"] == model
        ]
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(
            model_index + jitter,
            values,
            s=10,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.48,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        q25, median, q75 = (
            percentile(values, 25),
            statistics.median(values),
            percentile(values, 75),
        )
        ax.errorbar(
            model_index,
            median,
            yerr=[[median - q25], [q75 - median]],
            fmt=MODEL_MARKERS[model],
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=MODEL_COLORS[model],
            markeredgewidth=1.0,
            ecolor=MODEL_COLORS[model],
            elinewidth=1.15,
            capsize=2.2,
            zorder=4,
        )
    ax.axhline(1, color=INK, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.4, 25)
    plain_log_axis(ax, [0.5, 1, 2, 5, 10, 20])
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels([SHORT[model] for model in MODELS])
    ax.set_title(device.upper(), pad=4)
    if ylabel:
        ax.set_ylabel("SG speedup over S1 (×)")
    clean_axis(ax)


def hybrid_axis(ax, rows: list[dict], device: str, ylabel: bool) -> None:
    relevant = [
        row for row in rows if row["panel"] == "C" and row["device"] == device
    ]
    conditions = [
        "MA only",
        "MA initial /\nRipper delayed",
        "Ripper initial /\nMA delayed",
    ]
    x = np.arange(3)
    keys = sorted(
        {
            (
                row["model"],
                row["network_size"],
                row["k_total"],
                row["k_ma"],
                row["k_ripper"],
            )
            for row in relevant
            if row["composition"] != "Model-Actor"
        }
    )
    for model, network_size, k_total, k_ma, k_ripper in keys:
        matched = [
            row
            for row in relevant
            if row["model"] == model
            and row["network_size"] == network_size
            and row["k_total"] == k_total
            and (
                row["composition"] == "Model-Actor"
                or (
                    row["k_ma"] == k_ma
                    and row["k_ripper"] == k_ripper
                )
            )
        ]
        values = [
            float(next(row for row in matched if row["condition"] == condition)["ratio"])
            for condition in conditions
        ]
        ax.plot(
            x,
            values,
            color=MODEL_COLORS[model],
            linewidth=0.65,
            alpha=0.30,
            zorder=1,
        )
        ax.scatter(
            x,
            values,
            s=9,
            marker=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            alpha=0.48,
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
    for index, condition in enumerate(conditions):
        values = [
            float(row["ratio"]) for row in relevant if row["condition"] == condition
        ]
        q25, median, q75 = (
            percentile(values, 25),
            statistics.median(values),
            percentile(values, 75),
        )
        ax.errorbar(
            index,
            median,
            yerr=[[median - q25], [q75 - median]],
            fmt="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgecolor=INK,
            markeredgewidth=1.0,
            ecolor=INK,
            elinewidth=1.15,
            capsize=2.4,
            zorder=5,
        )
    ax.axhline(1, color=INK, linewidth=0.8, linestyle="--")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 35)
    plain_log_axis(ax, [1, 2, 5, 10, 20])
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_title(device.upper(), pad=4)
    if ylabel:
        ax.set_ylabel("p95 relative to Model-Actor (×)")
    clean_axis(ax)


def make_figure(rows: list[dict]) -> plt.Figure:
    figure = plt.figure(figsize=(10.0, 7.3))
    grid = figure.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        top=0.965,
        bottom=0.12,
        wspace=0.22,
        hspace=0.34,
        height_ratios=[1.0, 1.05],
    )

    panel_a = grid[0, 0].subgridspec(
        2, 2, height_ratios=[0.12, 1], hspace=0.05, wspace=0.22
    )
    header_axis(figure, panel_a[0, :], "A", "")
    dispatch_axis(figure.add_subplot(panel_a[1, 0]), rows, "cpu", True)
    dispatch_axis(figure.add_subplot(panel_a[1, 1]), rows, "gpu", False)

    panel_b = grid[0, 1].subgridspec(
        2, 2, height_ratios=[0.12, 1], hspace=0.05, wspace=0.22
    )
    header_axis(figure, panel_b[0, :], "B", "")
    grouping_axis(figure.add_subplot(panel_b[1, 0]), rows, "cpu", True)
    grouping_axis(figure.add_subplot(panel_b[1, 1]), rows, "gpu", False)

    panel_c = grid[1, :].subgridspec(
        2, 2, height_ratios=[0.12, 1], hspace=0.05, wspace=0.16
    )
    header_axis(figure, panel_c[0, :], "C", "")
    hybrid_axis(figure.add_subplot(panel_c[1, 0]), rows, "cpu", True)
    hybrid_axis(figure.add_subplot(panel_c[1, 1]), rows, "gpu", False)
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model],
                linestyle="none",
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=MODEL_COLORS[model],
                markersize=5.0,
                label=SHORT[model],
            )
            for model in MODELS
        ],
        loc="lower center",
        bbox_to_anchor=(0.52, 0.018),
        ncol=5,
        columnspacing=1.45,
        handletextpad=0.4,
        frameon=False,
        fontsize=7.2,
    )
    return figure


def main() -> int:
    configure_style()
    rows = panel_a_rows() + panel_b_rows() + panel_c_rows()
    validation = validate(rows)
    csv_path, validation_path = write_artifacts(rows, validation)
    print(validation, end="")

    figure = make_figure(rows)
    base = OUT / "figure06_streaming_performance"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    svg_path = base.with_suffix(".svg")
    final_png_path = OUT / "figure06_streaming_performance_final.png"
    no_titles_png_path = OUT / "figure06_streaming_performance_no_titles.png"
    figure.savefig(png_path, dpi=600)
    figure.savefig(pdf_path)
    figure.savefig(svg_path)
    figure.savefig(final_png_path, dpi=600)
    figure.savefig(no_titles_png_path, dpi=600)
    plt.close(figure)

    print(csv_path)
    print(validation_path)
    print(png_path)
    print(pdf_path)
    print(svg_path)
    print(final_png_path)
    print(no_titles_png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
