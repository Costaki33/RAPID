#!/usr/bin/env python3
"""Analyze paced soak (--feed-interval-s 60) stream results.

Reports warm latency, late_s, deadline misses, and backlog-style lateness.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

RAPID = Path(__file__).resolve().parents[2]
STREAM = RAPID / "results" / "iso_full_benchmark" / "stream" / "streaming"
OUT = RAPID / "results" / "iso_full_benchmark" / "PACED_SOAK_ANALYSIS.md"

METHODS = [
    ("stream_annotate", "Annotate"),
    ("stream_classify_batched", "NBC"),
    ("stream_modelactor_batched", "MA-NBC"),
]
MODELS = ["PhaseNet", "EQTransformer"]


def analyze_one(path: Path) -> dict | None:
    if not path.is_file():
        return None
    d = json.loads(path.read_text())
    samples = []
    lates = []
    waits = []
    for rep in d.get("timing", {}).get("repeats", []):
        for f in rep.get("feeds", []):
            if f.get("feed_index", 0) == 0:
                continue
            samples.append(float(f["feed_total_s"]))
            lates.append(float(f.get("late_s") or 0.0))
            waits.append(float(f.get("wait_s") or 0.0))
    if not samples:
        return None
    samples_sorted = sorted(samples)

    def pct(p):
        i = min(len(samples_sorted) - 1, max(0, int(round(p * (len(samples_sorted) - 1)))))
        return samples_sorted[i]

    return {
        "n": len(samples),
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": max(samples),
        "gt10": sum(1 for x in samples if x > 10),
        "gt60": sum(1 for x in samples if x > 60),
        "late_mean": statistics.mean(lates) if lates else 0.0,
        "late_max": max(lates) if lates else 0.0,
        "late_nonzero": sum(1 for x in lates if x > 0),
        "warm_meta": (d.get("latency") or {}).get("warm_feed_mean_s_mean"),
    }


def main() -> None:
    lines = [
        "# Paced soak analysis (feed interval = 60 s)\n",
        "Warm feeds only (feed 0 dropped). Lateness is wall-clock behind the "
        "scheduled feed start; nonzero late_s indicates the previous feed overran "
        "the 60 s cadence.\n",
        "| Model | Method | n | mean | median | p95 | p99 | max | >10s | >60s | late>0 | late_max |\n",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    found = 0
    for model in MODELS:
        for method, label in METHODS:
            p = STREAM / method / "stead" / "580st" / model / "soak_cpu_580" / "result.json"
            r = analyze_one(p)
            if not r:
                lines.append(f"| {model} | {label} | – | – | – | – | – | – | – | – | – | – |\n")
                continue
            found += 1
            lines.append(
                f"| {model} | {label} | {r['n']} | {r['mean']:.2f} | {r['median']:.2f} | "
                f"{r['p95']:.2f} | {r['p99']:.2f} | {r['max']:.2f} | {r['gt10']} | {r['gt60']} | "
                f"{r['late_nonzero']} | {r['late_max']:.2f} |\n"
            )
    lines.append(
        "\n*Interpretation:* methods with feed totals well below 60 s and late_max≈0 "
        "keep pace with a one-minute network cadence under this paced simulation. "
        "This is not live telemetry; station streams are still preloaded inventory "
        "windows submitted on a wall-clock schedule.\n"
    )
    OUT.write_text("".join(lines))
    print(f"wrote {OUT} ({found} cells)")


if __name__ == "__main__":
    main()
