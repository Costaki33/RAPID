#!/usr/bin/env python3
"""Generate figures + tables for the v2 RAPID paper draft from fair-benchmark results.

Reads every result.json under results/fair_benchmark and writes:
  docs/figures_v2/*.png            -- publication figures
  docs/figures_v2/tables_generated.md -- markdown tables to paste into the draft

Re-run any time; assets always reflect the current state of the benchmark.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "fair_benchmark"
OUT = ROOT / "docs" / "figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

STAGES = ["framework_init", "model_load", "waveform_access", "preprocess",
          "warmup", "inference", "pick_generation"]
STAGE_LABELS = ["framework init", "model load", "waveform access", "preprocess",
                "warmup", "inference", "pick generation"]
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
CPU_GRID = [5, 8, 11, 14, 17, 20]
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def canonical_regime(model: str) -> str:
    return "w6000" if model in ("EQTransformer", "EQT-NC") else "w6000ov03"


def regime_of(tag: str) -> str:
    for r in ("w6000ov03", "w6000x2", "w3001"):
        if r in tag:
            return r
    return "w6000"


def load_rows():
    rows = []
    for p in glob.glob(str(RES / "**" / "result.json"), recursive=True):
        try:
            r = json.loads(Path(p).read_text())
        except Exception:
            continue
        m = r.get("meta", {})
        t = r.get("timing", {})
        reps = [x for x in t.get("repeats", []) if x.get("success")]
        if not reps:
            continue
        bucket = "oversub" if "/oversub/" in p else ("streaming" if "/streaming/" in p else "matrix")
        row = dict(
            family=m.get("family"), method=m.get("method"), model=m.get("model"),
            ds=m.get("dataset"), st=m.get("n_stations"), dev=m.get("device"),
            ncpu=m.get("n_cpus"), dtype=m.get("dtype"), comp=bool(m.get("compile")),
            bs=m.get("batch_size"), tag=m.get("tag", ""), regime=regime_of(m.get("tag", "")),
            bucket=bucket, nrep=len(reps),
            total=sum(x["total_s"] for x in reps) / len(reps),
            total_std=t.get("total_s_std") or 0.0,
        )
        for s in STAGES:
            row[s] = t.get(f"{s}_s_mean") or 0.0
        mem = r.get("memory") or {}
        row["peak_pss"] = mem.get("peak_pss_mb_mean")
        row["peak_rss"] = mem.get("peak_ram_mb_mean")
        pq = r.get("pick_quality_vs_catalog") or {}
        for ph in ("P", "S"):
            for k in ("precision", "recall", "f1", "n_catalog", "n_detected", "matched",
                      "missing", "additional", "duplicated", "mean_dt", "median_dt",
                      "std_dt", "p95_abs_dt", "p99_abs_dt", "frac_within_1",
                      "frac_within_5", "frac_within_10"):
                row[f"{ph}.{k}"] = pq.get(f"{ph}.{k}_mean")
                row[f"{ph}.{k}_std"] = pq.get(f"{ph}.{k}_std")
        lat = r.get("latency") or {}
        row["cold_s"] = lat.get("cold_feed_total_s_mean")
        row["warm_s"] = lat.get("warm_feed_mean_s_mean")
        row["warm_std"] = lat.get("warm_feed_mean_s_std")
        row["n_modelactors"] = (reps[0] or {}).get("n_modelactors")
        # Per-repeat samples for distribution plots.
        row["_totals"] = [x["total_s"] for x in reps]
        row["_pq_reps"] = [x.get("P", {}) for x in (pq.get("repeats") or []) if isinstance(x, dict)]
        rows.append(row)
    return rows


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def best(rs, key="total"):
    rs = [r for r in rs if r.get(key) is not None]
    return min(rs, key=lambda r: r[key]) if rs else None


def prec_label(r):
    return f"{r['dtype']}{'+compile' if r['comp'] else ''}"


# ---------------------------------------------------------------------------
ROWS = load_rows()
print(f"loaded {len(ROWS)} successful trials")
TBL = open(OUT / "tables_generated.md", "w")
TBL.write("# Generated tables (auto; re-run scripts/generate_paper_v2_assets.py)\n\n")


# ---- Figure 1: native CPU scaling -----------------------------------------
def fig_native_cpu():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for ax, model in zip(axes.flat, MODELS):
        reg = canonical_regime(model)
        for label, flt, style in [
            ("annotate (FP32)", lambda r: r["method"] == "annotate", "o-"),
            ("classify (FP32)", lambda r: r["method"] == "classify", "s-"),
            ("slipstream FP32", lambda r: r["method"] == "slipstream" and r["dtype"] == "fp32", "^-"),
            ("slipstream BF16", lambda r: r["method"] == "slipstream" and r["dtype"] == "bf16" and not r["comp"], "v-"),
            ("slipstream FP16", lambda r: r["method"] == "slipstream" and r["dtype"] == "fp16" and not r["comp"], "d-"),
        ]:
            xs, ys = [], []
            for c in CPU_GRID:
                cand = [r for r in sel(ROWS, family="native", model=model, ds="stead",
                                       st=580, dev="cpu", ncpu=c)
                        if r["regime"] == reg and flt(r)]
                b = best(cand)
                if b:
                    xs.append(c); ys.append(b["total"])
            if xs:
                ax.plot(xs, ys, style, label=label, ms=4)
        ax.set_yscale("log")
        ax.set_title(model)
    for ax in axes[1]:
        ax.set_xlabel("CPU cores")
    for ax in axes[:, 0]:
        ax.set_ylabel("total wall time (s)")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Native single-process methods, STEAD 580 stations, CPU only "
                 "(cold start incl. model load + warmup)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_native_cpu_scaling.png", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 2: measured stage breakdown ------------------------------------
def fig_stage_breakdown():
    configs = []
    for label, flt in [
        ("annotate\nFP32", dict(family="native", method="annotate")),
        ("classify\nFP32", dict(family="native", method="classify")),
        ("slipstream\nFP32", dict(family="native", method="slipstream", dtype="fp32")),
        ("ripper\n(classify)", dict(family="orchestration", method="ripper")),
        ("modelactor\n(classify)", dict(family="orchestration", method="modelactor")),
        ("ripper+\nslipstream", dict(family="orchestration", method="ripper_slipstream", dtype="fp32")),
        ("modelactor+\nslipstream", dict(family="orchestration", method="modelactor_slipstream", dtype="fp32")),
    ]:
        cand = [r for r in sel(ROWS, model="PhaseNet", ds="stead", st=580, dev="cpu", ncpu=20, **flt)
                if r["regime"] == canonical_regime("PhaseNet") and not r["comp"]]
        b = best(cand)
        if b:
            configs.append((label, b))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bottoms = [0.0] * len(configs)
    colors = plt.cm.tab20.colors
    for si, (s, sl) in enumerate(zip(STAGES, STAGE_LABELS)):
        vals = [c[1][s] for c in configs]
        ax.bar([c[0] for c in configs], vals, bottom=bottoms, label=sl, color=colors[si * 2 % 20])
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for i, (_, r) in enumerate(configs):
        ax.text(i, bottoms[i] + 0.4, f"{r['total']:.1f}s", ha="center", fontsize=8)
    ax.set_ylabel("mean wall time per run (s)")
    ax.set_title("Measured stage decomposition: PhaseNet, STEAD 580 stations, 20 CPU cores\n"
                 "(orchestration pipelined stages normalized by measured busy-time shares)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage_breakdown.png", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 3: orchestration strategy comparison ---------------------------
def fig_orchestration():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    strategies = [
        ("classify (native)", dict(family="native", method="classify")),
        ("ripper", dict(family="orchestration", method="ripper")),
        ("modelactor", dict(family="orchestration", method="modelactor")),
        ("ripper+slip FP32", dict(family="orchestration", method="ripper_slipstream", dtype="fp32")),
        ("modelactor+slip FP32", dict(family="orchestration", method="modelactor_slipstream", dtype="fp32")),
        ("modelactor+slip best", dict(family="orchestration", method="modelactor_slipstream")),
    ]
    for ax, st in zip(axes, (250, 580)):
        width = 0.13
        x = range(len(MODELS))
        for i, (label, flt) in enumerate(strategies):
            ys = []
            for model in MODELS:
                cand = [r for r in sel(ROWS, model=model, ds="stead", st=st, dev="cpu", ncpu=20, **flt)
                        if r["regime"] == canonical_regime(model)]
                b = best(cand)
                ys.append(b["total"] if b else 0.0)
            ax.bar([xi + (i - 2.5) * width for xi in x], ys, width, label=label)
        ax.axhline(30, color="red", ls="--", lw=1, label="30 s real-time target" if st == 250 else None)
        ax.set_xticks(list(x)); ax.set_xticklabels(MODELS, fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"STEAD, {st} stations, 20 CPU cores")
        ax.set_ylabel("total wall time (s)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Orchestration strategies vs native classify (cold start, CPU)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_orchestration_cpu.png", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 4: pick quality by precision -----------------------------------
def fig_quality():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    modes = [("annotate", dict(method="annotate")),
             ("classify", dict(method="classify")),
             ("slip FP32", dict(method="slipstream", dtype="fp32")),
             ("slip FP16", dict(method="slipstream", dtype="fp16")),
             ("slip BF16", dict(method="slipstream", dtype="bf16"))]
    for ax, met in zip(axes, ("P.f1", "P.recall")):
        for mi, model in enumerate(MODELS):
            xs, ys, es = [], [], []
            for i, (label, flt) in enumerate(modes):
                cand = [r for r in sel(ROWS, family="native", model=model, ds="stead", st=580,
                                       dev="cpu", **flt)
                        if r["regime"] == canonical_regime(model) and not r["comp"]
                        and r.get(met) is not None]
                if cand:
                    vals = [r[met] for r in cand]
                    xs.append(i); ys.append(sum(vals) / len(vals))
                    es.append((max(vals) - min(vals)) / 2)
            if xs:
                ax.errorbar([x + mi * 0.08 - 0.12 for x in xs], ys, yerr=es, fmt="o-",
                            ms=4, capsize=2, label=model)
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m[0] for m in modes], fontsize=8)
        ax.set_ylabel(f"{'P-wave F1' if met == 'P.f1' else 'P-wave recall'} vs catalog")
        ax.set_ylim(0.5, 1.02)
    axes[0].legend(fontsize=7)
    fig.suptitle("Pick quality by method and precision (native, STEAD 580 st, CPU; "
                 "range across core budgets)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_pick_quality_precision.png", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 5: memory (PSS) -------------------------------------------------
def fig_memory():
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    cats = [("annotate", dict(family="native", method="annotate")),
            ("classify", dict(family="native", method="classify")),
            ("slipstream", dict(family="native", method="slipstream", dtype="fp32")),
            ("ripper+slip", dict(family="orchestration", method="ripper_slipstream", dtype="fp32")),
            ("modelactor+slip", dict(family="orchestration", method="modelactor_slipstream", dtype="fp32"))]
    width = 0.16
    for i, (label, flt) in enumerate(cats):
        ys = []
        for model in MODELS:
            cand = [r for r in sel(ROWS, model=model, ds="stead", st=580, dev="cpu", ncpu=20, **flt)
                    if r["regime"] == canonical_regime(model) and r.get("peak_pss")]
            b = best(cand)
            ys.append((b["peak_pss"] / 1000.0) if b else 0.0)
        ax.bar([xi + (i - 2) * width for xi in range(len(MODELS))], ys, width, label=label)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=8)
    ax.set_ylabel("peak process-tree PSS (GB)")
    ax.set_title("Peak memory (PSS; shared pages counted once) — STEAD 580 st, 20 CPU cores")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_memory_pss.png", bbox_inches="tight")
    plt.close(fig)


# ---- Figure 6: warm-actor latency (streaming family) ------------------------
def fig_latency():
    rows = [r for r in ROWS if r["family"] == "streaming" and r.get("cold_s")]
    if not rows:
        print("no latency rows yet; skipping fig_latency")
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels, colds, warms = [], [], []
    for model in MODELS:
        for meth, mlab in (("stream_modelactor", "classify"), ("stream_modelactor_slipstream", "slip FP32")):
            cand = [r for r in rows if r["model"] == model and r["method"] == meth
                    and r["ds"] == "stead" and r["st"] == 250 and r["dtype"] in (None, "fp32")
                    and not r["comp"]]
            b = best(cand, key="warm_s")
            if b:
                labels.append(f"{model}\n{mlab}")
                colds.append(b["cold_s"]); warms.append(b["warm_s"])
    x = range(len(labels))
    ax.bar([xi - 0.18 for xi in x], colds, 0.36, label="cold feed (incl. pool startup)")
    ax.bar([xi + 0.18 for xi in x], warms, 0.36, label="warm feed (steady state)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("per-feed wall time (s)")
    ax.set_title("Warm persistent-actor latency: cold vs steady-state feeds "
                 "(back-to-back, STEAD 250 st, CPU)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency_cold_warm.png", bbox_inches="tight")
    plt.close(fig)


# ---- Distribution figures ---------------------------------------------------
DIST_METHODS = [
    ("annotate", dict(family="native", method="annotate"), None),
    ("classify", dict(family="native", method="classify"), None),
    ("slip FP32", dict(family="native", method="slipstream", dtype="fp32"), None),
    ("slip FP16", dict(family="native", method="slipstream", dtype="fp16"), None),
    ("slip BF16", dict(family="native", method="slipstream", dtype="bf16"), None),
    ("ripper", dict(family="orchestration", method="ripper"), None),
    ("MA", dict(family="orchestration", method="modelactor"), None),
    ("ripper+slip", dict(family="orchestration", method="ripper_slipstream", dtype="fp32"), None),
    ("MA+slip", dict(family="orchestration", method="modelactor_slipstream", dtype="fp32"), None),
]


def _dist_pool(model, ds, flt, value):
    """Pool per-repeat samples over core budgets and batch sizes (CPU, 580 st)."""
    out = []
    for r in sel(ROWS, model=model, ds=ds, st=580, dev="cpu", **flt):
        if r["regime"] != canonical_regime(model) or r["comp"]:
            continue
        if value == "total":
            out.extend(r["_totals"])
        else:
            out.extend(p[value] for p in r["_pq_reps"] if p.get(value) is not None)
    return out


def fig_timing_dist():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    for ax, model in zip(axes.flat, MODELS):
        data, labels = [], []
        for label, flt, _ in DIST_METHODS:
            vals = _dist_pool(model, "stead", flt, "total")
            if vals:
                data.append(vals); labels.append(f"{label}\n(n={len(vals)})")
        if not data:
            continue
        bp = ax.boxplot(data, tick_labels=labels, showfliers=True,
                        flierprops=dict(ms=2, alpha=0.4), patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#7fb3d5")
        ax.set_yscale("log")
        ax.axhline(30, color="red", ls="--", lw=1)
        ax.set_title(model)
        ax.set_ylabel("total wall time (s)")
        ax.tick_params(axis="x", labelsize=6)
    fig.suptitle("Total wall-time distributions per method — per-repeat samples pooled over\n"
                 "core budgets and batch sizes (STEAD, 580 st, CPU; red line = 30 s target)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_timing_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_f1_dist():
    import random

    rng = random.Random(0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    q_methods = [m for m in DIST_METHODS if m[0] not in ("ripper", "MA")]
    for ax, model in zip(axes.flat, MODELS):
        pos, labels = [], []
        for i, (label, flt, _) in enumerate(q_methods):
            for j, (ds, color) in enumerate((("stead", "#2e86c1"), ("txed", "#e67e22"))):
                vals = _dist_pool(model, ds, flt, "f1")
                if vals:
                    xs = [i + (j - 0.5) * 0.34 + rng.uniform(-0.10, 0.10) for _ in vals]
                    ax.scatter(xs, vals, s=7, alpha=0.35, color=color, edgecolors="none")
                    med = sorted(vals)[len(vals) // 2]
                    ax.hlines(med, i + (j - 0.5) * 0.34 - 0.14, i + (j - 0.5) * 0.34 + 0.14,
                              color=color, lw=1.6)
            pos.append(i); labels.append(label)
        ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=6.5)
        ax.set_title(model)
        ax.set_ylabel("P-wave F1 vs catalog")
        ax.set_ylim(0.3, 1.02)
    import matplotlib.patches as mpatches
    fig.legend(handles=[mpatches.Patch(color="#2e86c1", label="STEAD"),
                        mpatches.Patch(color="#e67e22", label="TXED")],
               loc="lower center", ncol=2, fontsize=9)
    fig.suptitle("P-wave F1 per repeat (every dot = one run) — pooled over core budgets and\n"
                 "batch sizes, 580 st, CPU. Near-zero vertical spread = config-independent quality.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig_f1_distributions.png", bbox_inches="tight")
    plt.close(fig)


def fig_dt_dist():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    q_methods = [m for m in DIST_METHODS if m[0] not in ("ripper", "MA")]
    for ax, (metric, title) in zip(
        axes,
        (("median_dt", "median ΔT (samples; sign = early/late vs catalog)"),
         ("p95_abs_dt", "P95 |ΔT| (samples)")),
    ):
        data, labels = [], []
        for label, flt, _ in q_methods:
            vals = []
            for model in MODELS:
                vals.extend(_dist_pool(model, "stead", flt, metric))
            if vals:
                data.append(vals); labels.append(f"{label}\n(n={len(vals)})")
        bp = ax.boxplot(data, tick_labels=labels, showfliers=True,
                        flierprops=dict(ms=2, alpha=0.3), patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#a9dfbf")
        if metric == "median_dt":
            ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", labelsize=6.5)
    fig.suptitle("Pick-timing distributions per method — per-repeat samples pooled over models,\n"
                 "core budgets and batch sizes (STEAD, 580 st, CPU; 1 sample = 10 ms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dt_distributions.png", bbox_inches="tight")
    plt.close(fig)


# ---- Tables -----------------------------------------------------------------
def table_completion():
    # Target counts for the main matrix (from the scheduler builder).
    TARGETS = {
        ("matrix", "annotate"): 1536, ("matrix", "classify"): 384,
        ("matrix", "slipstream"): 6912, ("matrix", "ripper"): 384,
        ("matrix", "modelactor"): 384, ("matrix", "ripper_slipstream"): 4224,
        ("matrix", "modelactor_slipstream"): 6912,
        ("streaming", "stream_modelactor"): 96, ("streaming", "stream_modelactor_slipstream"): 432,
        ("oversub", "ripper"): 128, ("oversub", "modelactor"): 128,
        ("oversub", "ripper_slipstream"): 128, ("oversub", "modelactor_slipstream"): 128,
    }
    done = defaultdict(int)
    dims = defaultdict(lambda: dict(dev=set(), ncpu=set(), dtype=set()))
    for r in ROWS:
        k = (r["bucket"], r["method"])
        done[k] += 1
        dims[k]["dev"].add(r["dev"]); dims[k]["ncpu"].add(r["ncpu"])
        if r["dtype"]:
            dims[k]["dtype"].add(r["dtype"])

    def fmt(s):
        return ",".join(str(x) for x in sorted(s)) if s else "–"

    TBL.write("## T0. Trial completion by family and configuration\n\n")
    TBL.write("Main matrix: 8 model-window-regime combos x 2 datasets (STEAD,TXED) x 2 station "
              "counts (250,580) x 12 device points (6 CPU-core budgets + 6 GPU host-core budgets). "
              "Native trials average 5 repeats; orchestration 1.\n\n")
    TBL.write("| Phase | Strategy | Done | Target | % | Devices | CPU grid | Precisions |\n")
    TBL.write("|---|---|---:|---:|---:|---|---|---|\n")
    order = [("matrix", "annotate"), ("matrix", "classify"), ("matrix", "slipstream"),
             ("matrix", "ripper"), ("matrix", "modelactor"),
             ("matrix", "ripper_slipstream"), ("matrix", "modelactor_slipstream"),
             ("streaming", "stream_modelactor"), ("streaming", "stream_modelactor_slipstream"),
             ("oversub", "ripper"), ("oversub", "modelactor"),
             ("oversub", "ripper_slipstream"), ("oversub", "modelactor_slipstream")]
    tot_done = tot_tgt = 0
    phase_label = {"matrix": "Main matrix", "streaming": "Latency sweep", "oversub": "Oversub sweep"}
    for k in order:
        d = done.get(k, 0)
        tgt = TARGETS.get(k, 0)
        tot_done += d; tot_tgt += tgt
        pct = f"{100*d/tgt:.0f}%" if tgt else "–"
        di = dims.get(k, dict(dev=set(), ncpu=set(), dtype=set()))
        TBL.write(f"| {phase_label[k[0]]} | {k[1]} | {d} | {tgt} | {pct} | "
                  f"{fmt(di['dev'])} | {fmt(di['ncpu'])} | {fmt(di['dtype'])} |\n")
    TBL.write(f"| **All phases** | | **{tot_done}** | **{tot_tgt}** | "
              f"**{100*tot_done/tot_tgt:.0f}%** | | | |\n\n")
    TBL.write("Latency-sweep and oversub-sweep targets count only the portions launched so far "
              "(CPU halves ran on the idle pool during the GPU matrix tail; GPU halves chain after).\n\n")


def table_gpu_orchestration():
    TBL.write("## T2b. Orchestration on GPU (mean total s, cold start; STEAD 580 st)\n\n")
    TBL.write("| Model | Ripper | Model-Actor | Ripper+Slip FP32 | MA+Slip FP32 | MA+Slip best |\n")
    TBL.write("|---|---:|---:|---:|---:|---:|\n")
    any_data = False
    for model in MODELS:
        reg = canonical_regime(model)
        cells = []
        for flt in (dict(family="orchestration", method="ripper"),
                    dict(family="orchestration", method="modelactor"),
                    dict(family="orchestration", method="ripper_slipstream", dtype="fp32"),
                    dict(family="orchestration", method="modelactor_slipstream", dtype="fp32"),
                    dict(family="orchestration", method="modelactor_slipstream")):
            cand = [r for r in sel(ROWS, model=model, ds="stead", st=580, dev="gpu", **flt)
                    if r["regime"] == reg and r["bucket"] == "matrix"]
            b = best(cand)
            if b:
                any_data = True
                if flt.get("method") == "modelactor_slipstream" and "dtype" not in flt:
                    cells.append(f"{b['total']:.1f} ({prec_label(b)})")
                else:
                    cells.append(f"{b['total']:.1f}")
            else:
                cells.append("–")
        TBL.write(f"| {model} | " + " | ".join(cells) + " |\n")
    TBL.write("\n" + ("" if any_data else "_(GPU orchestration trials still running.)_\n") +
              "GPU cells: best over host-core budgets. Dashes = still running at asset-build time.\n\n")


def table_native_baselines():
    TBL.write("## T1. Native single-process baselines (mean total s, cold start; STEAD)\n\n")
    TBL.write("| Model | Method | 250 st CPU20 | 580 st CPU20 | 250 st GPU | 580 st GPU |\n")
    TBL.write("|---|---|---:|---:|---:|---:|\n")
    for model in MODELS:
        reg = canonical_regime(model)
        for meth, flt in (("annotate", dict(method="annotate")),
                          ("classify", dict(method="classify")),
                          ("slipstream FP32", dict(method="slipstream", dtype="fp32")),
                          ("slipstream BF16", dict(method="slipstream", dtype="bf16"))):
            cells = []
            for st, dev in ((250, "cpu"), (580, "cpu"), (250, "gpu"), (580, "gpu")):
                kw = dict(family="native", model=model, ds="stead", st=st, dev=dev, **flt)
                if dev == "cpu":
                    kw["ncpu"] = 20
                cand = [r for r in sel(ROWS, **kw) if r["regime"] == reg and not r["comp"]]
                b = best(cand)
                cells.append(f"{b['total']:.1f}" if b else "–")
            TBL.write(f"| {model} | {meth} | " + " | ".join(cells) + " |\n")
    TBL.write("\nGPU cells: best over host-core budgets and batch sizes.\n\n")


def table_orchestration():
    TBL.write("## T2. Orchestration strategies (mean total s, cold start; STEAD, CPU 20 cores)\n\n")
    TBL.write("| Model | classify (native) | Ripper | Model-Actor | Ripper+Slip FP32 | MA+Slip FP32 | MA+Slip best |\n")
    TBL.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for model in MODELS:
        reg = canonical_regime(model)
        cells = []
        for flt in (dict(family="native", method="classify"),
                    dict(family="orchestration", method="ripper"),
                    dict(family="orchestration", method="modelactor"),
                    dict(family="orchestration", method="ripper_slipstream", dtype="fp32"),
                    dict(family="orchestration", method="modelactor_slipstream", dtype="fp32"),
                    dict(family="orchestration", method="modelactor_slipstream")):
            cand = [r for r in sel(ROWS, model=model, ds="stead", st=580, dev="cpu", ncpu=20, **flt)
                    if r["regime"] == reg]
            b = best(cand)
            if b and flt.get("method") == "modelactor_slipstream" and "dtype" not in flt:
                cells.append(f"{b['total']:.1f} ({prec_label(b)})")
            else:
                cells.append(f"{b['total']:.1f}" if b else "–")
        TBL.write(f"| {model} | " + " | ".join(cells) + " |\n")
    TBL.write("\n")


def table_pick_quality():
    for ds in ("stead", "txed"):
        TBL.write(f"## T3{'a' if ds == 'stead' else 'b'}. P-wave pick quality vs catalog ({ds.upper()}, 580 st, "
                  "CPU 20, canonical regime; mean over repeats)\n\n")
        TBL.write("| Model | Method | det | match | miss | extra | dup | Prec | Rec | F1 | "
                  "med ΔT | P95\\|ΔT\\| | ±1 | ±5 | ±10 |\n")
        TBL.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for model in MODELS:
            reg = canonical_regime(model)
            for meth, flt in (("annotate", dict(family="native", method="annotate")),
                              ("classify", dict(family="native", method="classify")),
                              ("slip FP32", dict(family="native", method="slipstream", dtype="fp32")),
                              ("slip FP16", dict(family="native", method="slipstream", dtype="fp16")),
                              ("slip BF16", dict(family="native", method="slipstream", dtype="bf16")),
                              ("MA+slip FP32", dict(family="orchestration", method="modelactor_slipstream", dtype="fp32"))):
                cand = [r for r in sel(ROWS, model=model, ds=ds, st=580, dev="cpu", ncpu=20, **flt)
                        if r["regime"] == reg and not r["comp"] and r.get("P.f1") is not None]
                b = best(cand)
                if not b:
                    continue
                TBL.write(
                    f"| {model} | {meth} | {b['P.n_detected']:.0f} | {b['P.matched']:.0f} | "
                    f"{b['P.missing']:.0f} | {b['P.additional']:.0f} | {b['P.duplicated']:.0f} | "
                    f"{b['P.precision']:.3f} | {b['P.recall']:.3f} | {b['P.f1']:.3f} | "
                    f"{b['P.median_dt']:.0f} | {b['P.p95_abs_dt']:.0f} | "
                    f"{b['P.frac_within_1']:.2f} | {b['P.frac_within_5']:.2f} | {b['P.frac_within_10']:.2f} |\n")
        TBL.write("\nΔT in samples at 100 Hz (1 sample = 10 ms). det/match/miss/extra/dup are counts "
                  "against the catalog; ±N = fraction of matched picks within N samples.\n\n")


def table_latency():
    rows = [r for r in ROWS if r["family"] == "streaming" and r.get("cold_s")]
    if not rows:
        return
    TBL.write("## T4. Warm persistent-actor latency (back-to-back feeds; STEAD, CPU; mean s)\n\n")
    TBL.write("| Model | Strategy | st | cores | cold feed | warm feed | speedup cold→warm |\n")
    TBL.write("|---|---|---:|---:|---:|---:|---:|\n")
    for model in MODELS:
        for meth, mlab in (("stream_modelactor", "MA classify"),
                           ("stream_modelactor_slipstream", "MA slipstream FP32")):
            for st in (250, 580):
                cand = [r for r in rows if r["model"] == model and r["method"] == meth
                        and r["ds"] == "stead" and r["st"] == st
                        and (r["dtype"] in (None, "fp32")) and not r["comp"]]
                b = best(cand, key="warm_s")
                if b:
                    TBL.write(f"| {model} | {mlab} | {st} | {b['ncpu']} | {b['cold_s']:.2f} | "
                              f"{b['warm_s']:.2f} | {b['cold_s']/max(b['warm_s'],1e-9):.1f}x |\n")
    TBL.write("\n")


fig_native_cpu()
fig_stage_breakdown()
fig_orchestration()
fig_quality()
fig_memory()
fig_latency()
fig_timing_dist()
fig_f1_dist()
fig_dt_dist()
table_completion()
table_native_baselines()
table_orchestration()
table_gpu_orchestration()
table_pick_quality()
table_latency()
TBL.close()
print("wrote figures + tables to", OUT)
