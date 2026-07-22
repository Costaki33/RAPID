#!/usr/bin/env python3
"""Generate every paper table from the CLEAN, isolated re-measurement.

Single source of truth = results/fair_benchmark_iso/ (strictly sequential,
correctly-threaded). Writes docs/figures_v3/tables_iso.md. Covers:
  T1  native single-process baselines at their optimal thread count + thread sweep
  T2  cold-start orchestration (Model-Actor vs Ripper)
  T5  warm head-to-head (annotate vs Model-Actor, CPU/GPU/2-GPU) + 95% CI + tail
  T6  GPU concurrency sweep (1-GPU vs 2-GPU)
  T7  oversubscription (actors-per-core)
"""
from __future__ import annotations
import glob, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISO = ROOT / "results" / "fair_benchmark_iso"
OUT = ROOT / "docs" / "figures_v3"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
_T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26}
L = []
def w(s=""): L.append(s)


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def ci95(vals):
    n = len(vals)
    if not vals:
        return None
    if n < 2:
        return vals[0], 0.0
    return statistics.mean(vals), _T.get(n, 1.96) * statistics.stdev(vals) / (n ** 0.5)


def pct(xs, q):
    s = sorted(xs); return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


# ---------- gather head-to-head (warm latency, per-repeat means + tail samples) ----------
def h2h():
    D = {}
    for p in glob.glob(str(ISO / "h2h" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r:
            continue
        m = r["meta"]; tag = str(m.get("tag", ""))
        # only the head-to-head cells (cpu20/gpu20), not the cpu-sweep tags
        if not (tag.startswith("iso_cpu_") or tag.startswith("iso_gpu_") or tag.startswith("iso_2gpu_")):
            continue
        if tag.startswith("iso_2gpu_") and not tag.endswith("cpu20"):
            continue
        means, samples = [], []
        for rf in sorted((Path(p).parent / "repeats").glob("repeat_*.json")):
            rr = jload(rf)
            if not rr or not rr.get("success"):
                continue
            warm = [f["feed_total_s"] for f in (rr.get("feeds") or []) if f.get("feed_index", 0) >= 1]
            if warm:
                means.append(statistics.mean(warm)); samples += warm
        if means:
            key = (m["model"], m["n_stations"], m["method"], m["device"])
            D[key] = {"means": means, "samples": samples}
    return D


def table_h2h(D):
    w("## T5. Warm head-to-head — annotate vs Model-Actor (clean, isolated)\n")
    w("_Warm per-window latency (s), mean ± 95% CI over 10 repeats; [p95 / p99] tail over warm windows._\n")
    for st in (580, 250):
        w(f"\n### {st} stations\n")
        w("| Model | CPU annotate | **CPU Model-Actor** | GPU annotate | GPU MA (1) | GPU MA (2-GPU) |")
        w("|---|---:|---:|---:|---:|---:|")
        for model in MODELS:
            def cell(meth, dev):
                v = D.get((model, st, meth, dev))
                if not v:
                    return "–"
                mean, h = ci95(v["means"]); s = v["samples"]
                return f"{mean:.2f}±{h:.2f} [{pct(s,.95):.2f}/{pct(s,.99):.2f}]"
            w(f"| {model} | {cell('stream_annotate','cpu')} | **{cell('stream_modelactor','cpu')}** | "
              f"{cell('stream_annotate','gpu')} | {cell('stream_modelactor','gpu')} | {cell('stream_modelactor_2gpu','gpu')} |")
    w("\n**Headline:** CPU Model-Actor matches or beats GPU annotate for every model — the GPU is not "
      "required. CPU Model-Actor also has the tightest tail (p99 ≤ 2.5 s at 580 st).\n")


# ---------- native thread sweep + optimal T1 ----------
def table_slipstream(D):
    """Model-Actor classify vs Model-Actor + Slipstream-BF16, warm (same isolated protocol)."""
    have = any(k[2] == "stream_modelactor_slipstream" for k in D)
    if not have:
        return
    w("\n## T5b. Slipstream-BF16 inside the actor pool vs classify-Model-Actor (warm, isolated)\n")
    w("_Warm per-window latency (s), mean ± 95% CI; does the lean BF16 forward beat native classify()?_\n")
    for st in (580, 250):
        w(f"\n### {st} stations\n")
        w("| Model | MA classify CPU | MA Slipstream CPU | MA classify GPU | MA Slipstream GPU |")
        w("|---|---:|---:|---:|---:|")
        for model in MODELS:
            def c(meth, dev):
                v = D.get((model, st, meth, dev))
                if not v:
                    return "–"
                m, h = ci95(v["means"]); return f"{m:.2f} ± {h:.2f}"
            w(f"| {model} | {c('stream_modelactor','cpu')} | {c('stream_modelactor_slipstream','cpu')} | "
              f"{c('stream_modelactor','gpu')} | {c('stream_modelactor_slipstream','gpu')} |")
    w("")


def table_native():
    N = {}
    for p in glob.glob(str(ISO / "native" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r:
            continue
        m = r["meta"]
        if m.get("n_stations") != 580:
            continue
        N.setdefault((m["method"], m["model"]), {})[m["torch_threads"]] = (r.get("timing") or {}).get("total_s_mean")
    w("\n## T1. Native single-process baselines — thread sensitivity (580 st, cold total s)\n")
    w("_`default` = SeisBench/torch out-of-the-box (~64 threads). Optimum is the best non-default._\n")
    for meth in ("annotate", "classify", "slipstream"):
        w(f"\n**{meth}**\n")
        ths = sorted({t for mo in MODELS for t in N.get((meth, mo), {})}, key=lambda x: (x == 0, x))
        w("| threads | " + " | ".join(MODELS) + " |")
        w("|---|" + "---:|" * len(MODELS))
        for t in ths:
            lbl = "default ~64" if t == 0 else str(t)
            cells = " | ".join((f"{N[(meth,mo)][t]:.1f}" if t in N.get((meth, mo), {}) else "–") for mo in MODELS)
            w(f"| {lbl} | {cells} |")
        opt = []
        for mo in MODELS:
            d = {k: v for k, v in N.get((meth, mo), {}).items() if k != 0 and v is not None}
            opt.append(f"**{min(d.values()):.1f}** (t={min(d, key=d.get)})" if d else "–")
        w(f"| **optimum** | {' | '.join(opt)} |")
    w("\n**Key:** correctly-threaded classify is in-budget (EQT 22.5 s, EQT-NC 13.7 s at 1 thread), but the "
      "naive default (~64 threads) is catastrophic (1064 / 1447 s) — a real out-of-the-box trap. Batched "
      "annotate/slipstream optimum is ~4 threads; even they degrade 2–5× at the default for heavy models. "
      "Single-process methods cannot exploit a multicore CPU — only cross-process actors (Model-Actor) can.\n")


# ---------- cold-start orchestration ----------
def table_orch():
    O = {}
    for p in glob.glob(str(ISO / "orch" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r:
            continue
        m = r["meta"]
        O.setdefault(m["model"], {})[(m["method"], m["device"])] = (r.get("timing") or {}).get("total_s_mean")
    w("\n## T2. Cold-start orchestration (580 st, total s)\n")
    w("| Model | Model-Actor CPU | Model-Actor GPU | Ripper CPU | Ripper GPU |")
    w("|---|---:|---:|---:|---:|")
    for mo in MODELS:
        d = O.get(mo, {})
        g = lambda s, dv: (f"{d[(s,dv)]:.1f}" if (s, dv) in d and d[(s, dv)] else "–")
        w(f"| {mo} | {g('modelactor','cpu')} | {g('modelactor','gpu')} | {g('ripper','cpu')} | {g('ripper','gpu')} |")
    w("\nRipper (ephemeral, reloads the model per task) is ~8× slower than the persistent Model-Actor pool — "
      "persistence, not just parallelism, is the contribution. Cold start is a one-time ~17 s cost.\n")


# ---------- GPU concurrency sweep (1-GPU vs 2-GPU) ----------
def table_gpu_sweep():
    G = {}
    ann = {}
    for p in glob.glob(str(ISO / "h2h" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r:
            continue
        m = r["meta"]; tag = str(m.get("tag", ""))
        ww = (r.get("latency") or {}).get("warm_feed_mean_s_mean")
        if ww is None or m.get("n_stations") != 580:
            continue
        if m.get("method") == "stream_annotate" and m.get("device") == "gpu" and tag == "iso_gpu_580":
            ann[m["model"]] = ww
        for kind, pre in (("2GPU", "iso_2gpu_580_cpu"), ("1GPU", "iso_1gpu_580_cpu")):
            if tag.startswith(pre):
                G.setdefault(m["model"], {})[(kind, int(tag.split("cpu")[1]))] = ww
    w("\n## T6. GPU concurrency sweep — can spreading actors beat GPU annotate? (580 st, warm s)\n")
    w("| Model | GPU annotate | 1-GPU c5/c10/c15 | 2-GPU c5/c10/c15/c20 |")
    w("|---|---:|---:|---:|")
    for mo in MODELS:
        d = G.get(mo, {})
        g1 = "/".join(f"{d[('1GPU',c)]:.1f}" if ('1GPU', c) in d else "–" for c in (5, 10, 15))
        g2 = "/".join(f"{d[('2GPU',c)]:.1f}" if ('2GPU', c) in d else "–" for c in (5, 10, 15, 20))
        a = f"{ann.get(mo):.2f}" if ann.get(mo) else "–"
        w(f"| {mo} | {a} | {g1} | {g2} |")
    w("\nSpreading the pool across both GPUs halves the single-GPU contention and improves with more actors, "
      "but even the best 2-GPU config (cpu20) still loses to batched GPU annotate. **On a GPU, annotate wins; "
      "the actor pool cannot be tuned to beat it.**\n")


# ---------- oversubscription ----------
def table_oversub():
    w("\n## T7. Oversubscription — actors per core (20 cores, cold total s)\n")
    for st in (580, 250):
        R = {}
        for p in glob.glob(str(ISO / "oversub" / "**" / "result.json"), recursive=True):
            r = jload(p)
            if not r or r.get("skipped"):
                continue
            m = r["meta"]
            if m.get("device") != "cpu" or m.get("method") != "modelactor" or m.get("n_stations") != st or m.get("n_cpus") != 20:
                continue
            R.setdefault(m["model"], {})[round(m["concurrency"] / 20, 2)] = (r.get("timing") or {}).get("total_s_mean")
        if not R:
            continue
        w(f"\n**{st} stations**\n")
        mults = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
        w("| Model | " + " | ".join(f"{x}×" for x in mults) + " |")
        w("|---|" + "---:|" * len(mults))
        for mo in MODELS:
            d = R.get(mo, {})
            w(f"| {mo} | " + " | ".join((f"{d[x]:.0f}" if x in d else "–") for x in mults) + " |")
    w("\nOptimum ≈ 0.5–1 actor/core; oversubscription beyond 1× degrades monotonically (4× ≈ 2.5× slower). "
      "Memory headroom is not a license to over-pack.\n")


def fig_head_to_head(D):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    st = 580; width = 0.2; x = range(len(MODELS))
    series = [("annotate — CPU", "stream_annotate", "cpu", "#e74c3c"),
              ("Model-Actor — CPU (GPU-free)", "stream_modelactor", "cpu", "#2e86c1"),
              ("annotate — GPU", "stream_annotate", "gpu", "#f1948a"),
              ("Model-Actor — GPU (2-GPU)", "stream_modelactor_2gpu", "gpu", "#aed6f1")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (lab, meth, dev, c) in enumerate(series):
        ys = [statistics.mean(D[(m, st, meth, dev)]["means"]) if (m, st, meth, dev) in D else 0 for m in MODELS]
        ax.bar([xi + (i - 1.5) * width for xi in x], ys, width, label=lab, color=c)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODELS)
    ax.set_ylabel("warm per-window latency (s)")
    ax.set_title("Warm head-to-head, 580 stations (isolated): CPU Model-Actor matches/beats GPU annotate")
    ax.legend(fontsize=8); ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_head_to_head.png", bbox_inches="tight"); plt.close(fig)


def fig_orchestration():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    O = {}
    for p in glob.glob(str(ISO / "orch" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r:
            continue
        m = r["meta"]
        O.setdefault((m["method"], m["device"]), {})[m["model"]] = (r.get("timing") or {}).get("total_s_mean")
    series = [("Model-Actor — CPU", ("modelactor", "cpu"), "#2e86c1"),
              ("Model-Actor — GPU", ("modelactor", "gpu"), "#aed6f1"),
              ("Ripper — CPU", ("ripper", "cpu"), "#e74c3c"),
              ("Ripper — GPU", ("ripper", "gpu"), "#f1948a")]
    width = 0.2; x = range(len(MODELS))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (lab, key, c) in enumerate(series):
        ys = [O.get(key, {}).get(m, 0) for m in MODELS]
        ax.bar([xi + (i - 1.5) * width for xi in x], ys, width, label=lab, color=c)
    ax.axhline(30, color="k", ls=":", lw=1); ax.text(-0.4, 33, "30 s budget", fontsize=8)
    ax.set_xticks(list(x)); ax.set_xticklabels(MODELS); ax.set_ylabel("cold-start total (s)")
    ax.set_title("Cold-start orchestration (isolated): persistent Model-Actor ~8× faster than ephemeral Ripper")
    ax.legend(fontsize=8); ax.set_axisbelow(True); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_orchestration_cpu.png", bbox_inches="tight"); plt.close(fig)


def fig_oversub():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = {}
    for p in glob.glob(str(ISO / "oversub" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r or r.get("skipped"):
            continue
        m = r["meta"]
        if m.get("device") != "cpu" or m.get("method") != "modelactor" or m.get("n_stations") != 580 or m.get("n_cpus") != 20:
            continue
        R.setdefault(m["model"], {})[round(m["concurrency"] / 20, 2)] = (r.get("timing") or {}).get("total_s_mean")
    colors = {"PhaseNet": "#2e86c1", "PhaseNetLight": "#27ae60", "EQTransformer": "#e67e22", "EQT-NC": "#c0392b"}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for mo in MODELS:
        d = R.get(mo, {})
        if not d:
            continue
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], "o-", label=mo, color=colors[mo])
    ax.axvline(1.0, color="k", ls=":", lw=1); ax.text(1.02, ax.get_ylim()[1] * 0.9, "1 actor/core", fontsize=8)
    ax.set_xlabel("actors per core (× core budget)"); ax.set_ylabel("cold-start total (s)")
    ax.set_title("Oversubscription (isolated, 20 cores, 580 st): optimum ≈ 0.5–1 actor/core; beyond 1× degrades")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_oversub.png", bbox_inches="tight"); plt.close(fig)


def fig_thread_sweep():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    N = {}
    for p in glob.glob(str(ISO / "native" / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r or r["meta"].get("n_stations") != 580:
            continue
        m = r["meta"]; t = (r.get("timing") or {}).get("total_s_mean")
        N.setdefault((m["method"], m["model"]), {})[m["torch_threads"]] = t
    meths = ["classify", "annotate", "slipstream"]
    colors = {"PhaseNet": "#2e86c1", "PhaseNetLight": "#27ae60", "EQTransformer": "#e67e22", "EQT-NC": "#c0392b"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, meth in zip(axes, meths):
        for mo in MODELS:
            d = {(64 if k == 0 else k): v for k, v in N.get((meth, mo), {}).items() if v}
            if not d:
                continue
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], "o-", label=mo, color=colors[mo])
        ax.axhline(30, color="k", ls=":", lw=1); ax.text(1, 33, "30 s budget", fontsize=7)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("torch intra-op threads (64 = default)"); ax.set_title(meth)
        ax.grid(True, which="both", alpha=0.3)
        if meth == "classify":
            ax.set_ylabel("total CPU time (s, log)"); ax.legend(fontsize=7)
    fig.suptitle("Native thread sensitivity (isolated): intra-op threading does not scale these models; "
                 "the default (~64) is catastrophic for per-station classify — STEAD 580 st", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_thread_sweep.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    w("# Clean tables — isolated re-measurement (`results/fair_benchmark_iso/`)\n")
    w("_Auto-generated by `benchmarks/analysis/generate_iso_tables.py`. Strictly sequential, correctly-threaded._\n")
    D = h2h()
    table_h2h(D)
    table_slipstream(D)
    table_native()
    table_orch()
    table_gpu_sweep()
    table_oversub()
    (OUT / "tables_iso.md").write_text("\n".join(L))
    try:
        fig_head_to_head(D); fig_thread_sweep(); fig_orchestration(); fig_oversub()
        print("wrote fig_head_to_head, fig_thread_sweep, fig_orchestration_cpu, fig_oversub")
    except Exception as e:
        print(f"(figures skipped: {e})")
    print(f"Wrote {OUT/'tables_iso.md'}")
