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

ROOT = Path(__file__).resolve().parents[1]
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


if __name__ == "__main__":
    w("# Clean tables — isolated re-measurement (`results/fair_benchmark_iso/`)\n")
    w("_Auto-generated by `scripts/generate_iso_tables.py`. Strictly sequential, correctly-threaded._\n")
    table_h2h(h2h())
    table_native()
    table_orch()
    table_gpu_sweep()
    table_oversub()
    (OUT / "tables_iso.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n\nWrote {OUT/'tables_iso.md'}")
