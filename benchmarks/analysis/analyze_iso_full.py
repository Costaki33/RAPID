#!/usr/bin/env python3
"""Analyze results/iso_full_benchmark/ and emit a model-readable results report.

Walks every result.json under results/iso_full_benchmark/ (the new grid plus the
consolidated legacy cells that were copied in), extracts the headline metrics
(timing total_s, warm streaming latency, peak PSS memory, pick-quality F1), and
writes RESULTS_ANALYSIS.md with tables + one-line interpretations.

Run:  python3 benchmarks/analysis/analyze_iso_full.py
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results" / "iso_full_benchmark"
OUT = BASE / "RESULTS_ANALYSIS.md"
MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]

L: List[str] = []
def w(s: str = "") -> None:
    L.append(s)


def jload(p: str) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt(x: Optional[float], nd: int = 2) -> str:
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "–"


# ---------------------------------------------------------------------------
# Load every record once
# ---------------------------------------------------------------------------
print("loading result.json files ...")
RECS: List[dict] = []
for fam in ("native", "orch", "oversub", "stream"):
    for p in glob.glob(str(BASE / fam / "**" / "result.json"), recursive=True):
        r = jload(p)
        if not r or "meta" not in r:
            continue
        m = r["meta"]
        tim = r.get("timing") or {}
        mem = r.get("memory") or {}
        pq = r.get("pick_quality_vs_catalog") or {}
        lat = r.get("latency") or {}
        RECS.append({
            "fam": fam,
            "method": m.get("method"),
            "model": m.get("model"),
            "dataset": (m.get("dataset") or "").lower(),
            "nst": m.get("n_stations"),
            "device": m.get("device"),
            "ncpus": m.get("n_cpus"),
            "threads": m.get("torch_threads"),
            "dtype": m.get("dtype"),
            "compile": bool(m.get("compile")),
            "conc": m.get("concurrency"),
            "tag": str(m.get("tag", "")),
            "extractor": m.get("pick_extractor"),
            "total_s": tim.get("total_s_mean"),
            "total_s_std": tim.get("total_s_std"),
            "inf_s": tim.get("inference_s_mean"),
            "pss": mem.get("peak_pss_mb_mean"),
            "vram": mem.get("peak_vram_mb_mean"),
            "pf1": pq.get("P.f1_mean"),
            "sf1": pq.get("S.f1_mean"),
            "pprec": pq.get("P.precision_mean"),
            "prec_recall": pq.get("P.recall_mean"),
            "warm": lat.get("warm_feed_mean_s_mean"),
            "cold": lat.get("cold_feed_total_s_mean"),
            "warm_p95": None,
        })
print(f"loaded {len(RECS)} records")


def sel(**kw) -> List[dict]:
    out = RECS
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


# ---------------------------------------------------------------------------
# 1. Native single-process: thread sensitivity (the multicore trap)
# ---------------------------------------------------------------------------
def sec_native_threads() -> None:
    w("## 1. Native single-process: thread sensitivity and the multicore ceiling\n")
    w("STEAD 580 stations, cold total seconds at selected PyTorch intra-op thread counts. "
      "`default` = SeisBench/torch out-of-the-box (no thread cap, ~64 threads on this host); "
      "`opt` = the fastest thread count measured in the sweep.\n")
    for meth in ("classify", "annotate", "slipstream"):
        rows = [r for r in sel(fam="native", method=meth, dataset="stead", nst=580, device="cpu")
                if r["dtype"] in (None, "fp32")]
        by = defaultdict(dict)  # model -> {threads: total_s}
        for r in rows:
            if r["total_s"] is not None:
                by[r["model"]][r["threads"]] = r["total_s"]
        w(f"\n**{meth}**\n")
        w("| Model | t=1 | t=4 | t=8 | optimum (t) | default (~64) |")
        w("|---|---:|---:|---:|---:|---:|")
        for mo in MODELS:
            d = by.get(mo, {})
            default = d.get(0)
            nd = {k: v for k, v in d.items() if k not in (0, None)}
            if nd:
                opt_t = min(nd, key=nd.get)
                opt = f"{nd[opt_t]:.1f} (t={opt_t})"
            else:
                opt = "–"
            w(f"| {mo} | {fmt(d.get(1),1)} | {fmt(d.get(4),1)} | {fmt(d.get(8),1)} | {opt} | "
              f"{fmt(default,1)} |")
    w("\n*Interpretation:* the three methods have **different** thread profiles. Per-station "
      "`classify` must run at **1 thread** — the default (~64) is catastrophic (286-1447 s) and "
      "no higher count helps, because each call processes a single station. Batched `annotate` "
      "and `slipstream` are fastest at a **few threads** (~4-8: e.g. EQT annotate 5.3 s at t=8 vs "
      "7.6 s at t=1) but fall off a cliff at high thread counts for the heavy EQT models (>27 s at "
      "the default). No method scales past ~8 threads, and total time is independent of the CPU "
      "*core budget* — a single process cannot use a multicore CPU, which is the motivation for "
      "cross-process orchestration. **Note:** the orch/oversub/stream families run their "
      "single-process baselines at 1 thread to isolate the core-budget axis; the per-method "
      "optimum for `annotate`/`slipstream` is a few threads, so their best single-process numbers "
      "are the optima in this table, not the 1-thread column.\n")


# ---------------------------------------------------------------------------
# 2. Native picker comparison: classify vs classify_batched vs annotate
# ---------------------------------------------------------------------------
def sec_native_batched() -> None:
    w("\n## 2. Native picker comparison: per-station vs batched SeisBench\n")
    w("STEAD 580 stations. `classify` is the naive per-station loop; `classify_batched` "
      "is one SeisBench `classify()` call on the merged network stream, so it gets "
      "cross-station batching and SeisBench's own pick aggregation.\n")
    w("| Model | classify CPU t=1 | classify_batched CPU | classify_batched GPU | annotate CPU opt | batched/classify CPU speedup |")
    w("|---|---:|---:|---:|---:|---:|")
    for mo in MODELS:
        def best(method, *, device="cpu", dtype=None):
            rows = [r for r in sel(fam="native", method=method, model=mo, dataset="stead", nst=580, device=device)
                    if r["total_s"] is not None and (dtype is None or r["dtype"] == dtype)]
            return min((r["total_s"] for r in rows), default=None)

        classify = best("classify", dtype="fp32")
        bat_cpu = best("classify_batched", device="cpu", dtype="fp32")
        bat_gpu = best("classify_batched", device="gpu", dtype="fp32")
        annotate = best("annotate", device="cpu")
        speedup = (classify / bat_cpu) if (classify and bat_cpu) else None
        w(f"| {mo} | {fmt(classify,1)} | {fmt(bat_cpu,1)} | {fmt(bat_gpu,1)} | "
          f"{fmt(annotate,1)} | {fmt(speedup,1)+'x' if speedup else '–'} |")
    w("\n*Interpretation:* `classify_batched` is the fairest native SeisBench picker baseline: "
      "it keeps SeisBench's own pick extractor but removes the artificial per-station Python "
      "loop. It is much faster than naive per-station `classify`, and its pick quality should "
      "match `classify` because both use `classify_aggregate`; timing is the real difference. "
      "Batched `annotate` remains the fastest native probability-trace path, but its picks are "
      "generated by RAPID threshold-crossing rather than SeisBench's aggregator.\n")


# ---------------------------------------------------------------------------
# 3. Slipstream precision: timing + pick quality by dtype
# ---------------------------------------------------------------------------
def sec_precision() -> None:
    w("\n## 3. Slipstream precision: speed and pick quality by dtype\n")
    w("STEAD 580 stations, single process. F1 vs catalog (P / S). FP16 is excluded "
      "by design for EQTransformer / EQT-NC (numerically unsafe).\n")
    w("| Model | dtype | compile | total s | P F1 | S F1 |")
    w("|---|---|---|---:|---:|---:|")
    rows = sel(fam="native", method="slipstream", dataset="stead", nst=580, device="cpu")
    key = lambda r: (MODELS.index(r["model"]) if r["model"] in MODELS else 9,
                     {"fp32": 0, "fp16": 1, "bf16": 2}.get(r["dtype"], 9), r["compile"])
    seen = set()
    for r in sorted(rows, key=key):
        k = (r["model"], r["dtype"], r["compile"])
        if k in seen:
            continue
        seen.add(k)
        w(f"| {r['model']} | {r['dtype']} | {'yes' if r['compile'] else 'no'} | "
          f"{fmt(r['total_s'],1)} | {fmt(r['pf1'],3)} | {fmt(r['sf1'],3)} |")
    w("\n*Interpretation:* **BF16 is the safe reduced-precision default** — it holds pick quality "
      "within noise of FP32 for every model (e.g. EQT P/S F1 0.913/0.966 vs 0.915/0.966) and on "
      "this CPU it also cuts cold time for the heavy models substantially (EQT 26.1->7.5 s, "
      "EQT-NC 14.7->7.4 s). **FP16 is not safe in general:** it collapses PhaseNetLight pick "
      "quality (P/S F1 0.64/0.45 vs 0.95/0.89), is only benign for PhaseNet, and is excluded "
      "outright for the EQT family (pooling sentinel overflow). FP16 also gives no CPU speedup "
      "(no native FP16 CPU kernels; PhaseNet FP16 is *slower*, 13.1 vs 4.6 s). `torch.compile` "
      "adds large per-process warmup cost in these cold runs (EQT BF16 +compile 62.8 s), which is "
      "why compile is evaluated only in the warm streaming family where the warmup is amortized.\n")


# ---------------------------------------------------------------------------
# 4. Cold-start orchestration: Model-Actor vs Ripper
# ---------------------------------------------------------------------------
def sec_orch() -> None:
    w("\n## 4. Cold-start orchestration: persistence beats throwaway workers\n")
    w("STEAD 580 stations, cold total seconds (pool spin-up + model load + whole-network pick). "
      "Model-Actor = persistent pool; Ripper = throwaway worker per task.\n")
    w("| Model | Model-Actor CPU | Model-Actor GPU | Ripper CPU | Ripper GPU | MA-CPU speedup vs Ripper-CPU |")
    w("|---|---:|---:|---:|---:|---:|")
    for mo in MODELS:
        def bm(method, device):
            rows = [r for r in sel(fam="orch", method=method, model=mo, dataset="stead", nst=580, device=device)
                    if r["total_s"] is not None and r["dtype"] in (None, "fp32")]
            return min((r["total_s"] for r in rows), default=None)
        mac, mag, rc, rg = bm("modelactor","cpu"), bm("modelactor","gpu"), bm("ripper","cpu"), bm("ripper","gpu")
        sp = (rc / mac) if (rc and mac) else None
        w(f"| {mo} | {fmt(mac,1)} | {fmt(mag,1)} | {fmt(rc,1)} | {fmt(rg,1)} | {fmt(sp,1)+'x' if sp else '–'} |")
    w("\n*Interpretation:* the persistent Model-Actor pool is far faster than Ripper, which reloads "
      "the model in every task — persistence, not just parallelism, is the contribution. (Ripper "
      "cells come from the consolidated control; the new grid drops Ripper.)\n")


# ---------------------------------------------------------------------------
# 5. Oversubscription: actors per core
# ---------------------------------------------------------------------------
def sec_oversub() -> None:
    w("\n## 5. Oversubscription: how many actors per core?\n")
    w("STEAD 580 stations, CPU Model-Actor, 20-core budget, cold total seconds vs actors-per-core "
      "(multiplier of the core budget).\n")
    mults = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0]
    w("| Model | " + " | ".join(f"{x}x" for x in mults) + " | optimum |")
    w("|---|" + "---:|" * (len(mults) + 1))
    for mo in MODELS:
        rows = [r for r in sel(fam="oversub", method="modelactor", model=mo, dataset="stead",
                               nst=580, device="cpu", ncpus=20)
                if r["total_s"] is not None and r["conc"]]
        d = {}
        for r in rows:
            d[round(r["conc"] / 20, 2)] = r["total_s"]
        cells = " | ".join(fmt(d.get(x), 0) if d.get(x) else "–" for x in mults)
        opt = min(d, key=d.get) if d else None
        w(f"| {mo} | {cells} | {str(opt)+'x' if opt is not None else '–'} |")
    w("\n*Interpretation:* throughput is best around 0.5-1 actor per core; pushing past 1x degrades "
      "monotonically as actors contend for cores and memory. Spare RAM is not a reason to over-pack.\n")


# ---------------------------------------------------------------------------
# 6. Warm streaming head-to-head (the deployment number)
# ---------------------------------------------------------------------------
def sec_stream() -> None:
    w("\n## 6. Warm streaming head-to-head: deployed steady-state latency\n")
    w("Warm per-feed latency (s), mean over the warm feeds (feed 0 dropped) across 5 sessions, "
      "20-core budget. Lower is better.\n")
    for nst in (580, 250):
        w(f"\n### {nst} stations\n")
        w("| Model | annotate CPU | Model-Actor CPU | Slipstream-BF16 CPU | annotate GPU | "
          "Model-Actor GPU | 2-GPU Model-Actor |")
        w("|---|---:|---:|---:|---:|---:|---:|")
        for mo in MODELS:
            def warm(method, device, dtype=None):
                rows = [r for r in sel(fam="stream", method=method, model=mo, dataset="stead",
                                       nst=nst, device=device, ncpus=20)
                        if r["warm"] is not None and (dtype is None or r["dtype"] == dtype)]
                return min((r["warm"] for r in rows), default=None)
            w(f"| {mo} | {fmt(warm('stream_annotate','cpu'))} | {fmt(warm('stream_modelactor','cpu'))} | "
              f"{fmt(warm('stream_modelactor_slipstream','cpu','bf16'))} | {fmt(warm('stream_annotate','gpu'))} | "
              f"{fmt(warm('stream_modelactor','gpu'))} | {fmt(warm('stream_modelactor_2gpu','gpu'))} |")
    w("\n*Interpretation:* with a warm pool, CPU Model-Actor is competitive with — and often beats — "
      "GPU annotate, i.e. real-time picking does not require a GPU. Slipstream-BF16 inside the actor "
      "pool further trims the warm forward.\n")


# ---------------------------------------------------------------------------
# 7. Pick quality: STEAD vs TXED (cross-catalog generalization)
# ---------------------------------------------------------------------------
def sec_quality() -> None:
    w("\n## 7. Pick quality vs catalog: STEAD and cross-catalog TXED\n")
    w("F1 vs catalog at 580 stations. `classify` and `classify_batched` use SeisBench's "
      "picker; `annotate`/`slipstream` use RAPID's threshold extractor (note the provenance "
      "difference). `classify_batched` matches `classify` on both STEAD and TXED; TXED quality also comes from the "
      "consolidated native pick-quality rerun.\n")
    for ds in ("stead", "txed"):
        w(f"\n### {ds.upper()}\n")
        w("| Model | classify P/S | classify_batched P/S | annotate P/S | slipstream-fp32 P/S |")
        w("|---|---:|---:|---:|---:|")
        for mo in MODELS:
            def f1(method, dtype=None):
                rows = [r for r in sel(fam="native", method=method, model=mo, dataset=ds, nst=580)
                        if r["pf1"] is not None and (dtype is None or r["dtype"] == dtype)]
                if not rows:
                    return "–"
                r = rows[0]
                return f"{fmt(r['pf1'],2)}/{fmt(r['sf1'],2)}"
            w(f"| {mo} | {f1('classify')} | {f1('classify_batched')} | {f1('annotate')} | {f1('slipstream','fp32')} |")
    w("\n*Interpretation:* TXED is harder than STEAD, especially for EQTransformer P picks, but the "
      "relative behavior is consistent across methods. Absolute F1 differences between `classify*` "
      "and `annotate`/`slipstream` partly reflect the different pick extractors rather than the "
      "forward path.\n")


# ---------------------------------------------------------------------------
# 8. Memory footprint by family
# ---------------------------------------------------------------------------
def sec_memory() -> None:
    w("\n## 8. Memory footprint (peak process-tree PSS, MB)\n")
    w("STEAD 580 stations. PSS counts shared pages once across the process tree, so the actor "
      "pool is not over-counted versus single process.\n")
    w("| Model | native classify | classify_batched CPU | native slipstream-bf16 | Model-Actor CPU | Model-Actor-Slip CPU |")
    w("|---|---:|---:|---:|---:|---:|")
    for mo in MODELS:
        def pss(fam, method, dtype=None, device="cpu"):
            rows = [r for r in sel(fam=fam, method=method, model=mo, dataset="stead", nst=580, device=device)
                    if r["pss"] is not None and (dtype is None or r["dtype"] == dtype)]
            return min((r["pss"] for r in rows), default=None)
        w(f"| {mo} | {fmt(pss('native','classify'),0)} | {fmt(pss('native','classify_batched'),0)} | "
          f"{fmt(pss('native','slipstream','bf16'),0)} | "
          f"{fmt(pss('orch','modelactor'),0)} | {fmt(pss('orch','modelactor_slipstream','bf16'),0)} |")
    w("\n*Interpretation:* the persistent actor pool's peak PSS is the dominant memory cost; "
      "reduced-precision Slipstream actors trim it. Use these numbers for deployment sizing.\n")


if __name__ == "__main__":
    w("# iso_full_benchmark — Results Analysis (model-readable)\n")
    w("_Auto-generated by `benchmarks/analysis/analyze_iso_full.py` from `results/iso_full_benchmark/`. "
      "Numbers are means across repeats from the strictly-sequential isolated runs._\n")
    w("> **Coverage note:** the full cost-reduced isolated result set is complete: the original "
      "1,488-config grid plus the 64-cell `classify_batched` fill-in, for 1,552 generated cells "
      "and 2,412 `result.json` files including consolidated legacy controls. The completed data "
      "include native, cold orchestration, oversubscription, warm streaming, 2-GPU streaming, TXED "
      "pick-quality, precision, batch-size, and thread-sensitivity measurements.\n")
    w("## 0. Key findings (read this first)\n")
    w("1. **A single SeisBench process cannot use a multicore CPU.** Native total time is flat in "
      "the CPU *core budget*; the only single-process lever is thread count, and even that saturates "
      "by ~8 threads. This is the gap RAPID's orchestration fills (Sec. 1).\n")
    w("2. **The out-of-the-box thread default is a trap.** Per-station `classify` at the default "
      "(~64 threads) is 286-1447 s; at 1 thread it is 4.7-22.5 s. `annotate`/`slipstream` are best "
      "at ~4-8 threads but blow past 27 s for heavy EQT models at the default (Sec. 1).\n")
    w("3. **Batched SeisBench `classify()` is the right native picker upper bound.** "
      "`classify_batched` removes the artificial per-station Python loop while keeping "
      "SeisBench's own pick aggregation; use it beside naive `classify` and `annotate` when "
      "describing native baselines (Sec. 2).\n")
    w("4. **Persistence, not just parallelism, is the win.** Cold-start Model-Actor (persistent "
      "pool) picks the whole 580-station network in ~16-17 s on CPU; Ripper (reload-per-task) needs "
      "~138 s — an ~8x gap from persistence alone (Sec. 4).\n")
    w("5. **Real-time picking does not need a GPU.** With a warm pool, CPU Model-Actor matches or "
      "beats GPU `annotate` per feed for every model (e.g. EQT-NC 1.86 s CPU MA vs 2.66 s GPU "
      "annotate at 580 st); BF16 Slipstream in the pool trims it further (Sec. 6).\n")
    w("6. **BF16 is the safe fast precision; FP16 is not.** BF16 preserves F1 for all models and "
      "speeds up heavy EQT models on CPU; FP16 collapses PhaseNetLight (F1 0.64/0.45) and is unsafe "
      "for the EQT family (Sec. 3).\n")
    w("7. **Don't over-pack actors.** Throughput peaks near 0.5-1 actor/core; 4x oversubscription is "
      "~2.5x slower (Sec. 5).\n")
    w("8. **Pickers generalize across catalogs.** F1 is stable from STEAD to TXED; absolute gaps "
      "between `classify` and `annotate`/`slipstream` partly reflect the different pick extractors, "
      "not the forward path (Sec. 7).\n")
    sec_native_threads()
    sec_native_batched()
    sec_precision()
    sec_orch()
    sec_oversub()
    sec_stream()
    sec_quality()
    sec_memory()
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")
