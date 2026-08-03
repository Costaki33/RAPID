#!/usr/bin/env python3
"""Analyze isolated benchmark results and emit a model-readable report.

Walks every result.json under results/iso_full_benchmark/ (the new grid plus the
consolidated legacy cells that were copied in), extracts the headline metrics
(timing total_s, warm streaming latency, peak PSS memory, pick-quality F1), and
writes RESULTS_ANALYSIS.md with tables + one-line interpretations. Exact matched
controls in fair_benchmark_iso supplement cells not consolidated into the full tree.

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
FAIR = ROOT / "results" / "fair_benchmark_iso"
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
        allowed_tags = {f"iso_thr{t}" for t in (0, 1, 2, 4, 8)}
        if meth == "annotate":
            allowed_tags.discard("iso_thr8")
            allowed_tags.add("iso_cpu8_thr8")
        rows = [r for r in sel(fam="native", method=meth, dataset="stead", nst=580, device="cpu")
                if r["dtype"] in (None, "fp32") and r["tag"] in allowed_tags]
        by = defaultdict(dict)  # model -> {threads: total_s}
        for r in rows:
            if r["total_s"] is not None:
                by[r["model"]][r["threads"]] = r["total_s"]
        if meth == "annotate":
            for mo in MODELS:
                r = jload(str(FAIR / f"native/annotate/stead/580st/{mo}/iso_thr8/result.json"))
                if not r:
                    continue
                m = r["meta"]
                total = (r.get("timing") or {}).get("total_s_mean")
                if m.get("n_cpus") == 20 and m.get("torch_threads") == 8 and total is not None:
                    by[mo][8] = total
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
    w("\n*Interpretation:* the three methods have **different** thread profiles. In these tests, "
      "per-station `classify` was fastest at **1 thread** — the default (~64) is catastrophic "
      "(286-1447 s) and "
      "no higher count helps, because each call processes a single station. Batched `annotate` "
      "and `slipstream` are fastest at a **few threads** (~4-8: e.g. EQT annotate 5.6 s at t=8 vs "
      "9.5 s at t=1) but fall off a cliff at high thread counts for the heavy EQT models (>27 s at "
      "the default). No tested method improves past ~8 threads, and increasing the host CPU "
      "*core budget* alone does not improve runtime when intra-op threads are held fixed. "
      "Cross-process orchestration adds the station-level parallelism evaluated here. **Note:** "
      "the orch/oversub/stream families run their "
      "single-process baselines at 1 thread to isolate the core-budget axis; the per-method "
      "optimum for `annotate`/`slipstream` is a few threads, so their best single-process numbers "
      "are the optima in this table, not the 1-thread column.\n")


# ---------------------------------------------------------------------------
# 2. Native picker comparison: classify vs classify_batched vs annotate
# ---------------------------------------------------------------------------
def sec_native_batched() -> None:
    w("\n## 2. Native picker comparison: per-station vs batched SeisBench\n")
    w("STEAD 580 stations, matched 5-core host allocation and one intra-op thread. "
      "`classify` is the naive per-station loop; `classify_batched` "
      "is one SeisBench `classify()` call on the merged network stream, so it gets "
      "cross-station batching and SeisBench's own pick aggregation.\n")
    w("| Model | classify CPU | classify_batched CPU | classify_batched GPU | annotate CPU | batched/classify CPU speedup |")
    w("|---|---:|---:|---:|---:|---:|")
    for mo in MODELS:
        def exact(method, device, tag):
            rows = [r for r in sel(fam="native", method=method, model=mo, dataset="stead", nst=580, device=device)
                    if r["total_s"] is not None
                    and r["dtype"] in (None, "fp32")
                    and r["threads"] == 1
                    and r["ncpus"] == 5
                    and r["tag"] == tag]
            return rows[0]["total_s"] if rows else None

        classify = exact("classify", "cpu", "c5_thr1")
        bat_cpu = exact("classify_batched", "cpu", "cpu_c5_thr1")
        bat_gpu = exact("classify_batched", "gpu", "gpu_c5_thr1")
        annotate = exact("annotate", "cpu", "c5_thr1")
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
    w("STEAD 580 stations, single process, one PyTorch intra-op thread, batch size 256. "
      "F1 is reported against the catalog (P / S). FP16 was excluded for EQTransformer / "
      "EQT-NC because the tested SeisBench implementation uses a `-1e10` padding sentinel "
      "outside finite FP16 range.\n")
    w("| Model | dtype | compile | threads | total s | P F1 | S F1 |")
    w("|---|---|---|---:|---:|---:|---:|")
    rows = sel(fam="native", method="slipstream", dataset="stead", nst=580, device="cpu")
    key = lambda r: (MODELS.index(r["model"]) if r["model"] in MODELS else 9,
                     {"fp32": 0, "fp16": 1, "bf16": 2}.get(r["dtype"], 9), r["compile"])
    precision_tags = {
        ("fp32", False): "iso_fp32",
        ("fp16", False): "iso_fp16",
        ("fp16", True): "iso_fp16_compile",
        ("bf16", False): "iso_bf16",
        ("bf16", True): "iso_bf16_compile",
    }
    rows = [
        r for r in rows
        if r["threads"] == 1
        and r["tag"] == precision_tags.get((r["dtype"], r["compile"]))
    ]
    for r in sorted(rows, key=key):
        w(f"| {r['model']} | {r['dtype']} | {'yes' if r['compile'] else 'no'} | "
          f"{r['threads']} | {fmt(r['total_s'],1)} | {fmt(r['pf1'],3)} | {fmt(r['sf1'],3)} |")
    w("\n*Interpretation:* At the matched one-thread setting, BF16 reduced eager total time from "
      "4.4 to 4.0 s for PhaseNet, increased it from 4.1 to 5.1 s for PhaseNetLight, and reduced "
      "it from 9.0 to 7.5 s and 8.9 to 7.4 s for EQTransformer and EQT-NC. The largest absolute "
      "FP32-to-BF16 F1 difference was about 0.007 (PhaseNetLight S). FP16 substantially degraded "
      "PhaseNetLight pick quality and was slower for PhaseNet on the tested CPU/PyTorch path. "
      "`torch.compile` adds initialization cost in these cold totals and should not be interpreted "
      "as steady-state forward latency.\n")


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
            expected_tag = (
                f"{device}_c20" if method == "modelactor"
                else f"iso_{device}_580"
            )
            rows = [r for r in sel(fam="orch", method=method, model=mo, dataset="stead", nst=580, device=device)
                    if r["total_s"] is not None
                    and r["ncpus"] == 20
                    and r["dtype"] in (None, "fp32")
                    and r["tag"] == expected_tag]
            return rows[0]["total_s"] if rows else None
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
    w("Warm per-feed latency (s), mean over 10 independent repeats of eight feeds for the "
      "single-device head-to-head cells (feed 0 dropped; 70 warm feeds per cell), using a "
      "20-core budget. Two-GPU cells use five repeats (35 warm feeds). Lower is better.\n")
    for nst in (580, 250):
        w(f"\n### {nst} stations\n")
        w("| Model | NBC CPU | annotate CPU | Model-Actor CPU | Slipstream-BF16 CPU | "
          "NBC GPU | annotate GPU | Model-Actor GPU | 2-GPU Model-Actor |")
        w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mo in MODELS:
            def warm(method, device, dtype=None):
                if method == "stream_modelactor_2gpu":
                    wanted_tag = f"iso_2gpu_{nst}_cpu20"
                elif device == "cpu":
                    wanted_tag = f"iso_cpu_{nst}"
                else:
                    wanted_tag = f"iso_gpu_{nst}"
                rows = [r for r in sel(fam="stream", method=method, model=mo, dataset="stead",
                                       nst=nst, device=device, ncpus=20)
                        if r["warm"] is not None
                        and r["tag"] == wanted_tag
                        and (dtype is None or r["dtype"] == dtype)]
                return rows[0]["warm"] if rows else None
            w(f"| {mo} | {fmt(warm('stream_classify_batched','cpu'))} | "
              f"{fmt(warm('stream_annotate','cpu'))} | {fmt(warm('stream_modelactor','cpu'))} | "
              f"{fmt(warm('stream_modelactor_slipstream','cpu','bf16'))} | "
              f"{fmt(warm('stream_classify_batched','gpu'))} | {fmt(warm('stream_annotate','gpu'))} | "
              f"{fmt(warm('stream_modelactor','gpu'))} | {fmt(warm('stream_modelactor_2gpu','gpu'))} |")
    have_nbc = any(r["method"] == "stream_classify_batched" and r["warm"] is not None for r in RECS)
    if have_nbc:
        w("\n*Interpretation:* warm Network-Batched Classify (NBC) is the output-matched native "
          "discrete-pick baseline. Compare Model-Actor against NBC for like-for-like picks; Annotate "
          "remains a probability-trace baseline. Slipstream-BF16 changed actor-pool latency by only a "
          "few percent and was not uniformly faster.\n")
    else:
        w("\n*Interpretation:* under the tested configurations, warm CPU Model-Actor was faster than "
          "warm Annotate and the tested single-GPU Annotate path for all four 580-station cells. "
          "This comparison is configuration-specific; warm Network-Batched Classify was not measured. "
          "Slipstream-BF16 changed actor-pool latency by only a few percent and was not uniformly faster.\n")

# ---------------------------------------------------------------------------
# 7. Pick quality: STEAD vs TXED (cross-catalog generalization)
# ---------------------------------------------------------------------------
def sec_quality() -> None:
    w("\n## 7. Pick quality vs catalog: STEAD and cross-catalog TXED\n")
    w("F1 vs catalog at 580 stations. `classify` and `classify_batched` use SeisBench's "
      "picker; `annotate`/`slipstream` use RAPID's threshold extractor (note the provenance "
      "difference). `classify_batched` was evaluated for pick quality on STEAD, while TXED quality "
      "comes from the consolidated native pick-quality rerun.\n")
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
    w("\n*Interpretation:* TXED produced lower F1 values than STEAD, especially for EQTransformer "
      "P picks, consistent with domain shift. Discrete-pick metrics are not directly attributable "
      "to orchestration because `classify*` and `annotate`/`slipstream` use different extractors; "
      "the EQTransformer P-F1 differences are material and should not be described as equivalent.\n")


# ---------------------------------------------------------------------------
# 8. Memory footprint by family
# ---------------------------------------------------------------------------
def sec_memory() -> None:
    w("\n## 8. Memory footprint (peak process-tree PSS, MB)\n")
    w("STEAD 580 stations. PSS counts shared pages once across the process tree. Values are from "
      "the matched 20-core streaming protocol, with sampling spanning initialization and all "
      "feeds. GPU VRAM is a separate quantity and is not mixed into this table.\n")
    w("| Model | Annotate session peak | Model-Actor session peak | Model-Actor-Slip session peak |")
    w("|---|---:|---:|---:|")
    for mo in MODELS:
        def warm_pss(method, dtype=None):
            rows = [r for r in sel(fam="stream", method=method, model=mo, dataset="stead",
                                   nst=580, device="cpu", ncpus=20)
                    if r["pss"] is not None
                    and r["tag"] == "iso_cpu_580"
                    and (dtype is None or r["dtype"] == dtype)]
            return rows[0]["pss"] if rows else None
        w(f"| {mo} | {fmt(warm_pss('stream_annotate'),0)} | "
          f"{fmt(warm_pss('stream_modelactor'),0)} | "
          f"{fmt(warm_pss('stream_modelactor_slipstream','bf16'),0)} |")
    w("\n*Interpretation:* persistent actors trade memory for warm latency, reaching roughly "
      "11–12 GB peak PSS in the matched 20-worker streaming session. BF16 did not consistently "
      "or materially reduce peak host PSS relative to Model-Actor[classify].\n")


if __name__ == "__main__":
    have_nbc = any(r["method"] == "stream_classify_batched" and r["warm"] is not None for r in RECS)
    nbc_note = ("Warm Network-Batched Classify cells are included where present."
                if have_nbc else
                "Warm Network-Batched Classify was not measured.")
    w("# iso_full_benchmark — Results Analysis (model-readable)\n")
    w("_Auto-generated by `benchmarks/analysis/analyze_iso_full.py` from exact isolated tags in "
      "`results/iso_full_benchmark/`, supplemented by matched controls in `results/fair_benchmark_iso/`. "
      "Numbers are means across repeats from the strictly-sequential isolated runs._\n")
    w("> **Coverage note:** the report includes native, cold orchestration, oversubscription, warm "
      "streaming, two-GPU streaming, TXED pick-quality, precision, batch-size, and thread-sensitivity "
      f"measurements. {nbc_note}\n")
    w("## 0. Key findings (read this first)\n")
    w("1. **A larger host-core allocation alone does not speed up the tested single-process path.** "
      "With intra-op threads held fixed, native total time is flat across the core-budget sweep; "
      "station-level process parallelism is the gap RAPID's orchestration fills (Sec. 1).\n")
    w("2. **The out-of-the-box thread default is a trap.** Per-station `classify` at the default "
      "(~64 threads) is 286-1447 s; at 1 thread it is 4.7-22.5 s. `annotate`/`slipstream` are best "
      "at ~4-8 threads but blow past 27 s for heavy EQT models at the default (Sec. 1).\n")
    w("3. **Batched SeisBench `classify()` is an important native picker baseline.** "
      "`classify_batched` removes the artificial per-station Python loop while keeping "
      "SeisBench's own pick aggregation; use it beside naive `classify` and `annotate` when "
      "describing native baselines (Sec. 2).\n")
    w("4. **Persistence, not just parallelism, is the win.** Cold-start Model-Actor (persistent "
      "pool) picks the whole 580-station network in ~16-17 s on CPU; Ripper (reload-per-task) needs "
      "~138 s — an ~8x gap from persistence alone (Sec. 4).\n")
    if have_nbc:
        w("5. **Warm Network-Batched Classify closes the native discrete-pick gap.** Use Sec. 6 to "
          "compare Model-Actor against NBC (same SeisBench picks) rather than only against Annotate "
          "probability traces.\n")
    else:
        w("5. **A CPU-only deployment met the study target.** With a warm pool, CPU Model-Actor "
          "matched or beat the tested GPU `annotate` configuration for every 580-station model cell. "
          "Warm Network-Batched Classify was not measured, so this is not a universal GPU comparison "
          "(Sec. 6).\n")
    w("6. **BF16 was the safer reduced-precision option.** At matched thread settings, BF16 "
      "changed latency in a model-dependent way while keeping absolute F1 differences within "
      "about 0.007; FP16 substantially degraded PhaseNetLight and was excluded for EQTransformer "
      "because of an implementation sentinel outside finite FP16 range (Sec. 3).\n")
    w("7. **Don't over-pack actors.** Throughput peaks near 0.5-1 actor/core; 4x oversubscription is "
      "~2.5x slower (Sec. 5).\n")
    w("8. **TXED is a harder cross-catalog test.** F1 is lower on TXED, particularly for "
      "EQTransformer P picks. Absolute gaps among methods also reflect different pick extractors, "
      "so they are not a pure orchestration effect (Sec. 7).\n")
    sec_native_threads()
    sec_native_batched()
    sec_precision()
    sec_orch()
    sec_oversub()
    sec_stream()
    sec_quality()
    sec_memory()
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")
