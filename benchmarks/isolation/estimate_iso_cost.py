#!/usr/bin/env python3
"""Estimate wall-clock cost of the iso_full_benchmark grid under pruning options.

Cost model is built from MEASURED per-repeat total_s in results/fair_benchmark_iso:
 - native classify scales ~linearly in torch threads (heavy models blow up);
   annotate/slipstream are ~flat in threads.
 - orch cold-start: modelactor ~flat; ripper ~scales with stations.
 - oversub: measured mean per-config; streaming: measured session_wall x sessions.
All numbers are estimates meant for RELATIVE comparison of pruning choices.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

MODELS = ["PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]
DS = ["stead", "txed"]
ST = [250, 580]
CORES = [5, 10, 15, 20]
MULTS = [0.25, 0.5, 1, 2, 3, 4]
REPEATS = 3
SPAWN = 3.0           # per-repeat subprocess/import overhead (s)
COMPILE = 45.0        # per-repeat torch.compile warmup (s, rough)
STREAM_SESSIONS = 5

def rnd(m, b, hi=64): return min(hi, max(1, int(math.floor(m * b + 0.5))))
def is_heavy(m): return m in ("EQTransformer", "EQT-NC")
def fp16_ok(m): return m in ("PhaseNet", "PhaseNetLight")

# measured per-repeat thr=1 seconds {model: {st: s}} (txed ~= stead)
CLS1 = {"PhaseNet": {250: 3.6, 580: 5.5}, "PhaseNetLight": {250: 3.4, 580: 4.7},
        "EQTransformer": {250: 10.7, 580: 22.5}, "EQT-NC": {250: 7.1, 580: 13.7}}
ANN1 = {"PhaseNet": {250: 3.3, 580: 4.8}, "PhaseNetLight": {250: 3.2, 580: 4.4},
        "EQTransformer": {250: 5.5, 580: 9.5}, "EQT-NC": {250: 5.6, 580: 9.5}}
SLP1 = {"PhaseNet": {250: 3.6, 580: 5.2}, "PhaseNetLight": {250: 3.4, 580: 4.6},
        "EQTransformer": {250: 7.7, 580: 11.5}, "EQT-NC": {250: 7.7, 580: 11.9}}
# classify thread slope per-thread from 580 stead (time64-time1)/63
CLS_SLOPE580 = {"PhaseNet": (8.0 - 5.5) / 63, "PhaseNetLight": (286.4 - 4.7) / 63,
                "EQTransformer": (1063.8 - 22.5) / 63, "EQT-NC": (1447.1 - 13.7) / 63}

def cls_secs(model, st, thr):
    slope = CLS_SLOPE580[model] * (st / 580.0)
    return CLS1[model][st] + slope * (thr - 1)

def precisions(model):
    dts = ["fp32", "fp16", "bf16"] if fp16_ok(model) else ["fp32", "bf16"]
    return [(d, c) for d in dts for c in (False, True)]

@dataclass
class Prune:
    no_native_classify: bool = False     # drop new native classify (use existing 20-core data)
    no_orch_ripper: bool = False         # drop ripper from orch (keep existing control)
    flat_thread_collapse: bool = False   # annotate/slipstream: thr={1} only (thread-insensitive)
    txed_pq_only: bool = False           # txed native: keep only cores20/thr1 cell; drop txed timing in orch/oversub/stream
    no_compile: bool = False             # drop all +compile variants
    oversub_580_only: bool = False       # oversub: stead+txed 580 only

def native_secs(p: Prune):
    tot = 0.0
    for ds in DS:
        for st in ST:
            for model in MODELS:
                for cores in CORES:
                    threads = sorted({rnd(m, cores) for m in MULTS})
                    txed_pq = p.txed_pq_only and ds == "txed"
                    for thr in threads:
                        keep_pq_cell = (cores == 20 and thr == 1)
                        # classify
                        if not p.no_native_classify:
                            if not (txed_pq and not keep_pq_cell):
                                tot += REPEATS * (cls_secs(model, st, thr) + SPAWN)
                        # annotate + slipstream (flat in threads)
                        flat_threads = (thr == 1) if p.flat_thread_collapse else True
                        if flat_threads and not (txed_pq and not keep_pq_cell):
                            tot += REPEATS * (ANN1[model][st] + SPAWN)
                            for dt, cmp in precisions(model):
                                if cmp and p.no_compile:
                                    continue
                                tot += REPEATS * (SLP1[model][st] + SPAWN + (COMPILE if cmp else 0))
    return tot

def orch_secs(p: Prune):
    tot = 0.0
    for ds in DS:
        if p.txed_pq_only and ds == "txed":
            continue
        for st in ST:
            for model in MODELS:
                for cores in CORES:
                    for dev in ("cpu", "gpu"):
                        ma = (18 if dev == "cpu" else 22)
                        if not p.no_orch_ripper:
                            rip = (138 if dev == "cpu" else 175) * (st / 580.0)
                            tot += REPEATS * rip
                        tot += REPEATS * ma                       # modelactor
                        for dt, cmp in precisions(model):         # modelactor_slipstream
                            if cmp and p.no_compile:
                                continue
                            tot += REPEATS * (ma + 4 + (COMPILE if cmp else 0))
    return tot

def oversub_secs(p: Prune):
    tot = 0.0
    for ds in DS:
        if p.txed_pq_only and ds == "txed":
            continue
        for st in ST:
            if p.oversub_580_only and st != 580:
                continue
            for model in MODELS:
                for cores in CORES:
                    for dev in ("cpu", "gpu"):
                        for mult in MULTS:
                            ma = (24 if dev == "cpu" else 28)
                            tot += REPEATS * ma                   # modelactor fp32
                            for dt, cmp in precisions(model):
                                if cmp and p.no_compile:
                                    continue
                                tot += REPEATS * (28 + (COMPILE if cmp else 0))
    return tot

# streaming session_wall (s) {strategy: {dev: {heavy?: s}}}
SW = {"annotate": {"cpu": {True: 72, False: 15}, "gpu": {True: 17, False: 9}},
      "modelactor": {"cpu": {True: 13, False: 8}, "gpu": {True: 45, False: 13}},
      "slip": {"cpu": {True: 12, False: 8}, "gpu": {True: 45, False: 11}}}

def stream_secs(p: Prune):
    tot = 0.0
    for ds in DS:
        if p.txed_pq_only and ds == "txed":
            continue
        for st in ST:
            for model in MODELS:
                h = is_heavy(model)
                for cores in CORES:
                    for dev in ("cpu", "gpu"):
                        tot += STREAM_SESSIONS * SW["annotate"][dev][h]
                        tot += STREAM_SESSIONS * SW["modelactor"][dev][h]
                        for dt, cmp in precisions(model):
                            if cmp and p.no_compile:
                                continue
                            tot += STREAM_SESSIONS * (SW["slip"][dev][h] + (COMPILE if cmp else 0))
    return tot

def total(p: Prune):
    return {"native": native_secs(p), "orch": orch_secs(p),
            "oversub": oversub_secs(p), "stream": stream_secs(p)}

def show(name, p):
    t = total(p); s = sum(t.values())
    print(f"\n### {name}")
    for k, v in t.items():
        print(f"  {k:9s}: {v/3600:8.1f} h")
    print(f"  {'TOTAL':9s}: {s/3600:8.1f} h  ({s/86400:.1f} days)")
    return s

def stream_secs_keep_compile(p: Prune):
    """Streaming cost with compile KEPT (recommended scope keeps compile in stream)."""
    q = Prune(**{**p.__dict__, "no_compile": False})
    return stream_secs(q)


if __name__ == "__main__":
    full = show("Scenario 0: FULL grid", Prune())
    s1 = show("Scenario 1: your prunes (drop new native classify + drop orch ripper)",
              Prune(no_native_classify=True, no_orch_ripper=True))
    # Recommended/applied scope: all prunes, but compile kept in streaming only.
    applied = Prune(no_native_classify=True, no_orch_ripper=True, txed_pq_only=True,
                    no_compile=True, flat_thread_collapse=True, oversub_580_only=True)
    t = total(applied)
    t["stream"] = stream_secs_keep_compile(applied)   # add streaming compile back
    s_applied = sum(t.values())
    print("\n### APPLIED scope (all prunes; compile kept in streaming only; oversub 3 repeats)")
    for k, v in t.items():
        print(f"  {k:9s}: {v/3600:8.1f} h")
    print(f"  {'TOTAL':9s}: {s_applied/3600:8.1f} h  ({s_applied/86400:.1f} days)")
    print(f"\nSavings: Scenario 1 cuts {(1-s1/full)*100:.0f}% vs full; "
          f"APPLIED cuts {(1-s_applied/full)*100:.0f}% vs full ({full/3600:.0f} h -> {s_applied/3600:.0f} h).")
