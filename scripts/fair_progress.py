#!/usr/bin/env python3
"""Progress + quick-compare reader for the unified fair benchmark (schema v2).

Walks ``results/fair_benchmark/**/result.json`` and reports, per (family,
method), how many trials are complete (all repeats succeeded), partial, or
empty. With ``--table`` it prints a compact apples-to-apples comparison of the
same JSON columns (total_s, per-stage seconds, peak RAM, P/S F1) across methods
for a chosen dataset/stations/model.

Examples::

    python scripts/fair_progress.py
    python scripts/fair_progress.py --table --dataset stead --n-stations 250 --model EQTransformer
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

RAPID_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("framework_init", "model_load", "waveform_access", "preprocess", "inference", "pick_generation")


def _load(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _status(d: Dict[str, Any]) -> str:
    reps = d.get("timing", {}).get("repeats", [])
    ok = sum(1 for r in reps if r.get("success"))
    want = d.get("meta", {}).get("repeats", 0) or 0
    if want and ok >= want:
        return "complete"
    if ok > 0:
        return "partial"
    return "empty"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=RAPID_ROOT / "results" / "fair_benchmark")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-stations", type=int, default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "gpu"])
    args = ap.parse_args()

    results = sorted(args.results_root.glob("**/result.json"))
    docs: List[Dict[str, Any]] = []
    for p in results:
        d = _load(p)
        if d and "meta" in d:
            d["_path"] = str(p)
            docs.append(d)

    if not args.table:
        counts: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"complete": 0, "partial": 0, "empty": 0})
        for d in docs:
            m = d["meta"]
            key = (m.get("family", "?"), m.get("method", "?"))
            counts[key][_status(d)] += 1
        print(f"Fair benchmark results under {args.results_root}\n")
        print(f"{'family':<14}{'method':<26}{'complete':>9}{'partial':>9}{'empty':>7}")
        tot = {"complete": 0, "partial": 0, "empty": 0}
        for key in sorted(counts):
            c = counts[key]
            for k in tot:
                tot[k] += c[k]
            print(f"{key[0]:<14}{key[1]:<26}{c['complete']:>9}{c['partial']:>9}{c['empty']:>7}")
        print("-" * 65)
        print(f"{'TOTAL':<40}{tot['complete']:>9}{tot['partial']:>9}{tot['empty']:>7}")
        print(f"\n{len(docs)} result.json files found.")
        return 0

    # Comparison table
    rows = []
    for d in docs:
        m = d["meta"]
        if args.dataset and m.get("dataset") != args.dataset:
            continue
        if args.n_stations and m.get("n_stations") != args.n_stations:
            continue
        if args.model and m.get("model") != args.model:
            continue
        if args.device and m.get("device") != args.device:
            continue
        if _status(d) == "empty":
            continue
        t = d.get("timing", {})
        pq = d.get("pick_quality_vs_catalog", {}) or {}
        rows.append((m, t, d.get("memory", {}), pq))

    if not rows:
        print("No matching results.")
        return 0

    def _regime(m) -> str:
        """Distinguish the two windowing regimes that share in_samples=3001.

        Both PhaseNet-family regimes use overlap 0; the difference is the network
        trace length (and hence window count): regime A reads the 6000-sample net
        -> 2 windows/station ("6kx2"); regime B reads the trimmed 3001 net -> 1.
        """
        ins = int(m.get("in_samples", m.get("window_samples", 0)) or 0)
        net = str(m.get("net_window", ""))
        if ins >= 6000:
            return "6000"
        if net == "w3001":
            return "3001"
        return "6kx2"  # two 3001-sample windows over the 6000-sample trace

    hdr = (f"{'method':<22}{'dev':<5}{'cpus':>5}{'regime':>7}{'nwin':>6}{'dtype':>7}{'bs':>5}"
           f"{'total':>8}{'mload':>7}{'infer':>7}{'ram_mb':>9}{'vram':>7}{'P_f1':>6}{'S_f1':>6}")
    print(hdr)
    print("-" * len(hdr))
    def sortkey(r):
        m = r[0]
        return (m.get("method", ""), _regime(m), m.get("device", ""), m.get("n_cpus", 0), m.get("tag", ""))
    for m, t, mem, pq in sorted(rows, key=sortkey):
        total = t.get("total_s_mean", float("nan"))
        mload = t.get("model_load_s_mean", float("nan"))
        infer = t.get("inference_s_mean", float("nan"))
        ram = mem.get("peak_ram_mb_mean", float("nan"))
        vram = mem.get("peak_vram_mb_mean", float("nan"))
        pf1 = (pq.get("P") or {}).get("f1", float("nan"))
        sf1 = (pq.get("S") or {}).get("f1", float("nan"))
        dtype_disp = str(m.get("dtype", "")) + ("c" if m.get("compile") else "")
        print(f"{m.get('method',''):<22}{m.get('device',''):<5}{m.get('n_cpus',0):>5}"
              f"{_regime(m):>7}{str(m.get('n_windows','')):>6}{dtype_disp:>7}{str(m.get('batch_size','')):>5}"
              f"{total:>8.2f}{mload:>7.2f}{infer:>7.2f}{ram:>9.0f}{vram:>7.0f}{pf1:>6.3f}{sf1:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
