#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"
ROOT="results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19"

echo "=== progress ==="
RESULTS_ROOT="$ROOT" bash benchmarks/isolation/watch_locked_recipe_transfer.sh --once || true

echo "=== recent log ==="
tail -30 "$ROOT/locked_recipe_transfer.log" || true

echo "=== metrics ==="
python - <<'PY'
import json
from pathlib import Path
root = Path("results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19")
rows = []
for p in sorted(root.rglob("result.json")):
    d = json.loads(p.read_text())
    t = d.get("timing") or {}
    parts = p.parts
    rel = p.relative_to(root).as_posix()
    sr = t.get("success_rate")
    if "stead" not in parts:
        continue
    i = parts.index("stead")
    nst = parts[i + 1]
    model = parts[i + 2]
    device = parts[i + 3]
    if rel.startswith("annotate_bf16/"):
        layer = "native"
        ncpu = next(x for x in parts if x.startswith("cpus"))
        metric = t.get("inference_s_mean")
        tag = ncpu
        key = "inf_s"
    else:
        layer = "playback" if "/playback/" in rel else "staggered"
        tag = next(x for x in parts if x.startswith("kma"))
        if layer == "playback":
            metric = (d.get("orch") or {}).get("makespan_s_mean")
            key = "make_s"
        else:
            metric = (
                ((d.get("latency") or {}).get("pooled_across_repeats") or {})
                .get("e2e_finish_minus_ready") or {}
            ).get("p95")
            key = "p95_fr"
    pq = d.get("pick_quality_vs_catalog") or {}
    p_f1 = s_f1 = None
    if isinstance(pq, dict):
        p_f1 = pq.get("P.f1_mean")
        s_f1 = pq.get("S.f1_mean")
        if isinstance(pq.get("P"), dict) and p_f1 is None:
            p_f1 = pq["P"].get("f1")
        if isinstance(pq.get("S"), dict) and s_f1 is None:
            s_f1 = pq["S"].get("f1")
        if p_f1 is None or s_f1 is None:
            reps = pq.get("repeats") or []
            if isinstance(reps, list) and reps and isinstance(reps[0], dict):
                if p_f1 is None and isinstance(reps[0].get("P"), dict):
                    p_f1 = reps[0]["P"].get("f1")
                if s_f1 is None and isinstance(reps[0].get("S"), dict):
                    s_f1 = reps[0]["S"].get("f1")
    rows.append((layer, model, nst, device, tag, key, None if metric is None else round(float(metric), 3), sr, p_f1, s_f1))

print(f"{'layer':10} {'model':14} {'nst':6} {'dev':4} {'tag':18} {'metric':8} {'val':8} {'sr':5}")
for r in rows:
    print(f"{r[0]:10} {r[1]:14} {r[2]:6} {r[3]:4} {r[4]:18} {r[5]:8} {str(r[6]):8} {r[7]}")
print("n=", len(rows))
PY
