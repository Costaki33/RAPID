#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"
ROOT="results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19"

python - <<'PY'
import sys
sys.path.insert(0, "benchmarks/fair")
from locked_recipe_transfer_matrix import iter_cells, locked_k
from collections import defaultdict

cells = iter_cells(cpu_k_cap=10, gpu_k_cap=2)
keys = defaultdict(list)
for c in cells:
    if c["layer"] == "native":
        continue
    k = locked_k(
        device=c["device"],
        n_stations=c["n_stations"],
        model=c["model"],
        n_cpus=c["n_cpus"],
        cpu_k_cap=10,
        gpu_k_cap=2,
    )
    key = (c["layer"], c["model"], c["n_stations"], c["device"], k, c["arrival"], c["fill"])
    keys[key].append(c["n_cpus"])

orch_n = sum(1 for c in cells if c["layer"] != "native")
unique = len(keys)
skipped_extra = sum(len(v) - 1 for v in keys.values())
print(f"orch matrix cells={orch_n}")
print(f"unique result paths={unique}")
print(f"duplicate core-budget cells that SKIP after first write={skipped_extra}")
print()
print("Why GPU SKIP cpus=10/15/20:")
print("  result path uses kma{K} only — NOT n_cpus")
print("  with GPU_K_CAP=2 every GPU orch cell collapses to k=2")
print("  first core budget (cpus=5) writes the file; later budgets hit is_done and SKIP")
print()
print("GPU locked K with GPU_K_CAP=2 vs uncapped:")
for model in ["PhaseNet", "PhaseNetLight", "EQCCT", "EQTransformer", "EQT-NC"]:
    for nst in (250, 580):
        k2 = locked_k(device="gpu", n_stations=nst, model=model, n_cpus=20, gpu_k_cap=2)
        k4 = locked_k(device="gpu", n_stations=nst, model=model, n_cpus=20, gpu_k_cap=None)
        print(f"  {model:14} {nst}st  capped2={k2}  uncapped={k4}")
PY

echo
echo "=== GPU orch result.json by kma ==="
find "$ROOT/ma" -path '*/gpu/*/result.json' 2>/dev/null | sed 's|.*/\(kma[^/]*\)/.*|\1|' | sort | uniq -c
echo "GPU orch results:" "$(find "$ROOT/ma" -path '*/gpu/*/result.json' 2>/dev/null | wc -l)"
echo "CPU orch results:" "$(find "$ROOT/ma" -path '*/cpu/*/result.json' 2>/dev/null | wc -l)"
echo "SKIP lines (dedup, not CORES):" "$(grep -c 'SKIP playback\|SKIP staggered' "$ROOT/locked_recipe_transfer.log" || true)"
echo "SKIP not enough CORES:" "$(grep -c 'not enough CORES' "$ROOT/locked_recipe_transfer.log" || true)"
