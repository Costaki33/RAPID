#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"
ROOT=results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19

echo "=== result counts ==="
echo "native=$(find "$ROOT/annotate_bf16" -name result.json | wc -l)"
echo "cpu_orch=$(find "$ROOT/ma" -path '*/cpu/*/result.json' | wc -l)"
echo "gpu_orch=$(find "$ROOT/ma" -path '*/gpu/*/result.json' | wc -l)"
echo
echo "=== GPU kma used ==="
find "$ROOT/ma" -path '*/gpu/*/result.json' | sed 's|.*/gpu/\(kma[^/]*\)/.*|\1|' | sort | uniq -c
echo
echo "=== GPU success_rate ==="
python - <<'PY'
import json
from pathlib import Path
root = Path("results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19")
for p in sorted(root.glob("ma/**/gpu/**/result.json")):
    d=json.loads(p.read_text())
    sr=(d.get("timing") or {}).get("success_rate")
    parts=p.parts
    model=parts[parts.index("stead")+2] if False else None
    i=parts.index("stead"); model=parts[i+2]; nst=parts[i+1]
    k=[x for x in parts if x.startswith("kma")][0]
    layer="playback" if "/playback/" in p.as_posix() else "staggered"
    print(f"{layer:10} {model:14} {nst:6} {k:12} sr={sr}")
PY

echo
echo "=== matrix status ==="
export GPU_K_CAP=4 CPU_K_CAP=10
python benchmarks/fair/locked_recipe_transfer_matrix.py --status "$ROOT"
echo
echo "=== SKIP reasons in latest GPU re-run ==="
# lines after last TRANSFER START with gpu focus
awk '/TRANSFER START/{s=$0; n=NR} END{}' "$ROOT/locked_recipe_transfer.log" >/dev/null
# show counts
echo "not enough CORES: $(grep -c 'not enough CORES' "$ROOT/locked_recipe_transfer.log" || true)"
echo "SKIP playback/staggered (dedup): $(grep -cE 'SKIP (playback|staggered)' "$ROOT/locked_recipe_transfer.log" || true)"
echo "DONE gpu orch: $(grep -cE 'DONE  (playback|staggered).* gpu ' "$ROOT/locked_recipe_transfer.log" || true)"
