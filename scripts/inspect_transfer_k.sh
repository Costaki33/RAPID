#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cgs2528/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19
cd /home/cgs2528/RAPID
ls -la "$ROOT" | head
echo "==== README_machine ===="
cat "$ROOT/README_machine.txt" 2>/dev/null || echo missing
echo "==== README ===="
cat "$ROOT/README.md" 2>/dev/null || echo missing
echo "==== log head ===="
grep -E "GPU_K_CAP|CPU_K_CAP|CORES|TRANSFER START" "$ROOT/locked_recipe_transfer.log" | head -20
echo "==== GPU kma ===="
find "$ROOT/ma" -path '*/gpu/*/result.json' | sed 's|.*/\(kma[^/]*\)/.*|\1|' | sort | uniq -c
echo "==== CPU kma ===="
find "$ROOT/ma" -path '*/cpu/*/result.json' | sed 's|.*/\(kma[^/]*\)/.*|\1|' | sort | uniq -c
