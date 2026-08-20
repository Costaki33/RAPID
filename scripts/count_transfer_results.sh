#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/cgs2528/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19
echo "native=$(find "$ROOT/annotate_bf16" -name result.json 2>/dev/null | wc -l)"
echo "cpu_orch=$(find "$ROOT/ma" -path '*/cpu/*/result.json' 2>/dev/null | wc -l)"
echo "gpu_orch=$(find "$ROOT/ma" -path '*/gpu/*/result.json' 2>/dev/null | wc -l)"
echo "empty_gpu_dirs=$(find "$ROOT/ma" -type d -name gpu -empty 2>/dev/null | wc -l)"
echo "gpu DONE lines=$(grep -cE 'DONE  (playback|staggered).* gpu ' "$ROOT/locked_recipe_transfer.log" || true)"
echo "gpu START lines=$(grep -cE 'START (playback|staggered).* gpu ' "$ROOT/locked_recipe_transfer.log" || true)"
# sample one gpu tree
ls -la "$ROOT/ma/annotate_bf16/stead/580st/EQCCT/gpu" || true
find "$ROOT/ma/annotate_bf16/stead/580st/EQCCT/gpu" -maxdepth 3 2>/dev/null || true
