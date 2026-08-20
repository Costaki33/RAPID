#!/usr/bin/env bash
# Machine-local launcher for docs/RAPID_LOCKED_RECIPE_TRANSFER.md on this laptop.
# i7-13700H + RTX 4050 6GB / WSL2. Do not change locked recipes.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

export RAPID_PYTHON="$(command -v python)"
# Full logical CPU list so CORE_GRID 5/10/15/20 never skips on this 20-thread WSL host.
# (Earlier XPS-style "one ID per physical core" only had 10 IDs and skipped 15/20.)
export CORES="${CORES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
# Laptop caps from the transfer brief (6 GB VRAM; ~32 GB host / 24 GB WSL).
export GPU_K_CAP="${GPU_K_CAP:-2}"
export CPU_K_CAP="${CPU_K_CAP:-10}"

MODE="${1:-smoke}"
case "$MODE" in
  smoke)
    export RESULTS_ROOT="results/locked_recipe_transfer/$(hostname -s)_smoke"
    export SMOKE=1
    ;;
  full)
    export RESULTS_ROOT="results/locked_recipe_transfer/$(hostname -s)_$(date +%Y-%m-%d)"
    unset SMOKE || true
    ;;
  *)
    echo "usage: $0 smoke|full" >&2
    exit 2
    ;;
esac

mkdir -p "$RESULTS_ROOT"
cat > "$RESULTS_ROOT/README_machine.txt" <<EOF
Host transfer run — $(date -Ins)
hostname: $(hostname)
CPU: Intel Core i7-13700H (WSL: 10 physical CORE IDs / 20 logical)
GPU: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB
RAPID_PYTHON=$RAPID_PYTHON
CORES=$CORES
GPU_K_CAP=$GPU_K_CAP
CPU_K_CAP=$CPU_K_CAP
Note: CORE_GRID cells with n_cpus > 10 isolated IDs are skipped.
Do not change locked recipes (bf16/batch512/MA-SG/eager).
EOF

echo "MODE=$MODE RESULTS_ROOT=$RESULTS_ROOT"
echo "CORES=$CORES GPU_K_CAP=$GPU_K_CAP CPU_K_CAP=$CPU_K_CAP"
echo "RAPID_PYTHON=$RAPID_PYTHON"
"$RAPID_PYTHON" benchmarks/fair/locked_recipe_transfer_matrix.py --print-matrix | head -20
bash benchmarks/isolation/run_iso_locked_recipe_transfer.sh
