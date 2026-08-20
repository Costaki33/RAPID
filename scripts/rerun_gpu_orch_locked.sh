#!/usr/bin/env bash
# Re-run missing GPU orch cells at locked K (GPU_K_CAP=4).
# If actors OOM, stop and re-run with GPU_K_CAP=2, then note it in README_machine.txt.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

export RAPID_PYTHON="$(command -v python)"
export CORES="${CORES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
export CPU_K_CAP="${CPU_K_CAP:-10}"
# Locked GPU want is K=4 (PhaseNet@250 stays K=2 via matrix policy).
export GPU_K_CAP="${GPU_K_CAP:-4}"
export LAYER=playback,staggered
export DEVICES=gpu
export RESULTS_ROOT=results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19
unset SMOKE || true

{
  echo
  echo "GPU orch re-run $(date -Is)"
  echo "LAYER=$LAYER DEVICES=$DEVICES GPU_K_CAP=$GPU_K_CAP CPU_K_CAP=$CPU_K_CAP CORES=$CORES"
  echo "Intent: match locked GPU K=4; if OOM lower GPU_K_CAP and note VRAM."
} >> "$RESULTS_ROOT/README_machine.txt"

echo "Re-running GPU orch into $RESULTS_ROOT with GPU_K_CAP=$GPU_K_CAP"
bash benchmarks/isolation/run_iso_locked_recipe_transfer.sh
