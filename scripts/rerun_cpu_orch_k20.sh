#!/usr/bin/env bash
# Re-run missing CPU orch cells at workstation locked K=20 (uncap CPU_K_CAP).
# Extends existing results root; does NOT wipe. SKIP duplicates once kma20 exists.
#
# Locked want:
#   CPU 580 all models → K=20
#   CPU 250 EQCCT      → K=20
#   CPU 250 others     → K=10 (already present as kma10; SKIP)
#
# Usage (tmux-friendly):
#   bash scripts/rerun_cpu_orch_k20.sh
#   # or: tmux new -s k20 'bash ~/RAPID/scripts/rerun_cpu_orch_k20.sh'
#
# Watch:
#   tail -f results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/locked_recipe_transfer.log
#   bash benchmarks/isolation/watch_locked_recipe_transfer.sh
#
# If Ray/OOM at K=20: stop, set CPU_K_CAP=15, re-run, note in README_machine.txt.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

export RAPID_PYTHON="$(command -v python)"
export CORES="${CORES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}"
# Uncap CPU so locked want K=20 can be achieved when n_cpus=20.
unset CPU_K_CAP || true
export GPU_K_CAP="${GPU_K_CAP:-4}"
export LAYER=playback,staggered
export DEVICES=cpu
# Focused: only CORE_GRID=20 → produces kma20 for 580/* and 250/EQCCT;
# 250 non-EQCCT resolve to kma10 and SKIP (already present).
export CORE_GRID="${CORE_GRID:-20}"
export RESULTS_ROOT="${RESULTS_ROOT:-results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19}"
unset SMOKE || true

mkdir -p "$RESULTS_ROOT"
{
  echo
  echo "CPU orch K=20 re-run $(date -Is)"
  echo "LAYER=$LAYER DEVICES=$DEVICES CORE_GRID=$CORE_GRID GPU_K_CAP=$GPU_K_CAP CPU_K_CAP=<unset>"
  echo "CORES=$CORES"
  echo "Intent: workstation locked CPU K=20 (580 all + 250 EQCCT). RAM risk on laptop/WSL."
  echo "Fallback if OOM: CPU_K_CAP=15 then note failure; do not invent new recipes."
} >> "$RESULTS_ROOT/README_machine.txt"

echo "Re-running CPU orch into $RESULTS_ROOT (uncapped CPU_K_CAP, CORE_GRID=$CORE_GRID)"
echo "Audit before:"
"$RAPID_PYTHON" /mnt/c/Users/cgs2528/Projects/RAPID/scripts/audit_locked_k20_missing.py || true
# Prefer repo copy if present under ~/RAPID
if [[ -f scripts/audit_locked_k20_missing.py ]]; then
  "$RAPID_PYTHON" scripts/audit_locked_k20_missing.py || true
fi

bash benchmarks/isolation/run_iso_locked_recipe_transfer.sh
rc=$?

echo "Audit after (rc=$rc):"
if [[ -f scripts/audit_locked_k20_missing.py ]]; then
  "$RAPID_PYTHON" scripts/audit_locked_k20_missing.py || true
else
  "$RAPID_PYTHON" /mnt/c/Users/cgs2528/Projects/RAPID/scripts/audit_locked_k20_missing.py || true
fi
exit $rc
