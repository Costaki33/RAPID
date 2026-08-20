#!/usr/bin/env bash
set -euo pipefail
# Stop transfer workers (not this script)
pgrep -af 'run_iso_locked_recipe_transfer|run_annotate_precision_trial|run_orch_annotate_trial|run_locked_recipe_transfer' || true
pkill -f 'run_iso_locked_recipe_transfer.sh' || true
pkill -f 'run_annotate_precision_trial.py' || true
pkill -f 'run_orch_annotate_trial.py' || true
sleep 2
echo "=== remaining ==="
pgrep -af 'run_iso_locked_recipe_transfer|run_annotate_precision_trial|run_orch_annotate_trial' || echo STOPPED

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
pip install -U 'seisbench>=0.12'
python - <<'PY'
import seisbench
import seisbench.models as sbm
print("seisbench", seisbench.__version__)
print("EQCCTP", hasattr(sbm, "EQCCTP"))
print("EQCCTS", hasattr(sbm, "EQCCTS"))
PY

if ! command -v tmux >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y tmux
fi
which tmux
tmux -V
echo READY
