#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"
python /mnt/c/Users/cgs2528/Projects/RAPID/scripts/plot_locked_transfer_results.py
