#!/usr/bin/env bash
# Isolated Per-station Classify thread sweep at 20 cores: thr {2,4,8}
# (thr1 and thr0/default already measured). STEAD 580.
set -u
cd "$(dirname "$0")/../.."
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate rapid

ROOT_FULL=results/iso_full_benchmark
ROOT_ISO=results/fair_benchmark_iso
LOG=$ROOT_ISO/iso_classify_threadsweep.log
mkdir -p "$ROOT_FULL" "$ROOT_ISO"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
THREADS=(2 4 8)
CORES=$(seq -s, 0 19)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

note "CLASSIFY THREADSWEEP START"
for M in "${MODELS[@]}"; do
  ins=$(insamples "$M")
  for T in "${THREADS[@]}"; do
    note "CLASSIFY thr$T / $M / 580st (in=$ins)"
    for ROOT in "$ROOT_ISO/native" "$ROOT_FULL/native"; do
      python3 benchmarks/fair/run_fair_trial.py \
        --method classify --dataset stead --n-stations 580 --model "$M" \
        --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads "$T" \
        --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
        --repeats 3 --tag "iso_thr${T}" --results-root "$ROOT" --resume >> "$LOG" 2>&1
    done
  done
done
note "CLASSIFY THREADSWEEP DONE"
