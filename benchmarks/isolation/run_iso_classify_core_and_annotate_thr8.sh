#!/usr/bin/env bash
# Fill missing cells:
#   1) Per-station Classify core budget {5,10,15,20} at thr=1 (STEAD 580)
#   2) Annotate thr=8 for PhaseNet + PhaseNetLight (STEAD 580, 20 cores)
set -u
cd "$(dirname "$0")/../.."
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate rapid

ROOT_FULL=results/iso_full_benchmark
ROOT_ISO=results/fair_benchmark_iso
LOG=$ROOT_ISO/iso_classify_core_annotate_thr8.log
mkdir -p "$ROOT_FULL" "$ROOT_ISO"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
BUDGETS=(5 10 15 20)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

classify_core() {
  local model=$1 c=$2 ins; ins=$(insamples "$model")
  local cores; cores=$(seq -s, 0 $((c-1)))
  note "CLASSIFY core / $model / 580st c${c} thr1 (in=$ins)"
  python3 benchmarks/fair/run_fair_trial.py \
    --method classify --dataset stead --n-stations 580 --model "$model" \
    --device cpu --n-cpus "$c" --core-list "$cores" --torch-threads 1 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 3 --tag "c${c}_thr1" --results-root "$ROOT_FULL/native" --resume >> "$LOG" 2>&1
}

annotate_thr8() {
  local model=$1 ins; ins=$(insamples "$model")
  local cores; cores=$(seq -s, 0 19)
  note "ANNOTATE thr8 / $model / 580st (in=$ins)"
  # Write to both trees so fig1/fig2 loaders see iso_thr8
  for ROOT in "$ROOT_ISO/native" "$ROOT_FULL/native"; do
    python3 benchmarks/fair/run_fair_trial.py \
      --method annotate --dataset stead --n-stations 580 --model "$model" \
      --device cpu --n-cpus 20 --core-list "$cores" --torch-threads 8 \
      --in-samples "$ins" --overlap-samples 0 --dtype fp32 --batch-size 256 \
      --repeats 3 --tag "iso_thr8" --results-root "$ROOT" --resume >> "$LOG" 2>&1
  done
}

note "FILL START"
# Annotate thr8 first (fast)
for M in PhaseNet PhaseNetLight; do
  annotate_thr8 "$M"
done
# Classify core budget (includes re-running c20 for a uniform c*_thr1 series)
for M in "${MODELS[@]}"; do
  for C in "${BUDGETS[@]}"; do
    classify_core "$M" "$C"
  done
done
note "FILL DONE"
