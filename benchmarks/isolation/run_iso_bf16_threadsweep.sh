#!/usr/bin/env bash
# Isolated Slipstream-BF16 thread sweep vs Annotate (fair cold-start comparison).
# STEAD 580 only, EQT family, threads {1,2,4,8}, 3 repeats, sequential/idle.
set -u
cd "$(dirname "$0")/../.."
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
CORES=$(seq -s, 0 19)
ROOT=results/fair_benchmark_iso
OUT=$ROOT/bf16_threadsweep
LOG=$ROOT/iso_bf16_threadsweep.log
mkdir -p "$OUT"
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

run_bf16() {
  local model=$1 thr=$2 ins=6000
  note "BF16 slipstream / $model / 580st thr$thr"
  python3 benchmarks/fair/run_fair_trial.py \
    --method slipstream --dataset stead --n-stations 580 --model "$model" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads "$thr" \
    --in-samples "$ins" --overlap-samples 0 --dtype bf16 --batch-size 256 \
    --repeats 3 --tag "iso_bf16_thr${thr}" --results-root "$OUT" --resume >> "$LOG" 2>&1
}

note "ISO BF16 THREAD SWEEP START"
for M in EQTransformer EQT-NC; do
  for T in 1 2 4 8; do
    run_bf16 "$M" "$T"
  done
done
note "ISO BF16 THREAD SWEEP DONE"
