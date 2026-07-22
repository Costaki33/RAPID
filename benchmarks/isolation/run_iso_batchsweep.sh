#!/usr/bin/env bash
# Isolated batch-size sweep (TIMING), STEAD.
#
# Why: the paper claims annotate()/Slipstream were swept over batch sizes to pick
# the runtime-minimizing configuration. That is a LATENCY claim, so it must come
# from the strictly-sequential isolated protocol (one trial at a time, idle box,
# torch threads = 1, 20-core allocation). Batches {64,128,256,512}, 3 repeats.
#
# annotate (SeisBench batched) + Slipstream FP32 (RAPID lean forward) are the two
# batched single-process methods the original sweep covered. Per-model native
# window, overlap 0. DEDICATED results subtree (batchsweep/) so the fixed-256 iso
# native thread-sweep tables are untouched.
set -u
cd "$(dirname "$0")/../.."
CORES=$(seq -s, 0 19)
ROOT=results/fair_benchmark_iso
LOG=$ROOT/iso_batchsweep.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
BATCHES=(64 128 256 512)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

batched() {  # method model stations batch
  local meth=$1 model=$2 st=$3 bs=$4 ins; ins=$(insamples "$model")
  note "BATCHSWEEP $meth / $model / ${st}st / b${bs} (in=$ins)"
  python3 benchmarks/fair/run_fair_trial.py \
    --method "$meth" --dataset stead --n-stations "$st" --model "$model" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 --batch-size "$bs" \
    --repeats 3 --tag "iso_thr1_b${bs}" --results-root "$ROOT/batchsweep" --resume >> "$LOG" 2>&1
}

note "ISO BATCHSWEEP START"
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    for METH in annotate slipstream; do
      for BS in "${BATCHES[@]}"; do batched "$METH" "$M" "$ST" "$BS"; done
    done
  done
done
note "ISO BATCHSWEEP DONE"
