#!/usr/bin/env bash
# Isolated TXED re-measurement (PICK QUALITY only).
#
# Why: the paper reports TXED only for pick quality (precision/recall/F1); TXED
# latency is never reported. Pick quality is deterministic w.r.t. hardware
# scheduling, so the cheapest iso-native source is the single-process native
# family run sequentially on an idle box. classify(), annotate(), and FP32
# Slipstream cover the three forward paths whose TXED quality the paper compares.
#
# Same regime as the STEAD iso run: per-model native window (PN/PNL 3001 over the
# 6000-sample net -> 2 windows/station, EQT/EQT-NC 6000), overlap 0, batch 256,
# torch threads = 1 (the measured optimum), 3 repeats. Written to a DEDICATED
# results subtree (txed_native/) so it never mixes into the STEAD native/** tables.
set -u
cd "$(dirname "$0")/../.."
CORES=$(seq -s, 0 19)
ROOT=results/fair_benchmark_iso
LOG=$ROOT/iso_txed.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

native() {  # method model stations
  local meth=$1 model=$2 st=$3 ins; ins=$(insamples "$model")
  note "TXED-NATIVE $meth / $model / ${st}st (in=$ins)"
  python3 benchmarks/fair/run_fair_trial.py \
    --method "$meth" --dataset txed --n-stations "$st" --model "$model" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads 1 \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 3 --tag "iso_thr1" --results-root "$ROOT/txed_native" --resume >> "$LOG" 2>&1
}

note "ISO TXED START"
for ST in 580 250; do
  for M in "${MODELS[@]}"; do
    for METH in classify annotate slipstream; do native "$METH" "$M" "$ST"; done
  done
done
note "ISO TXED DONE"
