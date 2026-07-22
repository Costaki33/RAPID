#!/usr/bin/env bash
# Isolated CPU core-budget marching sweep {5,8,11,14,17,20} (TIMING), STEAD 580.
#
# Why: the paper claims a marching core budget of 5,8,11,14,17,20 for the
# native and orchestration families. That is a LATENCY-vs-cores claim, so it must
# come from the strictly-sequential isolated protocol. Two parts:
#
#   1. Cold-start Model-Actor (the meaningful scaling result: actors == cores),
#      CPU + GPU, 2 repeats. Ripper is NOT swept here -- it is the persistence
#      control and is already measured at 20 cores in results/.../orch/.
#   2. Single-process native annotate() at each budget (torch threads == cores,
#      pinned to that many cores), 3 repeats -- the honest "more cores do not help
#      a single process" companion. classify() is deliberately NOT swept across
#      budgets: at high thread counts it is catastrophic on the heavy models
#      (already documented by the iso thread sweep's naive-default anchor), so a
#      6-point classify sweep would cost hours to re-confirm a known blow-up.
#
# Per-model native window, overlap 0, fp32. DEDICATED results subtree (corebudget/).
set -u
cd "$(dirname "$0")/../.."
ROOT=results/fair_benchmark_iso
LOG=$ROOT/iso_corebudget.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
BUDGETS=(5 8 11 14 17 20)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

orch_ma() {  # device model cores
  local dev=$1 model=$2 c=$3 ins; ins=$(insamples "$model")
  local cores; cores=$(seq -s, 0 $((c-1)))
  note "COREBUDGET-ORCH modelactor / $dev / $model / 580st cpu${c} (in=$ins)"
  python3 benchmarks/fair/run_fair_orch_trial.py \
    --strategy modelactor --dataset stead --n-stations 580 --model "$model" \
    --device "$dev" --n-cpus "$c" --gpu-id 0 --core-list "$cores" --concurrency "$c" \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 \
    --repeats 2 --tag "iso_${dev}_580_cpu${c}" --results-root "$ROOT/corebudget" --resume >> "$LOG" 2>&1
}

native_annotate() {  # model cores
  local model=$1 c=$2 ins; ins=$(insamples "$model")
  local cores; cores=$(seq -s, 0 $((c-1)))
  note "COREBUDGET-NATIVE annotate / $model / 580st cpu${c} thr${c} (in=$ins)"
  python3 benchmarks/fair/run_fair_trial.py \
    --method annotate --dataset stead --n-stations 580 --model "$model" \
    --device cpu --n-cpus "$c" --core-list "$cores" --torch-threads "$c" \
    --in-samples "$ins" --overlap-samples 0 --dtype fp32 --batch-size 256 \
    --repeats 3 --tag "iso_cpu${c}_thr${c}" --results-root "$ROOT/corebudget" --resume >> "$LOG" 2>&1
}

note "ISO COREBUDGET START"
# ---- Part 1: cold-start Model-Actor scaling, CPU + GPU ----
for M in "${MODELS[@]}"; do
  for C in "${BUDGETS[@]}"; do
    for DEV in cpu gpu; do orch_ma "$DEV" "$M" "$C"; done
  done
done
# ---- Part 2: single-process annotate across the same budgets (CPU) ----
for M in "${MODELS[@]}"; do
  for C in "${BUDGETS[@]}"; do native_annotate "$M" "$C"; done
done
note "ISO COREBUDGET DONE"
