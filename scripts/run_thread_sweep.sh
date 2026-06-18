#!/usr/bin/env bash
# Thread-sensitivity sweep for the single-process CPU baselines.
#
# For these seismic models on CPU, torch intra-op threading does NOT convert
# cores into throughput: neutral-to-harmful for batched annotate/slipstream,
# catastrophic for per-station classify() (tiny per-call tensors x N threads =
# pure pool overhead). This sweep measures the full curve so the paper reports
# each method at its measured optimum -- and documents the oversubscription trap.
#
# Fixed: STEAD 580 st, 20 cores (affinity 40-59 to avoid other jobs on 0-19),
# fp32. Swept: torch intra/inter-op threads in {1,2,4,8,16,20}. 1 repeat per cell
# (the thread effect is ~75x, dwarfing run-to-run noise); the chosen optimum is
# re-run at full repeats separately for the reported baseline.
set -u
cd "$(dirname "$0")/.."
CORES=$(seq -s, 40 59)
ROOT=results/fair_benchmark_threadsweep
LOG=$ROOT/threadsweep.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
THREADS=(1 2 4 8 16 20)

cell() {  # method  model  threads
  echo "=== $(date +%H:%M:%S)  $1 / $2 / thr$3 ===" | tee -a "$LOG"
  python3 scripts/run_fair_trial.py \
    --method "$1" --dataset stead --n-stations 580 --model "$2" \
    --device cpu --n-cpus 20 --core-list "$CORES" --torch-threads "$3" \
    --dtype fp32 --repeats 1 --tag "thr$3" \
    --results-root "$ROOT" --resume >> "$LOG" 2>&1
}

# Cheap methods first (batched: ~seconds), then classify (catastrophic at high
# threads for heavy models), low threads first so the curve fills in early.
for METH in annotate slipstream classify; do
  for M in "${MODELS[@]}"; do
    for T in "${THREADS[@]}"; do
      cell "$METH" "$M" "$T"
    done
  done
done
echo "=== THREAD SWEEP DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
