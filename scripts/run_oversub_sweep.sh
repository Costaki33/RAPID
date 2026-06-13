#!/usr/bin/env bash
# Oversubscription sweep: how does requesting MORE concurrent actors/tasks than
# cores behave, with RAM (CPU mode) / VRAM (GPU mode) as the real constraint?
#
# eqcctpro never binds actors or Ripper tasks to CPUs (Ray num_cpus=0; only the
# trial's affinity mask limits cores), so concurrency can exceed the core
# budget until the memory caps bind. This sweep maps that curve:
#
#   concurrency = {0.25x, 0.5x, 1x, 2x, 3x, 4x} of the core budget
#   core budgets 5, 10, 15, 20 | CPU-only AND GPU
#   strategies: modelactor + ripper, classify() AND slipstream variants
#   precisions: fp32/fp16/bf16 (PN family), fp32/bf16 (EQT) for slipstream
#   all 4 models | one canonical window regime per model | bs 256
#   dataset stead, 580 stations | 3 repeats
#
# Each repeat records the REQUESTED concurrency and the ACHIEVED pool
# (n_modelactors), so RAM/VRAM capping is visible in the data. The slipstream-CPU
# in-flight clamp is bypassed when --concurrency is explicit, and VRAM/RAM-cap-
# redundant trials self-skip (result.json skipped=true) to save compute.
#
# SINGLE GPU (--num-gpus 1): the GPU half runs strictly one trial at a time on
# one physical GPU, so each oversubscription point owns the full GPU's VRAM with
# no co-tenant -- the cleanest measurement of concurrency-vs-VRAM, and it closes
# the dedup race (every redundant high-multiplier trial is skipped). GPU1 stays
# idle during this phase by design.
#
# Results: results/fair_benchmark/oversub/orchestration/... (never mixed with
# the main matrix). Resume-safe: re-run this script after any stop.
#
# Usage:
#   ./scripts/run_oversub_sweep.sh                  # foreground
#   nohup ./scripts/run_oversub_sweep.sh >> results/fair_benchmark/oversub_sweep.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

# The FCFS scheduler assumes it owns the core blocks; never run two at once.
if pgrep -f "run_fair_scheduler.py" >/dev/null 2>&1; then
    echo "ERROR: a run_fair_scheduler.py is already running (main matrix or latency sweep?)."
    echo "Run the oversubscription sweep after it finishes, or stop it first:"
    echo "  pkill -f 'run_fair_scheduler\\.py' && pkill -f 'run_fair_[ost]'"
    exit 1
fi

exec python3 scripts/run_fair_scheduler.py \
    --family oversub \
    --oversub-cpu-grid 5,10,15,20 \
    --oversub-multipliers 0.25,0.5,1,2,3,4 \
    --oversub-repeats 3 \
    --datasets stead \
    --stations 580 \
    --total-cpus 120 \
    --num-gpus 1 \
    "$@"
