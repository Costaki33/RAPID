#!/usr/bin/env bash
# Run the CPU-only portions of the latency + oversubscription sweeps on the
# idle CPU pool (cores 40-119) WHILE the main matrix grinds its GPU-only tail
# on the two GPU blocks (cores 0-39).
#
# Why this is safe: run_fair_scheduler.py always reserves the GPU host-core
# blocks [0, num_gpus*gpu_core_block) and draws CPU trials exclusively from
# the pool above them. With --no-sweep-gpu this scheduler builds CPU-only
# trials, so it allocates only from [40,120) -- disjoint from the main
# scheduler's GPU blocks by construction. We additionally verify below that
# the main matrix has NO pending CPU trials before starting.
#
# Later, when the babysitter chains the full sweeps (CPU+GPU), the CPU trials
# finished here are skipped by the resume logic and only the GPU halves run.
#
# Resume-safe: re-run this script after any stop.
set -uo pipefail
cd "$(dirname "$0")/.."

log() { echo "[$(date -Is)] $*"; }

# Refuse if a sweep scheduler is already running (main-matrix scheduler is OK).
if pgrep -af "run_fair_scheduler.py" | grep -qE -- "--family (streaming|oversub)"; then
    log "ERROR: a sweep scheduler is already running; nothing to do."
    exit 1
fi

# Safety: the main matrix must have zero pending CPU trials, otherwise the
# main scheduler could dispatch onto the CPU pool we are about to use.
pending_cpu=$(python3 scripts/run_fair_scheduler.py --total-cpus 120 --num-gpus 2 --dry-run 2>/dev/null \
    | grep -E "^  " | grep -cv " GPU$")
if [ "${pending_cpu:-1}" -gt 0 ]; then
    log "ERROR: main matrix still has $pending_cpu pending CPU trials; refusing to share the CPU pool."
    exit 1
fi
log "main matrix tail is GPU-only; CPU pool [40,120) is free. Starting CPU-only sweeps."

log "Phase 1/2: latency sweep (CPU-only portion)"
python3 scripts/run_fair_scheduler.py \
    --family streaming --stream-interval-s 0 --stream-feeds 8 --stream-repeats 3 \
    --datasets stead --batch-sizes 256 --cpu-grid 5,11,20 \
    --total-cpus 120 --num-gpus 2 --no-sweep-gpu
log "latency sweep CPU portion done (rc=$?)"

log "Phase 2/2: oversubscription sweep (CPU-only portion)"
python3 scripts/run_fair_scheduler.py \
    --family oversub --oversub-cpu-grid 5,10,15,20 --oversub-multipliers 0.25,0.5,1,2,3,4 \
    --oversub-repeats 3 --datasets stead --stations 580 \
    --total-cpus 120 --num-gpus 2 --no-sweep-gpu
log "oversub sweep CPU portion done (rc=$?)"
log "CPU-only sweep phases complete."
