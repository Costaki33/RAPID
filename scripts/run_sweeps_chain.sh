#!/usr/bin/env bash
# Event-driven chain for the follow-on sweeps: run the latency sweep to
# completion, then IMMEDIATELY run the oversubscription sweep -- without waiting
# for the once-daily benchmark_babysit.sh. Detached + resume-safe + idempotent.
#
# Behaviour:
#   1. If a latency/oversub sweep scheduler is already running, wait for it.
#   2. Run the latency sweep to completion (resume; fast no-op if already done).
#   3. Run the oversub sweep to completion (resume).
# Each launcher's guard refuses to start while another scheduler is alive, so the
# two phases are strictly sequential. The daily babysitter remains the safety net
# if this process ever dies.
#
# Usage (detached):
#   setsid bash -c './scripts/run_sweeps_chain.sh' < /dev/null &
# Log: results/fair_benchmark/sweeps_chain.log
set -uo pipefail
cd /home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/RAPID
log() { echo "[$(date -Is)] chain: $*"; }

sweep_running() {
    # True while a streaming/oversub scheduler is alive (pgrep -f on the script
    # name only matches real schedulers, not this bash chain).
    pgrep -af "run_fair_scheduler.py" 2>/dev/null | grep -Eq -- "--family (streaming|oversub)"
}

to_run() {  # $1 = streaming|oversub ; echoes the number of trials left to run
    if [ "$1" = "streaming" ]; then
        ./scripts/run_latency_sweep.sh --dry-run 2>/dev/null | grep -o '[0-9]* to run' | grep -o '[0-9]*'
    else
        ./scripts/run_oversub_sweep.sh --dry-run 2>/dev/null | grep -o '[0-9]* to run' | grep -o '[0-9]*'
    fi
}

log "started; waiting for any in-flight sweep scheduler to finish..."
while sweep_running; do sleep 60; done

# Phase 1: latency sweep to completion.
left=$(to_run streaming)
if [ "${left:-0}" -gt 0 ]; then
    log "latency sweep: $left trials left -> running to completion"
    ./scripts/run_latency_sweep.sh >> results/fair_benchmark/latency_sweep.log 2>&1
    log "latency sweep finished (rc=$?)"
else
    log "latency sweep already complete"
fi
while sweep_running; do sleep 30; done   # belt-and-suspenders

# Phase 2: oversub sweep to completion.
left=$(to_run oversub)
if [ "${left:-0}" -gt 0 ]; then
    log "oversub sweep: $left trials left -> running to completion"
    ./scripts/run_oversub_sweep.sh >> results/fair_benchmark/oversub_sweep.log 2>&1
    log "oversub sweep finished (rc=$?)"
else
    log "oversub sweep already complete"
fi
log "ALL SWEEPS COMPLETE. Benchmark fully done."
