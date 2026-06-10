#!/usr/bin/env bash
# Daily babysitter for the fair-benchmark v4 matrix + follow-on sweeps.
#
# Logic (mirrors the in-session daily check):
#   * scheduler running           -> log progress, do nothing
#   * scheduler dead, matrix left -> clean orphans, resume the main scheduler
#   * main matrix complete        -> launch scripts/run_latency_sweep.sh once
#   * latency sweep complete      -> launch scripts/run_oversub_sweep.sh once
#   * everything complete         -> log and exit
#
# Installed in crontab (daily); remove with `crontab -e`. Log: results/fair_benchmark/babysit.log
set -uo pipefail
cd /home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/RAPID

log() { echo "[$(date -Is)] $*"; }

if pgrep -f "run_fair_scheduler.py" >/dev/null 2>&1; then
    if pgrep -af "run_fair_scheduler.py" | grep -q -- "--family oversub"; then
        log "oversub sweep running: $(grep -o '\[done [0-9]*/[0-9]*\]' results/fair_benchmark/oversub_sweep.log 2>/dev/null | tail -1)"
    elif pgrep -af "run_fair_scheduler.py" | grep -q -- "--family streaming"; then
        log "latency sweep running: $(grep -o '\[done [0-9]*/[0-9]*\]' results/fair_benchmark/latency_sweep.log 2>/dev/null | tail -1)"
    else
        log "main matrix running: $(grep -o '\[done [0-9]*/[0-9]*\]' results/fair_benchmark/scheduler.log 2>/dev/null | tail -1)"
    fi
    exit 0
fi

# No scheduler alive: clean up any orphaned trial workers before acting.
# Bracket trick avoids self-match; [ost] covers scheduler/trial/stream/orch runners.
pkill -f "run_fair_[ost]" 2>/dev/null
sleep 5

main_line=$(python3 scripts/run_fair_scheduler.py --total-cpus 120 --num-gpus 2 --dry-run 2>/dev/null | grep "Matrix:" | head -1)
to_run=$(echo "$main_line" | grep -o '[0-9]* to run' | grep -o '[0-9]*' || echo "")
if [ -z "$to_run" ]; then
    log "ERROR: could not determine main-matrix state ($main_line); doing nothing"
    exit 1
fi
if [ "$to_run" -gt 0 ]; then
    log "main scheduler not running with $to_run trials left -> resuming"
    nohup python3 scripts/run_fair_scheduler.py --total-cpus 120 --num-gpus 2 >> results/fair_benchmark/scheduler.log 2>&1 &
    exit 0
fi

sweep_line=$(./scripts/run_latency_sweep.sh --dry-run 2>/dev/null | grep "Matrix:" | head -1)
sweep_to_run=$(echo "$sweep_line" | grep -o '[0-9]* to run' | grep -o '[0-9]*' || echo "")
if [ -z "$sweep_to_run" ]; then
    log "ERROR: could not determine latency-sweep state ($sweep_line); doing nothing"
    exit 1
fi
if [ "$sweep_to_run" -gt 0 ]; then
    log "main matrix COMPLETE -> launching back-to-back latency sweep ($sweep_to_run trials)"
    nohup ./scripts/run_latency_sweep.sh >> results/fair_benchmark/latency_sweep.log 2>&1 &
    exit 0
fi

oversub_line=$(./scripts/run_oversub_sweep.sh --dry-run 2>/dev/null | grep "Matrix:" | head -1)
oversub_to_run=$(echo "$oversub_line" | grep -o '[0-9]* to run' | grep -o '[0-9]*' || echo "")
if [ -z "$oversub_to_run" ]; then
    log "ERROR: could not determine oversub-sweep state ($oversub_line); doing nothing"
    exit 1
fi
if [ "$oversub_to_run" -gt 0 ]; then
    log "main matrix + latency sweep COMPLETE -> launching oversubscription sweep ($oversub_to_run trials)"
    nohup ./scripts/run_oversub_sweep.sh >> results/fair_benchmark/oversub_sweep.log 2>&1 &
else
    log "main matrix, latency sweep, and oversub sweep all COMPLETE; nothing to do (this cron entry can be removed)"
fi
