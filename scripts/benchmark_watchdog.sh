#!/usr/bin/env bash
# Lightweight watchdog for the fair benchmark -- runs FREQUENTLY (every 30 min
# via cron), independent of the once-daily benchmark_babysit.sh. It only does
# two cheap, safe things and never starts/stops the scheduler.
#
# Guard 1 -- wedged-trial watchdog: a trial repeat that deadlocks (e.g. a Ray
#   cluster wedged by a transient disk-full) keeps its driver's subprocess.run()
#   blocked forever, so the scheduler waits on that slot indefinitely while
#   throughput craters. The scheduler PROCESS stays alive, so a liveness check
#   never catches it. We kill any per-repeat worker older than the cutoff (the
#   slowest legitimate trial -- EQT-NC classify, 580 st -- runs well under
#   45 min), plus the orphaned Ray cluster it left behind. The driver's
#   subprocess.run() then returns non-zero, the slot frees, and the now-incomplete
#   trial is re-dispatched / re-run on resume.
#
# Guard 2 -- stale Ray temp cleanup: each orchestration trial spins up a Ray
#   session that drops logs/spill under /tmp/ray; across thousands of trials this
#   can fill the root filesystem and wedge in-flight trials. Remove session dirs
#   untouched for >2 h (far longer than any single trial, so active sessions are
#   never deleted).
#
# Log: results/fair_benchmark/watchdog.log
set -uo pipefail
cd /home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/RAPID
log() { echo "[$(date -Is)] $*"; }

WEDGE_CUTOFF_SECONDS=${WEDGE_CUTOFF_SECONDS:-2700}   # 45 min

# Guard 1: kill stale repeat workers + their orphaned Ray cluster.
wedged=$(ps -eo pid,etimes,args | awk -v c="$WEDGE_CUTOFF_SECONDS" \
    '$0 ~ /run_fair_(orch_)?trial\.py/ && $0 ~ /--repeat-index/ && $2+0 > c {print $1}')
if [ -n "$wedged" ]; then
    log "WEDGE: killing stale trial repeat workers (> $((WEDGE_CUTOFF_SECONDS/60)) min): $(echo $wedged | tr '\n' ' ')"
    kill -9 $wedged 2>/dev/null
    orphans=$(ps -eo pid,etimes,args | awk \
        '$2+0 > 3600 && ($0 ~ /raylet/ || $0 ~ /gcs_server/ || $0 ~ /ray::/ || $0 ~ /plasma/ || $0 ~ /RuntimeEnvAgent/ || $0 ~ /DashboardAgent/) {print $1}')
    if [ -n "$orphans" ]; then
        log "WEDGE: reaping orphaned Ray processes (>1 h): $(echo $orphans | tr '\n' ' ')"
        kill -9 $orphans 2>/dev/null
    fi
fi

# Guard 2: prune stale Ray session dirs from /tmp.
freed_before=$(df -P /tmp | awk 'NR==2{print $4}')
find /tmp/ray -maxdepth 1 -name 'session_*' -type d -mmin +120 -exec rm -rf {} + 2>/dev/null
freed_after=$(df -P /tmp | awk 'NR==2{print $4}')
if [ "${freed_after:-0}" -gt "${freed_before:-0}" ]; then
    log "RAY CLEANUP: freed $(( (freed_after - freed_before) / 1024 )) MB from /tmp/ray stale sessions"
fi

# Disk pressure warning (does not act, just records).
for fs in / /home; do
    pct=$(df -P "$fs" | awk 'NR==2{gsub("%","",$5); print $5}')
    [ "${pct:-0}" -ge 92 ] && log "WARNING: $fs at ${pct}% -- disk pressure risks wedging trials"
done
exit 0
