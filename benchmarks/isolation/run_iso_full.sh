#!/usr/bin/env bash
# Launch the FULL isolated benchmark (cost-reduced scope by default) into
# results/iso_full_benchmark/, strictly sequentially. Pauses the babysit/watchdog
# crons for the duration and restores them on exit (incl. Ctrl-C / kill).
#
# IMPORTANT: do NOT run this while any other isolated benchmark is still going
# (e.g. run_iso_reruns.sh). Concurrent trials destroy the contention-free timing
# guarantee. Let the other run finish, then run benchmarks/isolation/consolidate_iso.sh, then
# launch this.
#
# Usage:
#   benchmarks/isolation/run_iso_full.sh                 # all families, cost-reduced scope
#   benchmarks/isolation/run_iso_full.sh --family orch   # one family
#   benchmarks/isolation/run_iso_full.sh --full          # complete ideal grid (~15 days)
set -u
cd "$(dirname "$0")/../.."
ROOT=results/iso_full_benchmark
mkdir -p "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
CRON_BACKUP="$ROOT/crontab_backup_${STAMP}.txt"
LOG="$ROOT/run_iso_full_${STAMP}.log"
log() { echo "=== $(date '+%F %T')  $* ===" | tee -a "$LOG"; }

restore_cron() {
  if [ -f "$CRON_BACKUP" ]; then
    crontab "$CRON_BACKUP" && log "cron RESTORED" \
      || log "WARNING: cron restore FAILED; run: crontab $CRON_BACKUP"
  fi
}
trap restore_cron EXIT

if crontab -l > "$CRON_BACKUP" 2>/dev/null; then
  grep -vE 'benchmark_babysit\.sh|benchmark_watchdog\.sh' "$CRON_BACKUP" | crontab -
  log "cron PAUSED; backup at $CRON_BACKUP"
else
  log "no crontab found"; rm -f "$CRON_BACKUP"
fi

log "ISO FULL BENCHMARK START ($*)"
python3 benchmarks/isolation/iso_full_benchmark.py "$@" 2>&1 | tee -a "$LOG"
log "ISO FULL BENCHMARK DONE"
