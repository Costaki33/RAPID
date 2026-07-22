#!/usr/bin/env bash
# Master driver for the isolated re-runs (measurement-targeted minimal set).
#
# Runs the four iso re-run scripts STRICTLY SEQUENTIALLY (one trial at a time
# inside each, and one script after another) so every latency stays contention
# free. Each child script is --resume, so this whole driver is safe to re-launch
# after an interruption.
#
# Cron safety: benchmark_babysit.sh (daily) and benchmark_watchdog.sh (every
# 30 min, kills long workers + orphaned Ray) MUST NOT run during isolated timing.
# This driver snapshots the current crontab, installs a filtered crontab with
# those two jobs removed, and restores the snapshot on exit (including Ctrl-C /
# kill), so the watchdogs come back automatically when the run ends.
#
# Order: fastest first (TXED -> precision -> batchsweep -> corebudget).
set -u
cd "$(dirname "$0")/../.."
ROOT=results/fair_benchmark_iso
mkdir -p "$ROOT"
STAMP=$(date +%Y%m%d_%H%M%S)
CRON_BACKUP="$ROOT/crontab_backup_${STAMP}.txt"
DRIVER_LOG="$ROOT/iso_reruns_${STAMP}.log"

log() { echo "=== $(date '+%F %T')  $* ===" | tee -a "$DRIVER_LOG"; }

restore_cron() {
  if [ -f "$CRON_BACKUP" ]; then
    crontab "$CRON_BACKUP" && log "cron RESTORED from $CRON_BACKUP" \
      || log "WARNING: cron restore FAILED; restore manually with: crontab $CRON_BACKUP"
  fi
}
trap restore_cron EXIT

# ---- pause the watchdogs ----
if crontab -l > "$CRON_BACKUP" 2>/dev/null; then
  grep -vE 'benchmark_babysit\.sh|benchmark_watchdog\.sh' "$CRON_BACKUP" | crontab -
  log "cron PAUSED (babysit+watchdog removed); backup at $CRON_BACKUP"
else
  log "no crontab found; nothing to pause"
  rm -f "$CRON_BACKUP"
fi

log "ISO RE-RUNS START (minimal set)"
bash benchmarks/isolation/run_iso_txed.sh        && log "TXED done"        || log "TXED FAILED"
bash benchmarks/isolation/run_iso_precision.sh   && log "precision done"   || log "precision FAILED"
bash benchmarks/isolation/run_iso_batchsweep.sh  && log "batchsweep done"  || log "batchsweep FAILED"
bash benchmarks/isolation/run_iso_corebudget.sh  && log "corebudget done"  || log "corebudget FAILED"
log "ISO RE-RUNS DONE"
# cron restored by the EXIT trap
