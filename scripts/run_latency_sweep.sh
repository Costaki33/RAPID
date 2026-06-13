#!/usr/bin/env bash
# Back-to-back warm-actor latency sweep (streaming family, no wall-clock pacing).
#
# Measures cold-feed (feed 0, first-call overhead) and warm-feed (feeds 1..N-1,
# steady state) latency on a kept-alive Model-Actor pool. With
# --stream-interval-s 0 each feed is submitted the moment the previous one
# completes, so a trial repeat takes seconds instead of >= n_feeds * 60s --
# the whole sweep finishes in hours, not days.
#
# Slim matrix (override by appending scheduler flags):
#   dataset stead only | both station counts (250, 580) | batch size 256
#   CPU march 5/11/20 (+ the same as GPU host cores) | 8 feeds | 3 repeats
#   both strategies: stream_modelactor (SeisBench classify in actors) and
#   stream_modelactor_slipstream (lean PyTorch, full precision sweep)
#
# Results: results/fair_benchmark/streaming/... (schema v3, with a "latency"
# section: cold_feed_total_s / warm_feed_mean_s aggregated across repeats).
# Resume-safe: re-run this script after any stop.
#
# Usage:
#   ./scripts/run_latency_sweep.sh                  # foreground
#   nohup ./scripts/run_latency_sweep.sh >> results/fair_benchmark/latency_sweep.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

# The FCFS scheduler assumes it owns the core blocks; never run two at once.
if pgrep -f "run_fair_scheduler.py" >/dev/null 2>&1; then
    echo "ERROR: a run_fair_scheduler.py is already running (the main matrix?)."
    echo "Run the latency sweep after it finishes, or stop it first:"
    echo "  pkill -f 'run_fair_scheduler\\.py' && pkill -f 'run_fair_[ost]'"
    exit 1
fi

exec python3 scripts/run_fair_scheduler.py \
    --family streaming \
    --stream-interval-s 0 \
    --stream-feeds 8 \
    --stream-repeats 3 \
    --datasets stead \
    --batch-sizes 256 \
    --cpu-grid 5,10,15,20 \
    --total-cpus 120 \
    --num-gpus 2 \
    "$@"
