#!/usr/bin/env bash
# Consolidate the existing results/fair_benchmark_iso/ trees into the new master
# results/iso_full_benchmark/ folder, mapped into the matching family subfolder.
#
# Idempotent (rsync -a, incremental): safe to re-run, e.g. after the in-progress
# iso re-runs (txed_native/, precision/, batchsweep/, corebudget/) finish, to pull
# their completed trials in too. Nothing is deleted from the source.
#
# Path layout is preserved because every trial driver writes
#   <results-root>/<method-or-orchestration-or-streaming>/<dataset>/<Nst>/<model>/<tag>/
# so legacy tags (iso_thr1, ...) simply coexist beside the new grid's tags
# (c20_thr1, ...) inside the same model directory.
set -u
cd "$(dirname "$0")/../.."
SRC=results/fair_benchmark_iso
DST=results/iso_full_benchmark
mkdir -p "$DST"/{native,orch,oversub,stream}

cp_tree() {  # src_subdir dst_subdir
  local s="$SRC/$1" d="$DST/$2"
  [ -d "$s" ] || { echo "skip (missing): $s"; return; }
  mkdir -p "$d"
  rsync -a "$s/" "$d/"
  echo "copied $s/ -> $d/"
}

# native family (single-process): thread sweep + txed + precision + batch sweep
cp_tree native                  native
cp_tree txed_native             native
cp_tree precision               native
cp_tree batchsweep              native
cp_tree corebudget/annotate     native/annotate

# orchestration cold-start (+ corebudget Model-Actor scaling)
cp_tree orch/orchestration      orch/orchestration
cp_tree corebudget/orchestration orch/orchestration

# oversubscription
cp_tree oversub/orchestration   oversub/orchestration

# warm streaming head-to-head
cp_tree h2h/streaming           stream/streaming

echo "consolidation done."
