#!/usr/bin/env bash
# XPS / consumer-laptop validation for the FASTEST paper-relevant methods only.
#
# Intentionally excluded: Ripper, 2-GPU Model-Actor, Per-Station Classify,
# cold-start orchestration grids, oversubscription sweeps, FP16 EQTransformer.
#
# Included (warm streaming only):
#   1. stream_classify_batched  -- kept-warm Network-Batched Classify (SeisBench picks)
#   2. stream_annotate         -- kept-warm Annotate probability traces
#   3. stream_modelactor       -- CPU/GPU persistent Classify actors (fastest RAPID path)
#   4. stream_modelactor_slipstream bf16 -- secondary RAPID precision path
#
# Host CPU budgets match the paper sweep shape (5/10/15/20) but actor counts are
# capped for XPS memory/VRAM. See docs/XPS_AI_RUNBOOK.md for core isolation.
set -euo pipefail

cd "$(dirname "$0")/../.."

# ---------- operator overrides ----------
# REQUIRED: comma-separated logical CPU IDs, one hardware thread per selected P-core.
# Example after topology inspection: CORES=0,2,4,6,8
CORES="${CORES:-}"
RESULTS_ROOT="${RESULTS_ROOT:-results/xps_validation}"
LOG="${LOG:-$RESULTS_ROOT/xps_fastest.log}"
PHASE="${PHASE:-primary}"          # smoke | pilot | primary | extension
N_STATIONS="${N_STATIONS:-580}"
REPEATS="${REPEATS:-}"
N_FEEDS="${N_FEEDS:-8}"
CPU_GRID="${CPU_GRID:-}"           # e.g. "5 10" or "5 10 15 20"
GPU_ACTOR_CAP="${GPU_ACTOR_CAP:-2}"
CPU_ACTOR_CAP="${CPU_ACTOR_CAP:-10}"
MODELS_CSV="${MODELS_CSV:-}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NET_ROOT="${NET_ROOT:-data/seisbench_networks}"
TMP_DIR="${TMP_DIR:-$HOME/rapid_ray}"

mkdir -p "$RESULTS_ROOT" "$(dirname "$LOG")"

note() { echo "=== $(date -Ins)  $* ===" | tee -a "$LOG"; }
die()  { echo "ERROR: $*" | tee -a "$LOG" >&2; exit 1; }

[[ -n "$CORES" ]] || die "Set CORES to an explicit affinity list before running (see docs/XPS_AI_RUNBOOK.md)."

case "$PHASE" in
  smoke)
    MODELS=(PhaseNet)
    N_STATIONS=250
    REPEATS="${REPEATS:-1}"
    N_FEEDS=2
    CPU_GRID="${CPU_GRID:-5}"
    ;;
  pilot)
    MODELS=(PhaseNet EQTransformer)
    N_STATIONS=250
    REPEATS="${REPEATS:-1}"
    N_FEEDS=2
    CPU_GRID="${CPU_GRID:-5}"
    ;;
  primary)
    MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
    N_STATIONS="${N_STATIONS:-580}"
    REPEATS="${REPEATS:-5}"
    N_FEEDS="${N_FEEDS:-8}"
    # Full paper-shaped host budget sweep, but actor counts are capped below.
    CPU_GRID="${CPU_GRID:-5 10 15 20}"
    ;;
  extension)
    MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
    N_STATIONS=580
    REPEATS="${REPEATS:-10}"
    N_FEEDS=8
    CPU_GRID="${CPU_GRID:-5 10 15 20}"
    ;;
  *)
    die "Unknown PHASE=$PHASE (smoke|pilot|primary|extension)"
    ;;
esac

if [[ -n "$MODELS_CSV" ]]; then
  IFS=',' read -r -a MODELS <<< "$MODELS_CSV"
fi

read -r -a CORE_ARR <<< "${CORES//,/ }"
N_CORE_IDS=${#CORE_ARR[@]}
[[ "$N_CORE_IDS" -ge 5 ]] || die "CORES must list at least 5 logical CPUs; got $N_CORE_IDS ($CORES)"

in_samples() {
  case "$1" in
    PhaseNet|PhaseNetLight) echo 3001 ;;
    *) echo 6000 ;;
  esac
}
net_suffix() {
  case "$1" in
    PhaseNet|PhaseNetLight) echo "_w3001" ;;
    *) echo "" ;;
  esac
}

# Cap requested host CPUs to the number of isolated IDs supplied.
core_slice() {
  local n="$1"
  local out=()
  local i
  for ((i=0; i<n && i<N_CORE_IDS; i++)); do
    out+=("${CORE_ARR[$i]}")
  done
  (IFS=,; echo "${out[*]}")
}

run_one() {
  local strategy="$1" device="$2" model="$3" ncpus="$4" conc="$5" dtype="$6" thr="$7"
  local samples suffix cores tag thr_args=()
  samples=$(in_samples "$model")
  suffix=$(net_suffix "$model")
  cores=$(core_slice "$ncpus")
  tag="xps_${device}_c${ncpus}_a${conc}_${dtype}"
  if [[ -n "$thr" ]]; then
    thr_args=(--torch-threads "$thr")
    tag="${tag}_thr${thr}"
  fi

  note "$PHASE $strategy / $device / $model / ${N_STATIONS}st / c${ncpus} a${conc} $dtype thr=${thr:-auto}"
  python benchmarks/fair/run_fair_stream_trial.py \
    --strategy "$strategy" \
    --dataset stead \
    --n-stations "$N_STATIONS" \
    --model "$model" \
    --device "$device" \
    --n-cpus "$ncpus" \
    --gpu-id 0 \
    --core-list "$cores" \
    --concurrency "$conc" \
    "${thr_args[@]}" \
    --in-samples "$samples" \
    --net-suffix "$suffix" \
    --overlap-samples 0 \
    --dtype "$dtype" \
    --slipstream-batch-size "$BATCH_SIZE" \
    --repeats "$REPEATS" \
    --n-feeds "$N_FEEDS" \
    --feed-interval-s 0 \
    --tag "$tag" \
    --net-root "$NET_ROOT" \
    --results-root "$RESULTS_ROOT" \
    --tmp-dir "$TMP_DIR" \
    --resume 2>&1 | tee -a "$LOG"
}

note "XPS fastest-solutions START phase=$PHASE stations=$N_STATIONS repeats=$REPEATS feeds=$N_FEEDS grid=[$CPU_GRID] cores=$CORES"

for model in "${MODELS[@]}"; do
  for ncpus in $CPU_GRID; do
    [[ "$ncpus" -le "$N_CORE_IDS" ]] || {
      note "skip host budget c${ncpus}: only ${N_CORE_IDS} isolated IDs available"
      continue
    }

    # Native single-process baselines: concurrency=1.
    # Network-Batched Classify uses the measured 1-thread optimum.
    # Annotate uses min(8, ncpus) to stay near the measured 4-8 thread optimum.
    local_ann_thr="$ncpus"
    if (( local_ann_thr > 8 )); then local_ann_thr=8; fi
    run_one stream_classify_batched cpu "$model" "$ncpus" 1 fp32 1
    run_one stream_annotate           cpu "$model" "$ncpus" 1 fp32 "$local_ann_thr"

    # CPU Model-Actor: actors capped for RAM. Never request more actors than cores.
    cpu_conc=$ncpus
    if (( cpu_conc > CPU_ACTOR_CAP )); then cpu_conc=$CPU_ACTOR_CAP; fi
    run_one stream_modelactor            cpu "$model" "$ncpus" "$cpu_conc" fp32 ""
    run_one stream_modelactor_slipstream cpu "$model" "$ncpus" "$cpu_conc" bf16 ""

    # Single-GPU counterparts. Actor pool capped by laptop VRAM.
    gpu_conc=$ncpus
    if (( gpu_conc > GPU_ACTOR_CAP )); then gpu_conc=$GPU_ACTOR_CAP; fi
    run_one stream_classify_batched gpu "$model" "$ncpus" 1 fp32 1
    run_one stream_annotate         gpu "$model" "$ncpus" 1 fp32 "$local_ann_thr"
    run_one stream_modelactor       gpu "$model" "$ncpus" "$gpu_conc" fp32 ""
  done
done

note "XPS fastest-solutions DONE phase=$PHASE"
echo "Results under $RESULTS_ROOT"
echo "Log: $LOG"
