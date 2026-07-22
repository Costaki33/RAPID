#!/usr/bin/env bash
# Isolated re-measurement of the OVERSUBSCRIPTION sweep (absolute times).
# Strictly sequential -- one trial at a time, alone on the box -- so the
# concurrency curve (runtime vs actors-per-core) carries trustworthy absolute
# times, not just a trend. Mirrors the scheduler's oversub config:
#   strategies modelactor(fp32) + modelactor_slipstream(bf16),
#   cores {5,10,15,20}, multipliers {0.25,0.5,1,2,3,4} ascending,
#   conc = round-half-up(mult*cores) floored at 1, capped at stations,
#   --dedup-vram-capped (high mults that duplicate a VRAM/RAM-capped pool skip).
# Ascending multipliers are REQUIRED so the cap-establishing sibling runs first.
# STEAD 580 (headline); uniform native window per model, overlap 0.
set -u
cd "$(dirname "$0")/../.."
CORES_ALL=$(seq -s, 0 19)
ROOT=results/fair_benchmark_iso/oversub
LOG=results/fair_benchmark_iso/isolation_oversub.log
mkdir -p "$ROOT"
MODELS=(PhaseNet PhaseNetLight EQTransformer EQT-NC)
CORES=(5 10 15 20)
MULTS=(0.25 0.5 1 2 3 4)          # ASCENDING -- do not reorder
STATIONS=(580 250)
insamples() { case "$1" in PhaseNet|PhaseNetLight) echo 3001;; *) echo 6000;; esac; }
conc() { python3 -c "import sys,math;m,c,n=float(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]);print(max(1,min(int(math.floor(m*c+0.5)),n)))" "$1" "$2" "$3"; }
mtag() { python3 -c "import sys;m=float(sys.argv[1]);print(str(int(m)) if m==int(m) else ('%g'%m).replace('.','p'))" "$1"; }
note() { echo "=== $(date +%H:%M:%S)  $* ===" | tee -a "$LOG"; }

cell() {  # st strategy dtype device model cores mult
  local st=$1 strat=$2 dt=$3 dev=$4 model=$5 c=$6 mult=$7
  local ins; ins=$(insamples "$model")
  local cc; cc=$(conc "$mult" "$c" "$st")
  local mt; mt=$(mtag "$mult")
  local cores; cores=$(seq -s, 0 $((c-1)))      # pin to exactly c cores
  local btag=""; [ "$strat" = "modelactor_slipstream" ] && btag="--slipstream-batch-size 256"
  note "OVERSUB ${st}st $strat/$dt $dev $model cpu$c x${mult}->${cc}act"
  python3 benchmarks/fair/run_fair_orch_trial.py \
    --strategy "$strat" --dataset stead --n-stations "$st" --model "$model" \
    --device "$dev" --n-cpus "$c" --core-list "$cores" --concurrency "$cc" \
    --in-samples "$ins" --overlap-samples 0 --dtype "$dt" $btag \
    --repeats 2 --tag "iso_ov_${st}_${dev}_cpu${c}_c${mt}x" \
    --results-root "$ROOT" --resume --dedup-vram-capped >> "$LOG" 2>&1
}

note "ISOLATION OVERSUB START"
# ---- 580: full grid (both strategies, both devices) ----
for SPEC in "modelactor fp32" "modelactor_slipstream bf16"; do
  set -- $SPEC; strat=$1; dt=$2
  for DEV in cpu gpu; do
    for M in "${MODELS[@]}"; do
      for C in "${CORES[@]}"; do
        for MULT in "${MULTS[@]}"; do      # ascending -> dedup sees the cap first
          cell 580 "$strat" "$dt" "$DEV" "$M" "$C" "$MULT"
        done
      done
    done
  done
done
# ---- 250: spot-check that the concurrency curve generalizes (modelactor, CPU only) ----
for M in "${MODELS[@]}"; do
  for C in "${CORES[@]}"; do
    for MULT in "${MULTS[@]}"; do
      cell 250 modelactor fp32 cpu "$M" "$C" "$MULT"
    done
  done
done
note "ISOLATION OVERSUB DONE"
