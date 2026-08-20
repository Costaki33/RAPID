#!/usr/bin/env bash
# Build STEAD networks with correct net-suffix naming, then leave ready for XPS runner.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

# Keep Windows edits in sync
cp -a /mnt/c/Users/cgs2528/Projects/RAPID/environment.yml "$HOME/RAPID/environment.yml"
cp -a /mnt/c/Users/cgs2528/Projects/RAPID/benchmarks/isolation/run_xps_fastest.sh "$HOME/RAPID/benchmarks/isolation/run_xps_fastest.sh"
cp -a /mnt/c/Users/cgs2528/Projects/RAPID/scripts "$HOME/RAPID/"

chmod +x scripts/detect_xps_cores.sh benchmarks/isolation/run_xps_fastest.sh
if CORES="$(bash scripts/detect_xps_cores.sh)"; then
  # Prefer ascending logical IDs for stable core_slice budgets
  CORES="$(echo "$CORES" | tr ',' '\n' | sort -n | paste -sd, -)"
else
  CORES="$(lscpu -e=CPU,CORE | awk 'NR>1 {if (!seen[$2]++) print $1}' | sort -n | paste -sd, -)"
fi
echo "$CORES" | tee "$HOME/RAPID/.xps_cores"
echo "CORES=$CORES"

build_one() {
  local n="$1" trim="$2" suffix="$3"
  local out="data/seisbench_networks/stead_${n}st${suffix}"
  if [[ -f "$out/manifest.json" ]]; then
    echo "exists $out"
    return 0
  fi
  echo "building $out ..."
  python examples/build_seisbench_network.py \
    --dataset stead --n-stations "$n" --require-s \
    --min-pick-sample 0 --max-pick-sample $((trim - 50)) \
    --trim-samples "$trim" \
    --net-suffix "$suffix" \
    --out-root data/seisbench_networks
}

mkdir -p data/seisbench_networks
build_one 250 3001 "_w3001"
build_one 250 6000 ""
build_one 580 3001 "_w3001"
build_one 580 6000 ""
ls -la data/seisbench_networks/
echo READY
