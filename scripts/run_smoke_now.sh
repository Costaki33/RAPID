#!/usr/bin/env bash
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
cd "$HOME/RAPID"

# Sync scripts from Windows workspace
cp -a /mnt/c/Users/cgs2528/Projects/RAPID/scripts "$HOME/RAPID/"
cp -a /mnt/c/Users/cgs2528/Projects/RAPID/benchmarks/isolation/run_xps_fastest.sh "$HOME/RAPID/benchmarks/isolation/run_xps_fastest.sh"
cp -a /mnt/c/Users/cgs2528/Projects/RAPID/environment.yml "$HOME/RAPID/environment.yml"

chmod +x scripts/*.sh benchmarks/isolation/run_xps_fastest.sh
CORES="$(bash scripts/detect_xps_cores.sh | tr ',' '\n' | sort -n | paste -sd, -)"
echo "$CORES" | tee .xps_cores

python scripts/build_synthetic_smoke_network.py --n-stations 250 --trim-samples 3001 --net-suffix _w3001
test -f data/seisbench_networks/stead_250st_w3001/manifest.json

mkdir -p results/xps_validation/provenance
set +o pipefail
{
  date -Ins
  uname -a
  cat /proc/version || true
  lscpu
  free -h
  echo "CORES=$CORES"
  echo "GPU_ACTOR_CAP=1  # RTX 4050 Laptop 6141 MiB"
  echo "NOTE: smoke network is synthetic; replace with STEAD build after download"
  nvidia-smi -q 2>/dev/null | head -80 || true
  python - <<'PY'
import torch, ray, seisbench, numpy, obspy, sys
print("python", sys.version)
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
    print("bf16", torch.cuda.is_bf16_supported())
print("ray", ray.__version__)
print("seisbench", seisbench.__version__)
print("numpy", numpy.__version__)
print("obspy", obspy.__version__)
PY
} > results/xps_validation/provenance/environment.txt
set -o pipefail

export CORES
export GPU_ACTOR_CAP=1
export CPU_ACTOR_CAP=10
export PHASE=smoke
bash benchmarks/isolation/run_xps_fastest.sh
