#!/usr/bin/env bash
set -euo pipefail
f=/home/cgs2528/.seisbench/datasets/stead/waveforms.hdf5.partial
ls -lh /home/cgs2528/.seisbench/datasets/stead/ || true
if [[ -f "$f" ]]; then
  s1=$(stat -c%s "$f")
  echo "size1=$s1"
  sleep 10
  s2=$(stat -c%s "$f")
  echo "size2=$s2"
  echo "delta_bytes=$((s2-s1)) rate_MBps=$(python3 -c "print(round(($s2-$s1)/10/1e6, 3))")"
else
  echo "no partial waveforms file"
fi
ps -ef | grep -E 'build_seisbench|wsl_bootstrap|wait_stead' | grep -v grep || true
