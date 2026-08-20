#!/usr/bin/env bash
# Live STEAD download progress (file size vs ~84.9 GB target).
set -euo pipefail
TARGET=84900000000
f="$HOME/.seisbench/datasets/stead/waveforms.hdf5.partial"
donef="$HOME/.seisbench/datasets/stead/waveforms.hdf5"
while true; do
  if [[ -f "$donef" ]]; then
    ls -lh "$donef"
    echo "COMPLETE: waveforms.hdf5 present"
    exit 0
  fi
  if [[ -f "$f" ]]; then
    s=$(stat -c%s "$f")
    pct=$(python3 -c "print(round(100.0*$s/$TARGET, 1))")
    echo "$(date '+%H:%M:%S')  ${pct}%  $(numfmt --to=iec --suffix=B "$s") / ~79GiB"
  else
    echo "$(date '+%H:%M:%S')  waiting for partial file..."
  fi
  sleep 15
done
