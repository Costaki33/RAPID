#!/usr/bin/env bash
# Download-only STEAD via SeisBench. Resumes an in-progress .partial if present.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rapid
mkdir -p "$HOME/.seisbench/datasets/stead"
cd "$HOME/RAPID"

echo "=== $(date -Ins) STEAD download start ==="
python - <<'PY'
import seisbench.data as sbd
# wait_for_file=True attaches to an in-progress download instead of failing on .partial
ds = sbd.STEAD(download_kwargs={"progress_bar": True}, wait_for_file=True)
print(f"STEAD ready: n={len(ds)}")
print(ds)
PY
echo "=== $(date -Ins) STEAD download done ==="
ls -lh "$HOME/.seisbench/datasets/stead/"
