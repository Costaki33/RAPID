# RAPID

RAPID (Resource-Aware Parallel Inference Dispatcher) is a toolkit for running
and benchmarking deep-learning seismic phase pickers on whole station networks.
It packages lean inference (Slipstream), Ray-based orchestration (Model-Actor
and Ripper), and a fair deployment benchmark so you can compare strategies on
the same waveforms and the same pick-quality scores.

## Background

This repository is part of a larger effort to enable real-time seismic phase
picking at the Texas Seismological Network (TexNet). The preliminary work,
[EQCCTPro](https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro), achieved
sub-11-second processing of 1-minute, 3-component waveforms from 228 stations
using persistent Ray model actors. That architecture became the backbone of
[SCMLPick](https://github.com/ut-beg-texnet/scmlpick), the SeisComP module
running in production at TexNet today.

RAPID is the next step of that line of work: the same orchestration ideas,
extended with Slipstream (lean reduced-precision inference) and a fair
deployment benchmark across models, devices, and scheduling strategies. The
central finding is that **how you parallelize the work matters more than the
device alone** — a pool of persistent, single-threaded workers (Model-Actor)
keeps a many-core CPU busy on heavy models where a single-process picker leaves
cores idle. Slipstream is a forward path that can run inside that pool; it is
not a competing orchestrator.

EQCCTPro as a standalone package is deprecated in favor of RAPID. Install and
import RAPID going forward; the orchestration code that began in EQCCTPro now
ships here (under `eqcctpro/` inside this repo, and via the `rapid` PyPI
package).

Supported SeisBench models: PhaseNet, PhaseNetLight, EQTransformer, and EQT-NC
(non-conservative EQTransformer). EQCCT is planned once it lands in SeisBench.

## What's in the repo

```
eqcctpro/           Orchestration (Model-Actor, Ripper, Slipstream actors)
rapid/              Lean inference backends and fair-benchmark machinery
examples/           Build synthetic networks and run a first pick
benchmarks/
  fair/             Fair deployment matrix, latency and oversubscription sweeps
  isolation/        Sequential, contention-free re-measurements
  analysis/         Summaries, pick-quality scoring, table generators
configs/            JSON configs for matrix / dtype sweeps
```

Runtime outputs (`results/`, `figures/`, `logs/`, `data/`) are created locally
and are not part of the published tree.

## Installation

From PyPI:

```bash
pip install rapid
pip install "rapid[orchestration]"   # Model-Actor / Ripper (needs Ray)
```

From a clone:

```bash
conda env create -f environment.yml
conda activate rapid
cd RAPID
pip install -e ".[orchestration]"
```

## Quick start: build a network and pick it

Build a synthetic station network from STEAD or TXED — choose how many
stations, how long each trace is, and where the catalog P and S picks must fall
so they sit inside your model window.

```bash
# 50 stations, 3001-sample traces, both P and S inside the window
python examples/build_seisbench_network.py \
    --dataset stead --n-stations 50 --require-s \
    --min-pick-sample 0 --max-pick-sample 2951 \
    --trim-samples 3001 \
    --out-root data/seisbench_networks
```

Useful knobs:

| Flag | What it does |
| --- | --- |
| `--n-stations` | Network size (traces are tiled if you ask for more than unique catalog events) |
| `--trim-samples` | Trace length written to disk (e.g. 3001 or 6000) |
| `--min-pick-sample` / `--max-pick-sample` | Keep catalog P and S inside this sample range |
| `--dataset stead\|txed` | Source catalog |
| `--seed` | Reproducible station sampling |

The builder writes per-station miniSEED plus a `manifest.json` with start/end
times and catalog pick sample indices for quality scoring.

### Orchestrated picking (Model-Actor or Ripper)

```python
from rapid import pick

# Persistent workers + SeisBench classify (recommended default)
pick(
    "data/seisbench_networks/<your_network>",
    "results/picks_ma_classify",
    model="EQTransformer",
    strategy="modelactor",
    forward="classify",
    n_workers=20,
)

# Same pool, Slipstream forward at BF16
pick(
    "data/seisbench_networks/<your_network>",
    "results/picks_ma_slipstream",
    model="EQTransformer",
    strategy="modelactor",
    forward="slipstream",
    dtype="bf16",
    n_workers=20,
)

# Ripper control (fresh task per station; slower cold start)
pick(..., strategy="ripper", forward="classify", n_workers=20)

# Use GPUs: pass physical device indices
pick(..., gpus=[0])          # one GPU
pick(..., gpus=[0, 1])       # two GPUs
```

Or from the shell:

```bash
python examples/pick_network.py \
    --input-dir data/seisbench_networks/<your_network> \
    --strategy modelactor --forward slipstream --dtype bf16 \
    --n-workers 16
```

`strategy` is how stations are scheduled (`modelactor` or `ripper`).
`forward` is how each worker computes (`classify` or `slipstream`).
`dtype` applies to Slipstream (`fp32`, `fp16`, `bf16`). Prefer BF16 for
EQTransformer; FP16 can overflow its attention padding sentinel.

### Single-process baselines (no Ray)

```python
from rapid import annotate, classify, slipstream

annotate("data/.../network", "results/ann", model="PhaseNet", device="cpu", batch_size=256)
classify("data/.../network", "results/cls", model="PhaseNet", device="cpu", torch_threads=1)
slipstream("data/.../network", "results/ss", model="PhaseNet", dtype="bf16", device="cuda:0")
```

These are the native paths the fair benchmark compares against Model-Actor.

## Benchmarking your machine

The fair benchmark puts every deployment strategy on the same waveforms, the
same timed stages, the same memory metric, and the same catalog pick scores.
You can sweep cores, GPUs, torch threads, batch sizes, and window regimes to
see how your hardware behaves.

```bash
# One native trial
python benchmarks/fair/run_fair_trial.py \
    --method slipstream --dataset stead --n-stations 250 \
    --model PhaseNet --device cpu --n-cpus 8 \
    --dtype bf16 --batch-size 256 --tag smoke

# One orchestration trial
python benchmarks/fair/run_fair_orch_trial.py \
    --strategy modelactor_slipstream --dataset stead --n-stations 250 \
    --model PhaseNet --device cpu --n-cpus 8 --dtype bf16 --tag smoke

# Full matrix (resume-safe; pin dedicated core blocks per trial)
python benchmarks/fair/run_fair_scheduler.py --total-cpus 120 --num-gpus 2

# Warm latency and oversubscription sweeps
bash benchmarks/fair/run_latency_sweep.sh
bash benchmarks/fair/run_oversub_sweep.sh

# Sequential isolation suite (one trial at a time — trust this for latency)
bash benchmarks/isolation/run_iso_full.sh
```

For longer system sweeps (core budgets, concurrency step sizes, GPU marches),
`eqcctpro.EvaluateSystem` remains available; the fair runners above are the
usual entry point for publication-style comparisons.

Pick-quality scoring against a network manifest:

```bash
python benchmarks/fair/compare_orchestrated_picks.py \
    --manifest data/.../manifest.json --picks-dir results/.../output
```

More analysis scripts live in `benchmarks/analysis/`.

## Citation and lineage

If you use RAPID in published work, please also cite the SeisBench models you
run and the TexNet / EQCCTPro / SCMLPick lineage described in Background.
