# RAPID

RAPID (Resource-Aware Parallel Inference Dispatcher) is a toolkit for
**real-time** deep-learning seismic phase picking on whole station networks. It
packages lean inference (Slipstream), Ray-based orchestration (Model-Actor and
Ripper), and a fair deployment benchmark so you can compare strategies on the
same waveforms and the same pick-quality scores — and keep up with continuous
60-second network windows.

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
deployment benchmark across models, devices, and scheduling strategies. EQCCTPro
as a standalone package is deprecated; install and import RAPID going forward.
The orchestration runtime that began in EQCCTPro now lives under
`rapid/orchestration/`.

The central finding is that **how you parallelize the work matters more than the
device alone**. A pool of persistent, single-threaded workers (Model-Actor) keeps
a many-core CPU busy on heavy models where a single-process Classify or Annotate
call leaves cores idle. Slipstream is a forward path that can run inside that
pool; it is not a competing orchestrator.

On a 20-core CPU allocation, warm Model-Actor finished a 60-second, 580-station
window in about 1.3–2.3 s across four SeisBench pickers (p99 under 2.5 s).
Single-process Annotate averaged 1.5–11.7 s on the same CPU but with heavy-model
p99 latencies near 30 s. Model-Actor on CPU also beat single-GPU Annotate on the
heavier models (roughly 15% faster on average, up to about 34% for EQT-NC).

Supported models: PhaseNet, PhaseNetLight, EQTransformer, and EQT-NC
(non-conservative EQTransformer). EQCCT is planned once it lands in SeisBench.

### Measured speedups (isolated re-measurement)

Hardware: AMD Threadripper PRO 7985WX, 512 GB RAM, two NVIDIA RTX 6000 Ada
(49 GB). Workload: STEAD synthetic network, **580 stations**, native model
windows. Values are **mean seconds** per network window. Bold marks the fastest
method for that model within each table.

**CPU — 20 cores.** Classify is single-process SeisBench `classify()` at its
best thread setting (1 intra-op thread), cold end-to-end over **3 repeats**.
Annotate, Model-Actor[classify], and Model-Actor[Slipstream-BF16] are **warm**
streaming means: **10 repeats × 8 feeds**, first feed discarded (**70**
steady-state windows per cell).

| Model | Classify | Annotate | Model-Actor[classify] | Model-Actor[Slipstream-BF16] |
| --- | ---: | ---: | ---: | ---: |
| PhaseNet | 5.52 | 1.52 | 1.35 | **1.29** |
| PhaseNetLight | 4.71 | 3.40 | 1.34 | **1.29** |
| EQTransformer | 22.51 | 8.88 | 2.28 | **2.16** |
| EQT-NC | 13.65 | 11.72 | **1.87** | 1.88 |

**GPU — one RTX 6000 Ada (host pinned to 20 cores), same warm protocol**
(10 × 7 windows), plus a two-GPU Model-Actor[classify] column (5 repeats × 7
windows). Annotate on one GPU is fastest for every model; the actor pool helps
most when you stay on CPU.

| Model | Annotate (1 GPU) | Model-Actor[classify] (1 GPU) | Model-Actor[Slipstream-BF16] (1 GPU) | Model-Actor[classify] (2 GPUs) |
| --- | ---: | ---: | ---: | ---: |
| PhaseNet | **1.37** | 1.81 | 1.52 | 1.81 |
| PhaseNetLight | **1.38** | 1.58 | 1.45 | 1.54 |
| EQTransformer | **2.93** | 9.10 | 9.30 | 5.18 |
| EQT-NC | **2.85** | 4.74 | 5.11 | 3.03 |

**GPU Model-Actor[classify] by host CPU count** (warm mean, 580 stations; 5
repeats × 7 windows at 5/10/15 cores, 10 × 7 at 20 cores). Annotate on one GPU
(20 host cores) is shown for reference.

| Model | 5 cores | 10 cores | 15 cores | 20 cores | Annotate (1 GPU, 20 cores) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhaseNet | 2.94 | 2.47 | 1.88 | 1.81 | **1.37** |
| PhaseNetLight | 2.15 | 1.97 | 1.68 | 1.58 | **1.38** |
| EQTransformer | 9.82 | 9.13 | 9.10 | 9.10 | **2.93** |
| EQT-NC | 5.85 | 4.91 | 4.66 | 4.74 | **2.85** |

## What's in the repo

```
rapid/
  api.py                 Single-process annotate / classify / slipstream
  backends/              Lean PyTorch + SeisBench baseline backends
  benchmark/             Fair-benchmark timing, memory, pick scoring
  orchestration/         Ray Model-Actor / Ripper / Slipstream (ex-EQCCTPro)
    api.py               pick(), model_actor(), ripper()
    runtime/             RunEQCCTPro, EvaluateSystem
    actors/              parallelization.py, slipstream_actor.py
    models/              SeisBench + EQCCT TF model wrappers
    support/             tools, timing, picks, waveform helpers
examples/                Build synthetic networks and run a first pick
benchmarks/
  fair/                  Fair matrix, latency / oversubscription sweeps
  isolation/             Sequential, contention-free re-measurements
  analysis/              Summaries and pick-quality aggregation
configs/                 JSON configs for matrix / dtype sweeps
```

Runtime outputs (`results/`, `figures/`, `logs/`, `data/`) stay local and are
gitignored.

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
`rapid.EvaluateSystem` remains available; the fair runners above are the usual
entry point for publication-style comparisons.

Pick-quality scoring against a network manifest:

```bash
python benchmarks/fair/compare_orchestrated_picks.py \
    --manifest data/.../manifest.json --picks-dir results/.../output
```

More analysis scripts live in `benchmarks/analysis/`.

## Citation and lineage

If you use RAPID in published work, please also cite the SeisBench models you
run and the TexNet / EQCCTPro / SCMLPick lineage described in Background.
