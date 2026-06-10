# RAPID — benchmarking real-time deep-learning seismic phase picking

RAPID is a benchmarking toolkit built around one practical question: **what is
the fastest way to run deep-learning phase pickers on real seismic workloads,
without giving up pick quality?** It measures every stage of the picking
pipeline — from framework startup to picks landing on disk — across models,
precisions, batch sizes, CPU/GPU budgets, and deployment strategies, so the
comparisons are fair and the numbers mean something.

## Background

This repository is part of a larger effort to enable real-time seismic phase
picking at the Texas Seismological Network (TexNet). The preliminary work,
[EQCCTPro](https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro), achieved
sub-11-second processing of 1-minute, 3-component waveforms from 228 stations
using persistent Ray model actors. That architecture became the backbone of
[SCMLPick](https://github.com/ut-beg-texnet/scmlpick), the SeisComP module
running in production at TexNet today.

RAPID pushes past the persistent-actor approach by combining reduced numerical
precision (FP16/BF16), `torch.compile`, and aggressive batching. Preliminary
results show these "lean" inference paths beat SeisBench's own `annotate()` —
see [RAPID_Seisbench_speedup.pdf](RAPID_Seisbench_speedup.pdf). The fair
deployment benchmark described below is the publication-grade follow-up.

**Models:** PhaseNet, PhaseNetLight (3001-sample window), EQTransformer, and
EQT-NC (the non-conservative EQTransformer variant). EQCCT is planned once it
lands in SeisBench.

**Inference backends:**

| Backend             | What it is                                                        |
| ------------------- | ----------------------------------------------------------------- |
| `baseline_annotate` | Unmodified SeisBench `annotate()` — the reference point.          |
| `lean_pytorch`      | Our stripped-down path: FP32/FP16/BF16, optional `torch.compile`. |
| `onnx`              | ONNX Runtime (optional; registered only if the package imports).  |
| `tensorrt`          | Prebuilt `.plan` engines (optional; same).                        |

## What's in the repo

```
rapid/                  The benchmarking package
  backends/             Inference backends (baseline, lean PyTorch, ONNX, TensorRT)
  runners/              Single-GPU, dual-GPU, and pipelined execution paths
  benchmark/            Fair-benchmark machinery: stage timing, memory/resource
                        sampling, pick export, pick-quality scoring
eqcctpro/               The Ray orchestration framework (Ripper, Model-Actor,
                        Slipstream) — vendored here; this is its home now
scripts/                Runnable entry points (benchmarks, sweeps, plots, analysis)
configs/                JSON configs for the matrix and dtype sweeps
models_exported/        Where ONNX/TensorRT exports land (kept out of git)
environment.yml         Full pinned conda environment
requirements-extra.txt  Optional ONNX/TensorRT dependencies
```

`results/`, `figures/`, `logs/`, and `data/` (the benchmark networks you build
locally) are created at runtime and intentionally not committed.

> **Note on eqcctpro:** the standalone
> [eqcctpro](https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro)
> repository and the `eqcctpro` PyPI package are **deprecated**. The
> orchestration framework lives in this repository now (`eqcctpro/`) and is
> developed and versioned here. If you have the old PyPI or editable install
> in your environment, remove it first: `pip uninstall eqcctpro`.

## Installation

```bash
# 1. Create the environment (pinned, known-good versions)
conda env create -f environment.yml
conda activate rapid

# 2. Install the package (editable) — installs both `rapid` and the
#    vendored `eqcctpro` orchestration framework
cd RAPID
pip install -e .
```

If you'd rather build your own environment, `pip install -e .` pulls in the
core stack (NumPy, ObsPy, SeisBench, PyTorch, psutil, …). Two optional extras:

```bash
pip install -e ".[orchestration]"        # Ray, for the orchestration benchmarks
pip install -r requirements-extra.txt    # ONNX Runtime / TensorRT helpers
```

Everything is self-contained: the native benchmarks need only the core stack,
and the orchestration benchmarks (Ripper / Model-Actor / Slipstream) use the
vendored `eqcctpro/` package plus Ray. No other repository is required.

## Quick start

```bash
# Sanity check: one model, one backend, ~a minute on one GPU
python scripts/run_benchmark.py \
    --dataset-dir /path/to/data/20241215T120000Z_20241215T120100Z \
    --model PhaseNet --child original \
    --backend lean_pytorch --dtype fp16 \
    --device cuda:0 --n-stations 228 --batch-size 256 --repeats 3

# The fast path: parallel CPU preprocessing feeding megabatched GPU inference
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --batch-size 256 --dtype fp16 \
    --mode single_gpu --n-cpu-workers 16 --repeats 3

# Full backend matrix + plots
python scripts/run_matrix.py --config configs/full_matrix.json
python scripts/make_plots.py --jsonl results/matrix.jsonl --out-dir figures
```

## The fair deployment benchmark

This is the heart of the repo: one benchmark that puts **every deployment
strategy on identical footing** — same waveforms, byte-identical model-input
windows, the same seven timed stages, the same memory metric, and pick quality
scored against the same catalog ground truth.

### What gets compared

**Native family** — the model running directly in one process:

- `annotate` — SeisBench `annotate()` writing probability streams + picks
- `classify` — SeisBench `classify()` (its internal picker extracts the picks)
- `slipstream` — our lean path: FP32/FP16/BF16 ± `torch.compile`, batched windows

**Orchestration family** — the same pickers wrapped in eqcctpro's Ray
deployment strategies, the way they'd actually run in production:

- `ripper` / `ripper_slipstream` — task-per-station queue; each task loads the model
- `modelactor` / `modelactor_slipstream` — persistent actor pool, models loaded once

Each trial sweeps datasets (STEAD/TXED test networks at 250 and 580 stations),
CPU budgets (5–20 cores, plus CPU+GPU), window regimes, precisions, and batch
sizes. Native trials run 5 repeats, each in a fresh subprocess; orchestration
trials run once per configuration. The full matrix is ~23,000 trials.

### What gets measured

- **Seven stages that sum to the total**: `framework_init`, `model_load`,
  `waveform_access`, `preprocess`, `warmup`, `inference`, `pick_generation`.
  Orchestration stages are *measured* (per-task busy-time sums plus driver
  wall segments), never estimated.
- **Memory**: process-tree RSS *and* PSS (PSS counts Ray's shared pages once,
  so single-process and actor-pool numbers are actually comparable), plus VRAM
  for GPU trials.
- **Resources**: CPU utilization, disk I/O, GPU utilization/power/energy
  (NVML), and host package energy (RAPL).
- **Pick quality**: precision/recall/F1 and onset-time residuals vs the
  catalog, scored for every repeat.

Every trial writes a self-contained `result.json`. Orchestration runs also
record per-trial `trial_results.json` files with snake_case fields
(`orchestration_strategy`, `n_modelactors`, `concurrent_tasks`, `batch_size`,
per-stage busy sums) — the eqcctpro CSVs are legacy plumbing, not the record.

### Running it

```bash
# Build the test networks once (downloads STEAD/TXED via SeisBench)
python scripts/build_seisbench_network.py --dataset stead --n-stations 580

# Launch the whole matrix (resume-safe: re-run the same command after any stop)
nohup python scripts/run_fair_scheduler.py --total-cpus 120 --num-gpus 2 \
    >> results/fair_benchmark/scheduler.log 2>&1 &

# Watch progress
python scripts/fair_progress.py
tail -f results/fair_benchmark/scheduler.log
```

The scheduler is a core-block FCFS dispatcher: every trial gets a dedicated,
`taskset`-pinned block of cores (and optionally a GPU), so concurrent trials
never share hardware. Kill it any time; nothing is lost.

Two follow-on sweeps reuse the same machinery:

- `scripts/run_latency_sweep.sh` — warm-actor, back-to-back feed latency:
  cold-start vs warm-feed times for a persistent actor pool.
- `scripts/run_oversub_sweep.sh` — oversubscription: requesting 1–4× more
  concurrent actors/tasks than cores, mapping where RAM/VRAM becomes the real
  constraint (eqcctpro never binds tasks to cores; memory is the true limit).

For unattended multi-day runs, `scripts/benchmark_babysit.sh` (installed as a
plain cron job) resumes the scheduler if it dies and chains the sweeps when
the main matrix finishes.

### Running a single trial

Every trial the scheduler launches is also a standalone script, which is handy
for debugging one configuration:

```bash
python scripts/run_fair_trial.py --method slipstream --dataset stead \
    --n-stations 250 --model PhaseNet --device cpu --n-cpus 5 \
    --dtype fp16 --batch-size 256 --tag my_test

python scripts/run_fair_orch_trial.py --strategy modelactor_slipstream \
    --dataset stead --n-stations 250 --model PhaseNet --device cpu \
    --n-cpus 5 --dtype fp32 --tag my_test
```

## Pick quality

Pick quality is never an afterthought here — every benchmark trial scores its
own picks. For standalone analysis:

```bash
# Score orchestrated picks against a network's catalog manifest
python scripts/compare_orchestrated_picks.py \
    --manifest .../manifest.json --picks-dir .../output

# Dtype matrix with per-trial pick quality vs catalog
python scripts/run_seisbench_matrix.py --config configs/seisbench_dtype_matrix.json

# Quick FP16-vs-FP32 drift check on any local miniSEED chunk (no catalog needed)
python scripts/compare_fp16_fp32.py --dataset-dir /path/to/timechunk \
    --model PhaseNet --child original --device cuda:0 --n-stations 228
```

`scripts/README_pick_quality.md` documents the aggregation and figure scripts.

## Optional backends: ONNX and TensorRT

```bash
# Export pretrained weights to ONNX (and optionally TensorRT engines)
python scripts/export_models.py --onnx-dir models_exported/onnx --skip-trt
python scripts/export_models.py --onnx-dir models_exported/onnx \
    --trt-dir models_exported/trt --opt-batch 228 --max-batch 1024
```

Then point `configs/full_matrix.json` at the exported paths:

```json
{ "name": "onnx",     "dtype": "fp32", "onnx_path": "models_exported/onnx/PhaseNet_original.onnx" },
{ "name": "tensorrt", "dtype": "fp16", "engine_path": "models_exported/trt/PhaseNet_original_fp16.plan", "max_batch_size": 1024 }
```

TensorRT itself comes from NVIDIA for your CUDA version; see the notes at the
bottom of `requirements-extra.txt`.

## Stage glossary (lean benchmark)

| Stage                  | What happens                                                                 |
| ---------------------- | ----------------------------------------------------------------------------- |
| `merge_streams`        | (baseline only) concatenating station ObsPy Streams for `model.annotate()`.  |
| `annotate_end_to_end`  | (baseline only) all of SeisBench's internal pipeline, end to end.             |
| `preprocess`           | SeisBench's `annotate_stream_pre` (filter, resample), once per station.      |
| `window_cut_and_stack` | Building one `(N_windows, 3, in_samples)` array across all stations.         |
| `forward`              | The backend's `infer_chunked` — the model forward pass (CUDA-synchronized).  |

The baseline collapses the lean stages into `annotate_end_to_end`; the lean
backends expose them separately so you can see *where* the speedup comes from.

## Method families in `matrix.jsonl`

Rows from `scripts/run_matrix.py` / `run_pipelined.py` carry a `kind` +
`variant` pair so analysis scripts can plot the evolution of speedups side by
side:

| #   | Kind               | Variant suffix                        | What it is                                                                  |
| --- | ------------------ | ------------------------------------- | ---------------------------------------------------------------------------- |
| 1   | `baseline`         | (none)                                | SeisBench `annotate()` on one device.                                       |
| 2   | `dual_gpu`         | `2gpu_baseline`                       | `annotate()` in parallel on 2 GPUs, stations split 50/50.                   |
| 3   | `single`           | (none)                                | Lean path, 1 GPU, single-threaded preprocess.                               |
| 4   | `cpu_worker_sweep` | `cpuN` (device `cuda:0`)              | Lean path, 1 GPU, N CPU preprocess workers feeding one GPU inference actor. |
| 5   | `dual_gpu_serial`  | `2gpu_serial`                         | Lean path, 2 GPUs, single-threaded preprocess per shard.                    |
| 6   | `dual_gpu`         | `2gpu_cpuN`                           | Lean path, 2 GPUs, each shard with its own N-worker CPU pool (pipelined).   |
| 7   | `cpu_worker_sweep` | `cpu_infer_poolN[_tT]` (device `cpu`) | Lean path, CPU inference, N preprocess workers + T BLAS threads.            |

## Older sweeps

Earlier orchestration sweeps (`run_seisbench_sweep.py`, `run_parallel_sweep.py`,
`run_native_seisbench_sweep.py`, `run_modelactor_slipstream_sweep.py`) predate
the fair benchmark and remain usable, but the fair benchmark supersedes them
for any cross-strategy comparison — it is the only path where warmup, stage
accounting, memory metrics, and pick scoring are guaranteed identical across
strategies.
