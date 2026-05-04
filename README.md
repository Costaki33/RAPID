# RAPID — faster-than-`annotate()` benchmarking toolkit

This repository contains a backend-agnostic benchmarking framework for SeisBench pickers. Its job is to measure, break down, and beat SeisBench's `model.annotate()` wall time across a configurable matrix of:

- **Models**: PhaseNet, PhaseNetLight (3001-sample window), EQTransformer,
EQT-NC (6000-sample window). Adding EQCCT is a follow-on once its integratied into SeisBench.
- **Backends**: `baseline_annotate` (unmodified SeisBench), `lean_pytorch`
(FP32 / FP16 / BF16, with optional `torch.compile`), `onnx` (ONNX Runtime),
`tensorrt` (prebuilt `.plan` engines). ONNX and TensorRT are optional —
they're only registered if the packages import.
- **Devices**: CPU and CUDA. Dual-GPU Ray runner supplies the parallel scaling
baseline.
- **Shapes**: station counts 228 / 256 / 512 / 580, batch-size sweep
`{32, 64, 128, 228, 256, 384, 512, 768, 1024}`, overlap-samples sweep.
- **Parallelism**: 1 actor on 1 GPU, **2 actors on 2 GPUs each with its own CPU
preprocess pool** (pipelined), and single-GPU CPU preprocessing worker pool
(1..20) feeding one GPU inference actor.

This repository is part of a larger project focused on enabling real-time seismic phase picking for seismic event detection using deep learning models. 

The preliminary work, [EQCCTPro/RAPID](https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro), enabled sub-11s 3-C waveform processing using persistant model actors to handle 228 stations of 1-minute seismic data for production applications with the Texas Seismological Network (TexNet). This architecture was integrated into [SCMLPick](https://github.com/ut-beg-texnet/scmlpick), a SeisComP module that integrates deep learning models into the SeisComP interface for real-time seismic phase picking, serving as the backbone of the processing approach currently operational in producation at TexNet. 

Further work is focused on improving processing speeds beyond the persistent actor approach by combing different levels of numerical precision with batching. Batching has been applied in SeisBench's `annotate()`, and preliminary trials show that we can achieve faster processing than `annotate()` through these techniques. Prelimary results can be found [here](RAPID_Seisbench_speedup.pdf), with final trials are being finalized for publication in the near future.

## Conda environment

**1. Create and activate an env** (match the Python version to the PyTorch CUDA wheels you will install):

```bash
conda create -n rapid python=3.11 -y
conda activate rapid
```

**2. Install the core stack** the same way you do for EQCCTPro / SCMLPick: PyTorch with the CUDA build that matches your driver, plus ObsPy, NumPy, Ray, SeisBench, Matplotlib, and anything else your workflows need. Practical options:

- **`conda env create -f environment.yml`** from this `RAPID/` directory (pinned pip stack including PyTorch and SeisBench; adjust the env `name:` in that file if it clashes with an existing env), **or**
- follow the parent **`eqcctpro`** repository `README.md` / `environment.yml` if you prefer to manage one env at the repo root.

RAPID does not ship a single core `requirements.txt`; core pins live in those manifests.

**3. Install RAPID optional backend pins** (ONNX, ONNX Runtime GPU, and related helpers; see [Optional backends](#optional-backends)):

```bash
cd RAPID
pip install -r requirements-extra.txt
```

`requirements-extra.txt` assumes the core packages above are already present. Use `onnxruntime` instead of `onnxruntime-gpu` in that file if you only need CPU inference. TensorRT is distributed by NVIDIA for your CUDA toolkit version; follow the comments at the bottom of `requirements-extra.txt` for `tensorrt` / `pycuda`.

## Quick start

```bash
cd RAPID

# Single config sanity check — runs in ~a minute on one GPU.
python scripts/run_benchmark.py \
    --dataset-dir /home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/data/580_stations_1_min_dt/20241215T120000Z_20241215T120100Z \
    --model PhaseNet --child original \
    --backend lean_pytorch --dtype fp16 \
    --device cuda:0 --n-stations 228 --batch-size 256 --repeats 3

# Pipelined path — the "production" fast path (parallel CPU preprocess
#      + megabatched single-GPU forward with CPU↔GPU overlap). This is the
#      configuration that actually beats annotate().
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --batch-size 256 --dtype fp16 \
    --mode single_gpu --n-cpu-workers 16 --repeats 3

# Fair dual-GPU comparison — runs SeisBench's annotate() on 2 GPUs
#       (one loaded model per GPU, stations split 50/50).
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --mode baseline_dual_gpu --repeats 3

# Pipelined dual-GPU — our fast path across 2 GPUs. Each GPU shard
#        runs its own CPU preprocess pool (``--n-cpu-workers`` per GPU) feeding
#        its own inference actor. This is what the matrix runner now uses for
#        ``kind=dual_gpu`` on lean backends.
python scripts/run_pipelined.py \
    --dataset-dir "$DATA_DIR" --model PhaseNet --child original \
    --n-stations 580 --batch-size 512 --dtype bf16 \
    --mode dual_gpu --n-cpu-workers 8 --repeats 3

# Smoke matrix (PhaseNet only, 32 stations, a couple of batch sizes).
python scripts/run_matrix.py --config configs/smoke.json

# Full matrix run (all 4 models × 4 N × 5 backends × 9 batch sizes × 3 repeats).
python scripts/run_matrix.py --config configs/full_matrix.json

# Render every plot from the resulting JSONL.
python scripts/make_plots.py --jsonl results/matrix.jsonl --out-dir figures
```

## Optional backends

The ONNX / TensorRT backends need extra wheels (see `requirements-extra.txt`).
After installing, export pretrained weights once:

```bash
# ONNX for every model
python scripts/export_models.py --onnx-dir models_exported/onnx --skip-trt

# Then build TRT engines (pick your opt batch for the shape you'll process the most with)
python scripts/export_models.py \
    --onnx-dir models_exported/onnx \
    --trt-dir  models_exported/trt \
    --opt-batch 228 --max-batch 1024
```

Add the exported paths to `configs/full_matrix.json` under the ONNX/TensorRT
backend entries:

```json
{ "name": "onnx",     "dtype": "fp32", "onnx_path": "models_exported/onnx/PhaseNet_original.onnx" },
{ "name": "tensorrt", "dtype": "fp16", "engine_path": "models_exported/trt/PhaseNet_original_fp16.plan", "max_batch_size": 1024 }
```

## Pick quality (catalog ground truth and optional A/B drift)

We now evaluate picks against **catalog ground truth** on the SeisBench evaluation traces (the 100-event manual-pick validation set is reflected in those catalog columns). The dtype / timing matrix appends `pick_quality` on every trial, including median absolute onset offset vs catalog P and S (`onset_delta_*_vs_catalog` in samples at the model sampling rate). Run it with:

```bash
cd RAPID
python scripts/run_seisbench_matrix.py --config configs/seisbench_dtype_matrix.json
```

Use `traces_per_dataset` in the JSON config to control how many traces are drawn per dataset (100 is our standard for the publication matrix). For a sweep of probability and pick drift **vs FP32** on miniSEED workloads (separate from SeisBench), there is also `scripts/run_quality_matrix.py`.

For a **quick same-waveform A/B** on any local miniSEED time chunk (no catalog required), FP16 vs FP32 on `lean_pytorch`:

```bash
python scripts/compare_fp16_fp32.py \
    --dataset-dir /path/to/timechunk \
    --model PhaseNet --child original \
    --device cuda:0 --n-stations 228 \
    --out-json results/fp16_vs_fp32_PhaseNet.json
```

That script reports:

- probability trace drift (MAE, max absolute error, RMSE, Pearson correlation)
- pick-time delta at a threshold (median, p95, max — in samples @ model sr)
- speedup FP16 over FP32


## What each timed stage means

| Stage                  | What happens                                                                       |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `merge_streams`        | (baseline only) concatenating all station ObsPy Streams for `model.annotate()`.    |
| `annotate_end_to_end`  | (baseline only) all of SeisBench's internal pipeline, end-to-end.                  |
| `preprocess`           | SeisBench's `annotate_stream_pre` (filter, resample) run once per station.         |
| `window_cut_and_stack` | Build a single `(N_total_windows, 3, in_samples)` numpy array across all stations. |
| `forward`              | Backend's `infer_chunked` — the model forward pass (CUDA-synchronized).            |


Baseline collapses the lean stages into `annotate_end_to_end`; the lean backends
expose them separately so we can see *where* the speedup comes from.

### The method families (evolution of speedups)

Every row in `results/matrix.jsonl` falls into one of these families. They're
recorded as distinct `kind` + `variant` combinations so analysis scripts can
tell them apart and plot the evolution side by side.


| #   | Kind               | Variant suffix                        | What it is                                                                                                                                                         |
| --- | ------------------ | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `baseline`         | (none)                                | SeisBench's `model.annotate()` on one device (CPU or CUDA).                                                                                                        |
| 2   | `dual_gpu`         | `2gpu_baseline`                       | SeisBench's `model.annotate()` run in parallel on 2 GPUs, stations split 50/50.                                                                                    |
| 3   | `single`           | (none)                                | Lean path, 1 GPU, single-threaded preprocess.                                                                                                                      |
| 4   | `cpu_worker_sweep` | `cpuN` (device `cuda:0`)              | Lean path, 1 GPU, N parallel CPU preprocess workers feeding one GPU inference actor.                                                                               |
| 5   | `dual_gpu_serial`  | `2gpu_serial`                         | Lean path, 2 GPUs, single-threaded preprocess per shard (no CPU pool). Kept for the evolution comparison; roughly equivalent to #3 on half the stations per shard. |
| 6   | `dual_gpu`         | `2gpu_cpuN`                           | Lean path, 2 GPUs, each shard runs its own N-worker CPU preprocess pool (pipelined).                                                                               |
| 7   | `cpu_worker_sweep` | `cpu_infer_poolN[_tT]` (device `cpu`) | Lean path, CPU inference, N parallel CPU preprocess workers feeding one CPU inference actor pinned to `T` BLAS threads (or auto-split when `T` is absent).         |


Typical progression seen in the data:
**#3 → #5** (serial, 1 GPU → 2 GPUs): small gain — preprocess still serial per shard.
**#1 → #2** (baseline, 1 GPU → 2 GPUs): small gain — asyncio overhead eats most of the compute win.
**#1 → #4** (pool preprocess on 1 GPU): large gain — preprocess is parallelized.
**#4 → #6** (pool + 2 GPUs): compounding gain — both axes optimized.

### Why CPU speedups plateau well before GPU

Family #7 (`cpu_worker_sweep` on `device="cpu"`) applies the same
parallel-preprocess + megabatch-inference pipeline to the CPU backend, so
you can compare evolution #1 → #7 the same way you do #1 → #4 on GPU. The
ceiling is much lower though, because of a hard physical asymmetry:

- **GPU** preprocessing and inference live on *different silicon*. Adding
CPU preprocessing workers is free — they don't steal from the GPU.
Net: `wall ≈ max(preprocess_parallel, gpu_forward)`. 3-4× speedups are
normal.
- **CPU** preprocessing and inference share the *same cores*. Every
preprocessing worker we add literally takes BLAS threads away from
the inference actor. Net: `wall ≈ preprocess_parallel(k_pre) + inference(k_inf)` with `k_pre + k_inf ≤ N_cores`. Realistic speedups
land in the 1.3-1.8× range unless we switch compute backends (IPEX,
ONNX Runtime CPU) to get more throughput from the same cores.

The runner prevents BLAS over-subscription by pinning each preprocess
worker to 1 thread and giving the inference actor
`max(1, N_cores - n_cpu_workers - 1)` BLAS threads (tunable explicitly via
the `cpu_infer_threads` config axis). Writing the wrong split (e.g. 16
preprocess workers *and* 16 BLAS threads for inference on a 20-core box)
produces the oversubscription penalty that tanks throughput by 2-3× —
that's exactly what the sweep is designed to find.

Config knobs (in `configs/full_matrix.json`):

```json
"cpu_worker_sweep": [1, 2, 4, 8, 12, 16, 20],
"cpu_worker_sweep_on_cpu": true,
"cpu_infer_threads": []
```

- `cpu_worker_sweep_on_cpu`: enable family #7. New rows get
`device="cpu"` so they never collide with existing GPU-path cells on
resume; no existing data needs re-running.
- `cpu_infer_threads`: explicit thread-count axis for the CPU inference
actor. Empty = auto-split. Supply e.g. `[4, 8, 12, 16]` to sweep the
preprocess/inference core split.

### Dual-GPU wall-time semantics

For `kind: "dual_gpu"` and `kind: "dual_gpu_serial"` rows we emit two wall numbers.
Both are persisted; pick whichever matches the question you're asking.


| Column              | Meaning                                                                                                                                                                                                                           |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `wall_time_s`       | Critical-path compute wall (slower of the two shards, excluding per-shard actor setup / model load / warmup). Apples-to-apples with single-GPU `cpu_worker_sweep.wall_time_s`. Use this for speedup comparisons.                  |
| `end_to_end_wall_s` | Real first-call latency: Ray init, actor spawn, model load, warmup, work, teardown. Use this to reason about cold-start cost. In a persistent deployment the actors are reused, so this approaches `wall_time_s` over many calls. |


`_bench_dual_gpu` uses `run_pipelined_dual_gpu` for lean backends and
`run_baseline_dual_gpu` for `baseline_annotate`. `_bench_dual_gpu_serial`
uses the original `run_dual_gpu` (one `run_lean_single` per shard, serial
preprocess inside each actor) and is enabled by `dual_gpu_serial: true`
in the matrix config (default on).

## Folder map

```
RAPID/
├── rapid/
│   ├── backends/     # pluggable inference backends
│   ├── runners/      # single-GPU, dual-GPU, CPU-worker-pool strategies
│   ├── data.py       # station discovery, stream load, windowing, megabatching
│   ├── export.py     # PyTorch -> ONNX -> TRT engine
│   ├── matrix.py     # orchestrator + JSONL writer
│   ├── quality.py    # FP16 vs FP32 drift + pick-time comparison
│   ├── timing.py     # stage-level timers with CUDA sync
│   └── visualize.py  # matplotlib plots
├── scripts/          # CLI entry points
├── configs/          # matrix configuration files
├── models_exported/  # (gitignored) .onnx / .plan artifacts
├── results/          # (gitignored) JSONL outputs
└── figures/          # (gitignored) PNG/SVG plots
```
