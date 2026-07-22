# RAPID
### Resource-Aware Parallel Inference Dispatcher for Real-Time, Network-Scale Deep-Learning Seismic Phase Picking

RAPID (Resource-Aware Parallel Inference Dispatcher) is a toolkit for evaluating and deploying deep-learning seismic phase pickers under real-time, network-scale workloads. It extends SeisBench with orchestration strategies designed for persistent parallel execution, reduced-precision inference through Slipstream, and a benchmarking framework for comparing deployment strategies across different hardware configurations. Rather than focusing solely on model inference speed, RAPID is intended to evaluate how different deployment strategies affect end-to-end processing performance for continuous seismic monitoring.

RAPID does not introduce new deep-learning phase pickers. Instead, it provides deployment strategies and benchmarking tools for existing SeisBench models, allowing users to evaluate how different orchestration approaches affect real-time processing performance.

## Background

Deep-learning phase pickers have substantially improved automated seismic phase identification, but deploying these models in real-time monitoring systems remains computationally challenging. While modern deep-learning models can process individual waveform windows quickly, monitoring centers must continuously process hundreds of stations while satisfying strict latency requirements. Consequently, deployment performance depends not only on the underlying model, but also on how waveform streams are distributed across the available computational resources.

RAPID was developed as part of a broader effort to enable real-time deep-learning phase picking at the Texas Seismological Network (TexNet). Earlier work through
[EQCCTPro](https://github.com/ut-beg-texnet/eqcct/tree/main/eqcctpro) introduced ephemeral and persistent Ray-based model actors capable of processing one-minute waveform windows from hundreds of stations in real time. That orchestration framework later became the foundation of
[SCMLPick](https://github.com/ut-beg-texnet/scmlpick), the real-time deep-learning processing SeisComP module currently deployed within TexNet.

RAPID builds on that work by extending the same orchestration concepts with Slipstream, a reduced-precision inference pipeline, together with a standardized benchmarking framework for evaluating deployment strategies across different models, hardware platforms, and scheduling approaches. As a standalone package, EQCCTPro is now deprecated, and its orchestration runtime has been incorporated into RAPID under `rapid/orchestration/`.

## Design philosophy

RAPID was developed to evaluate deployment strategies rather than introduce new deep-learning phase pickers. Existing SeisBench models remain unchanged. Instead, RAPID focuses on how waveform streams are orchestrated across the available computational resources through persistent worker pools, reduced-precision inference, and reproducible benchmarking. The goal is to determine how different deployment strategies affect end-to-end processing performance while preserving the underlying model behavior.

Accordingly, RAPID provides three complementary capabilities:

- Native single-process SeisBench inference through Annotate, Classify, and RAPID's Slipstream.
- Persistent and ephemeral orchestration strategies, including Model-Actor and Ripper, for evaluating station-level parallelism.
- A benchmarking framework for comparing inference pipelines, orchestration strategies, hardware configurations, and pick quality under identical workloads.

These components allow deployment strategies to be evaluated under the same waveform inputs, timing measurements, and quality metrics, enabling direct comparisons between different execution approaches.

## Repository structure

RAPID is organized into three primary components. The core package contains the inference and orchestration implementations, the benchmarking scripts reproduce the experiments described in the accompanying manuscript, and the example utilities illustrate how synthetic networks can be constructed for controlled evaluation.

```
rapid/
  api.py                 Single-process annotate / classify / slipstream
  backends/              Lean PyTorch + SeisBench baseline backends
  benchmark/             Fair-benchmark timing, memory, pick scoring
  orchestration/         Ray Model-Actor / Ripper / Slipstream
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

The orchestration package contains the persistent execution strategies evaluated throughout the manuscript, while the benchmarking scripts reproduce the experiments discussed there. Runtime outputs (`results/`, `figures/`, `logs/`, `data/`) are created locally and are intentionally excluded from version control.

## Installation

RAPID currently requires Python 3.10 or newer and builds directly on SeisBench. Installation is intentionally lightweight so that existing SeisBench workflows can be migrated with minimal modification. The import package remains ``rapid``; the PyPI distribution name is ``rapid-seis`` because the name ``rapid`` is already taken.

From PyPI:

```bash
pip install rapid-seis
```

This installs single-process Annotate, Classify, Slipstream, and the benchmarking utilities. If you also want the Ray-based orchestration strategies (Model-Actor or Ripper), install the optional extra:

```bash
pip install "rapid-seis[orchestration]"
```

From a clone:

```bash
conda env create -f environment.yml
conda activate rapid
cd RAPID
pip install -e .
# optional, only if you need Model-Actor / Ripper:
pip install -e ".[orchestration]"
```
## Getting started

RAPID can be used as a drop-in replacement for SeisBench inference or as a framework for evaluating deployment strategies. The examples below demonstrate the most common workflows, beginning with the construction of a synthetic station network and continuing through single-process and orchestrated picking.

### Building a synthetic network

Because the deprecated hand-curated waveform dumps are no longer distributed with the repository, synthetic networks are constructed from STEAD or TXED catalog traces. This allows the number of stations, the length of each trace, and the placement of catalog P and S arrivals to be controlled explicitly so that phase arrivals fall within the model input window.

```bash
# 50 stations, 3001-sample traces, both P and S inside the window
python examples/build_seisbench_network.py \
    --dataset stead --n-stations 50 --require-s \
    --min-pick-sample 0 --max-pick-sample 2951 \
    --trim-samples 3001 \
    --out-root data/seisbench_networks
```

| Flag | Purpose |
| --- | --- |
| `--n-stations` | Network size (catalog traces are tiled when more stations are requested than unique events are available) |
| `--trim-samples` | Trace length written to disk (for example, 3001 or 6000) |
| `--min-pick-sample` / `--max-pick-sample` | Keep catalog P and S samples inside this range |
| `--dataset stead\|txed` | Source catalog |
| `--seed` | Reproducible station sampling |

The builder writes per-station miniSEED files together with a `manifest.json` containing start and end times and catalog pick sample indices for later quality scoring.

### Model-Actor

Model-Actor maintains persistent model instances in memory and is intended for continuously running monitoring systems where repeated model initialization would otherwise dominate runtime. Station streams are distributed across the worker pool, allowing preprocessing, inference, and pick generation to proceed concurrently. This workflow requires the optional orchestration extra (`pip install "rapid-seis[orchestration]"`).

```python
from rapid import pick

pick(
    "data/seisbench_networks/<your_network>",
    "results/picks_ma_classify",
    model="EQTransformer",
    strategy="modelactor",
    forward="classify",
    n_workers=20,
)
```

The same persistent pool can execute Slipstream rather than SeisBench Classify by selecting the corresponding forward path and numerical precision:

```python
pick(
    "data/seisbench_networks/<your_network>",
    "results/picks_ma_slipstream",
    model="EQTransformer",
    strategy="modelactor",
    forward="slipstream",
    dtype="bf16",
    n_workers=20,
)
```

GPU devices may be specified through physical device indices:

```python
pick(..., gpus=[0])       # one GPU
pick(..., gpus=[0, 1])    # two GPUs
```

### Ripper

Ripper launches short-lived workers for each station task and therefore reloads the model repeatedly during execution. It is included primarily as a baseline for evaluating the cost of ephemeral orchestration relative to persistent Model-Actor execution. Like Model-Actor, it requires the optional orchestration extra.

```python
pick(
    "data/seisbench_networks/<your_network>",
    "results/picks_ripper",
    model="EQTransformer",
    strategy="ripper",
    forward="classify",
    n_workers=20,
)
```

### Slipstream

Slipstream provides a reduced-precision inference pathway compatible with both standalone single-process execution and persistent Model-Actor workers. Supported precisions include FP32, FP16, and BF16. For EQTransformer models, BF16 is preferred because FP16 can overflow the model's attention padding sentinel.

```python
from rapid import slipstream

slipstream(
    "data/seisbench_networks/<your_network>",
    "results/slipstream",
    model="PhaseNet",
    dtype="bf16",
    device="cuda:0",
)
```

### Single-process Annotate and Classify

Annotate and Classify remain available as native SeisBench baselines. These interfaces execute within a single process and therefore provide the reference against which orchestration strategies are compared.

```python
from rapid import annotate, classify

annotate("data/.../network", "results/ann", model="PhaseNet", device="cpu", batch_size=256)
classify("data/.../network", "results/cls", model="PhaseNet", device="cpu", torch_threads=1)
```

The same workflows are also available from the command line:

```bash
python examples/pick_network.py \
    --input-dir data/seisbench_networks/<your_network> \
    --strategy modelactor --forward slipstream --dtype bf16 \
    --n-workers 16
```

## Representative benchmark results

The following results summarize representative performance observed during the experiments described in the accompanying manuscript. They are intended to illustrate the relative behavior of the different deployment strategies rather than reproduce every benchmark configuration. Complete methodology, additional configurations, and pick-quality analyses are provided in the manuscript.

All timing-sensitive measurements were obtained on an AMD Threadripper PRO 7985WX workstation with 512 GB of memory and two NVIDIA RTX 6000 Ada GPUs. The workload is a synthetic 580-station STEAD network. Values are mean seconds per network window. Bold entries indicate the fastest method for that model within each table.

### Single-process inference

Single-process Classify and Annotate remain confined to one Python process. Classify processes stations sequentially, whereas Annotate batches windows across the merged network stream. Because neither interface distributes station workloads across multiple processes, additional CPU cores provide limited benefit once the optimal PyTorch thread setting has been selected.

| Model | Classify | Annotate |
| --- | ---: | ---: |
| PhaseNet | 5.52 | **1.52** |
| PhaseNetLight | 4.71 | **3.40** |
| EQTransformer | 22.51 | **8.88** |
| EQT-NC | 13.65 | **11.72** |

Classify corresponds to single-process SeisBench `classify()` at its best thread setting (one intra-op thread), measured as a cold end-to-end total over three repeats. Annotate corresponds to warm streaming means under the protocol described below. The comparison illustrates that Annotate improves upon sequential Classify, but both remain limited by single-process execution on the heavier models.

### Orchestration on CPU

Model-Actor maintains persistent workers so that the model-loading cost is paid once and then amortized across successive network windows. Slipstream may be executed inside the same worker pool as an alternative forward path. The warm results below use ten repeats of eight consecutive feeds, with the first feed of each repeat discarded, yielding seventy steady-state windows per cell on a twenty-core CPU allocation.

| Model | Annotate | Model-Actor[classify] | Model-Actor[Slipstream-BF16] |
| --- | ---: | ---: | ---: |
| PhaseNet | 1.52 | 1.35 | **1.29** |
| PhaseNetLight | 3.40 | 1.34 | **1.29** |
| EQTransformer | 8.88 | 2.28 | **2.16** |
| EQT-NC | 11.72 | **1.87** | 1.88 |

These results illustrate that persistent model instances substantially reduce processing time once initialization has been amortized. The largest improvements relative to single-process Annotate are observed for the heavier EQTransformer models, where station-level parallelism reduces the inference term that otherwise dominates the window. Within the pool, Classify and Slipstream-BF16 remain close, indicating that the primary benefit arises from orchestration rather than from the choice of forward path alone.

### GPU execution

On GPU, single-process Annotate batches the network efficiently on a single device. Spreading the same workload across Model-Actor workers can introduce contention for that device, although allocating the pool across two GPUs narrows the gap for the heavier models. The warm protocol matches the CPU orchestration experiments for one-GPU configurations; the two-GPU Model-Actor column uses five repeats of seven steady-state windows.

| Model | Annotate (1 GPU) | Model-Actor[classify] (1 GPU) | Model-Actor[Slipstream-BF16] (1 GPU) | Model-Actor[classify] (2 GPUs) |
| --- | ---: | ---: | ---: | ---: |
| PhaseNet | **1.37** | 1.81 | 1.52 | 1.81 |
| PhaseNetLight | **1.38** | 1.58 | 1.45 | 1.54 |
| EQTransformer | **2.93** | 9.10 | 9.30 | 5.18 |
| EQT-NC | **2.85** | 4.74 | 5.11 | 3.03 |

Increasing the number of host cores available to a one-GPU Model-Actor pool improves runtime up to a point, after which additional cores provide diminishing returns because inference remains limited by the shared device.

| Model | 5 cores | 10 cores | 15 cores | 20 cores | Annotate (1 GPU, 20 cores) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PhaseNet | 2.94 | 2.47 | 1.88 | 1.81 | **1.37** |
| PhaseNetLight | 2.15 | 1.97 | 1.68 | 1.58 | **1.38** |
| EQTransformer | 9.82 | 9.13 | 9.10 | 9.10 | **2.93** |
| EQT-NC | 5.85 | 4.91 | 4.66 | 4.74 | **2.85** |

### Memory

Improving warm-streaming latency through persistent Model-Actor workers comes at the cost of additional host memory. To fairly compare memory usage between the single-process and orchestration strategies, we measured the peak process-tree proportional set size (PSS), which counts memory pages shared among Ray workers only once. The values below represent the mean peak PSS in gigabytes for the 580-station STEAD workload using a 20-core CPU allocation. Cold-start measurements capture the maximum memory footprint while the model pool is being initialized, whereas warm-streaming measurements represent continuous operation after the one-time initialization cost has already been paid.

Warm-streaming memory for Annotate and both Model-Actor configurations was measured during the isolated warm-streaming benchmarks. Single-process Classify and Slipstream were not evaluated under that same streaming protocol because they retain only a single model instance throughout execution. Therefore, their warm-streaming values are reported using the same peak memory measured during the isolated cold-start experiments, which represents their operational memory footprint once the process is resident. Ripper is not included because it is not intended to operate as a persistent warm-streaming deployment.

| Method | Workers | Cold-start peak PSS (GB) | Warm streaming peak PSS (GB) |
| --- | ---: | ---: | ---: |
| Classify | 1 | 0.7 | 0.7 |
| Annotate | 1 | 1.1 | 1.4 |
| Slipstream | 1 | 1.0 | 1.0 |
| Model-Actor[classify] | 20 | 22.1 | 11.4 |
| Model-Actor[Slipstream-BF16] | 20 | 22.2 | 11.3 |

As expected, the single-process methods remained near 1 GB because only a single model instance was loaded into memory. In contrast, Model-Actor maintains one model replica for each persistent worker, resulting in a substantially larger memory footprint while the worker pool is being initialized. After initialization, however, the warm-streaming footprint stabilized near 11 GB for the 20-worker pool. Using BF16 Slipstream did not noticeably reduce CPU host-memory usage compared to Classify because the replicated model parameters dominate the overall footprint. Its primary memory advantage is instead on GPU VRAM. Because the memory footprint increases with the number of persistent workers, these results further support sizing the actor pool to approximately one worker per available CPU core.

Taken together with the runtime results, these memory measurements illustrate the tradeoff between latency and memory usage. On CPU-only systems, Model-Actor provides the lowest warm-streaming latency but requires sufficient host memory to maintain the persistent model pool. When a GPU is available, single-process Annotate remained the fastest configuration evaluated while also requiring substantially less host memory than Model-Actor.

## Reproducing the manuscript benchmarks

Each benchmarking script reproduces one or more experiments described in the manuscript, including single-process inference, orchestration performance, warm streaming latency, oversubscription behavior, and pick-quality evaluation. Trials are designed so that deployment strategies are compared under identical waveforms, timed stages, memory metrics, and catalog pick scores.

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

# Full matrix (resume-safe; dedicated core blocks per trial)
python benchmarks/fair/run_fair_scheduler.py --total-cpus 120 --num-gpus 2

# Warm latency and oversubscription sweeps
bash benchmarks/fair/run_latency_sweep.sh
bash benchmarks/fair/run_oversub_sweep.sh

# Sequential isolation suite used for the timing results reported above
bash benchmarks/isolation/run_iso_full.sh
```

For longer system characterization sweeps, including core budgets and concurrency marches, `rapid.EvaluateSystem` remains available. Pick quality relative to a network manifest can be scored with:

```bash
python benchmarks/fair/compare_orchestrated_picks.py \
    --manifest data/.../manifest.json --picks-dir results/.../output
```

Additional analysis utilities are provided under `benchmarks/analysis/`.

## Citation

If RAPID contributes to your research, please cite the accompanying publication below. The manuscript provides the complete implementation details, benchmarking methodology, and experimental evaluation of the deployment strategies included in this repository. Please also cite the SeisBench models you use and the TexNet / EQCCTPro / SCMLPick lineage described in Background.

```bibtex
@article{rapid2026,
  title   = {RAPID: Resource-Aware Parallel Inference for Real-Time Seismic Phase Picking},
  author  = {Skevofilax, Constantinos and others},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## Current status

RAPID is under active development. The orchestration strategies and benchmarking framework described in the accompanying manuscript are fully implemented, while additional deployment strategies, supported models, and hardware configurations will continue to be incorporated as the project evolves.
