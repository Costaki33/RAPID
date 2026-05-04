"""Test matrix configuration and execution orchestration.

A single YAML/JSON-compatible dict describes the full sweep:

    models         : list of (parent, child, label)
    n_stations_list: e.g. [228, 256, 512, 580]
    backends       : list of backend dicts
                       {name: "baseline_annotate", dtype: "fp32"}
                       {name: "lean_pytorch", dtype: "fp32"}
                       {name: "lean_pytorch", dtype: "fp16"}
                       {name: "lean_pytorch", dtype: "bf16"}
                       {name: "onnx",  dtype: "fp32", onnx_path: "..."}
                       {name: "tensorrt", dtype: "fp16", engine_path: "..."}
    devices        : ["cpu", "cuda:0"]
    batch_sizes    : [32, 64, 128, 256, 384, 512, 768, 1024]
    overlap_sweep  : [0, 1500]   # 0 for our fast path; classic for parity
    repeats        : 5
    dual_gpu       : true        # run the 2-GPU shard test per backend
    cpu_worker_sweep: [1,2,4,8,12,16,20]  # only for GPU runs

All combinations are evaluated and written to a single JSONL file so a
partially-completed run can be resumed. Failures don't stop the sweep.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


LOG = logging.getLogger("rapid.matrix")


# Bumped whenever the row schema changes in a way that affects resume keys.
# Persisted into each row as ``schema_version`` and into the env row as
# ``schema_version`` too, so stale JSONLs can be detected easily later.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class MatrixConfig:
    dataset_dir: str
    models: List[Dict[str, str]]
    n_stations_list: List[int]
    backends: List[Dict[str, Any]]
    devices: List[str]
    batch_sizes: List[int]
    overlap_sweep: List[int] = field(default_factory=lambda: [0])
    repeats: int = 3
    warmup_iters: int = 1
    dual_gpu: bool = True
    # Also benchmark the old serial-preprocess lean 2-GPU path (one
    # ``run_lean_single`` per shard, no CPU pool). Kept around because it's
    # the intermediate step between single-GPU lean and the fully pipelined
    # 2-GPU path — useful for the "evolution of methods" comparison.
    dual_gpu_serial: bool = True
    cpu_worker_sweep: List[int] = field(default_factory=list)
    # When True, also run ``cpu_worker_sweep`` with the CPU inference path
    # (``device="cpu"``). Uses the same ``cpu_worker_sweep`` values for
    # ``n_cpu_workers`` and auto-splits threads between preprocess workers
    # and the inference actor (see ``run_cpu_worker_sweep``). New cells get
    # ``device="cpu"`` so they don't collide with existing GPU rows on resume.
    cpu_worker_sweep_on_cpu: bool = False
    # Optional explicit thread-count axis for the CPU inference actor. When
    # non-empty, each ``n_cpu_workers`` value is crossed with every
    # ``cpu_infer_threads`` value — so you can test e.g. 4 preprocess workers
    # with either 8 or 16 BLAS threads for inference. Left empty means "let
    # the runner auto-pick a sensible split" (best-effort default).
    cpu_infer_threads: List[int] = field(default_factory=list)
    # CPU-only overrides. When set, these replace the global ``backends`` /
    # ``batch_sizes`` lists for trials where ``device == "cpu"``. GPU trials
    # still use the global lists. Useful for trimming the CPU sweep (drop
    # FP16 which most CPUs don't support in hardware, narrow batch-size axis
    # since it barely matters on CPU) without touching GPU coverage. Leave
    # empty to use the global lists on both devices (default, unchanged
    # behavior).
    cpu_backends_override: List[Dict[str, Any]] = field(default_factory=list)
    cpu_batch_sizes_override: List[int] = field(default_factory=list)
    # Per-shard CPU preprocess pool sizes for the pipelined dual-GPU runner.
    # Only consumed when ``dual_gpu`` is True and the backend is not
    # ``baseline_annotate`` (the baseline path uses SeisBench's internal
    # single-threaded preprocess and has no such knob).
    dual_gpu_cpu_workers: List[int] = field(default_factory=lambda: [6, 8, 12])
    output_jsonl: str = "results/matrix.jsonl"
    seed: int = 0
    resume: bool = True
    retry_errors: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MatrixConfig":
        # Drop JSON "comment" keys (anything starting with ``_comment``).
        # We use them liberally to document surprising config choices
        # (FP16 dropped on CPU, for instance), and the dataclass ctor
        # would otherwise reject them as unexpected kwargs.
        clean = {k: v for k, v in d.items() if not str(k).startswith("_comment")}
        return cls(**clean)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


CellKey = Tuple[Any, ...]


def _backend_extra_sig(backend_cfg_or_extra: Dict[str, Any]) -> str:
    """Canonical JSON signature of a backend config's *extra* knobs.

    Only non-identity fields are kept: ``name``, ``dtype``, ``device`` are
    already explicit axes of the cell key, so they're stripped. The remaining
    dict (``compile``, ``onnx_path``, ``engine_path``, quantization opts, …)
    is JSON-dumped with sorted keys so dict order never affects the hash.

    Accepts either a full backend config dict (with ``name``/``dtype``) or a
    pre-stripped ``backend_extra`` dict — both normalize to the same string.
    """
    d = {
        k: v for k, v in (backend_cfg_or_extra or {}).items()
        if k not in ("name", "dtype", "device")
    }
    return json.dumps(d, sort_keys=True, default=str)


def _row_key(row: Dict[str, Any]) -> CellKey:
    """Uniquely identify a completed benchmark cell + repeat.

    Includes every axis the runner sweeps over, including the canonicalized
    ``backend_extra`` signature so variants like ``lean_pytorch fp16`` vs
    ``lean_pytorch fp16 compile=true`` hash to distinct keys.

    Missing ``overlap_samples`` defaults to 0 (older ``cpu_worker_sweep``
    writers didn't emit that field).

    ``n_cpu_workers_per_gpu`` distinguishes old serial-preprocess dual-GPU
    rows (default ``-1``) from new pipelined dual-GPU rows (which emit a
    positive integer per-shard pool size). Single-GPU rows don't carry this
    axis and default to ``-1`` too.

    ``infer_num_threads`` is only written by CPU-inference cpu_worker_sweep
    rows (``device="cpu"``). GPU-inference rows default it to ``-1`` so
    legacy rows hash the same way as before this axis was added.
    """
    label = row.get("model_label") or f"{row.get('model_parent')}/{row.get('model_child')}"
    overlap = row.get("overlap_samples")
    if overlap is None:
        overlap = 0
    extra_sig = _backend_extra_sig(row.get("backend_extra") or {})
    n_cpu_per_gpu = row.get("n_cpu_workers_per_gpu", -1)
    if n_cpu_per_gpu is None:
        n_cpu_per_gpu = -1
    infer_threads = row.get("infer_num_threads", -1)
    if infer_threads is None:
        infer_threads = -1
    return (
        row.get("kind"),
        row.get("backend"),
        row.get("dtype"),
        extra_sig,
        label,
        row.get("device"),
        row.get("n_stations", -1),
        row.get("batch_size", -1),
        overlap,
        row.get("n_cpu_workers", -1),
        n_cpu_per_gpu,
        int(infer_threads),
        row.get("repeat", -1),
    )


def _load_completed(path: Path, retry_errors: bool) -> set:
    """Parse the existing JSONL and return the set of already-done cell keys.

    Rows with ``kind == "error"`` are only counted as done if
    ``retry_errors`` is False. Malformed lines are skipped with a warning.

    Backward-compat: rows written before the dual-GPU split had
    ``kind="dual_gpu"`` for the (then-serial-preprocess) lean 2-GPU path.
    Those rows are now equivalent to the new ``kind="dual_gpu_serial"`` and
    are admitted as such so we don't re-run them on resume.
    """
    completed: set = set()
    if not path.exists():
        return completed
    n_total = 0
    n_err = 0
    n_env = 0
    n_legacy = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            LOG.warning("Skipping malformed JSONL line %d in %s", i, path)
            continue
        kind = row.get("kind")
        if kind == "env":
            n_env += 1
            continue  # env snapshots are metadata, not a benchmark cell
        n_total += 1
        if kind == "error":
            n_err += 1
            if retry_errors:
                continue

        # Legacy ``dual_gpu`` + lean backend + no n_cpu_workers_per_gpu field
        # is semantically the new ``dual_gpu_serial`` kind. Emit the key
        # under both names so resume accepts it.
        if (
            kind == "dual_gpu"
            and row.get("backend") != "baseline_annotate"
            and row.get("n_cpu_workers_per_gpu") in (None, -1)
        ):
            alias = dict(row)
            alias["kind"] = "dual_gpu_serial"
            completed.add(_row_key(alias))
            n_legacy += 1

        # Legacy CPU ``cpu_worker_sweep`` rows: before the resume-key fix, we
        # persisted the *resolved* thread count under ``infer_num_threads`` and
        # used the *requested* count (``-1`` for auto) in the cell key. On
        # resume, the row-side value (e.g. 126) then failed to match the new
        # writer's key (-1), causing the cell to be re-run. Detect those rows
        # (they have ``infer_num_threads > 0`` but no ``infer_num_threads_actual``)
        # and also admit the key under ``infer_num_threads=-1`` so a fresh run
        # against the same config properly skips them.
        if (
            kind == "cpu_worker_sweep"
            and row.get("device", "").startswith("cpu")
            and "infer_num_threads_actual" not in row
            and isinstance(row.get("infer_num_threads"), int)
            and row.get("infer_num_threads", -1) > 0
        ):
            alias = dict(row)
            alias["infer_num_threads"] = -1
            completed.add(_row_key(alias))
            n_legacy += 1

        completed.add(_row_key(row))
    LOG.info(
        "Resume: loaded %d prior rows (%d errors, %d env, %d legacy "
        "dual_gpu→dual_gpu_serial aliases) from %s; %d cells counted as done",
        n_total, n_err, n_env, n_legacy, path, len(completed),
    )
    return completed


def _make_cell_key(
    *, kind: str, backend_cfg: Dict[str, Any], parent: str, child: str,
    label: str, device: str, n_stations: int, batch_size: int,
    overlap: int, repeat: int, n_cpu_workers: int = -1,
    n_cpu_workers_per_gpu: int = -1, infer_num_threads: int = -1,
) -> CellKey:
    return (
        kind,
        backend_cfg["name"],
        backend_cfg.get("dtype", "fp32"),
        _backend_extra_sig(backend_cfg),
        label or f"{parent}/{child}",
        device,
        n_stations,
        batch_size,
        overlap,
        n_cpu_workers,
        n_cpu_workers_per_gpu,
        int(infer_num_threads),
        repeat,
    )


def _eqt_lean_fp16_incompatible(model_parent: str, backend_cfg: Dict[str, Any]) -> bool:
    """True for EQTransformer + lean_pytorch + fp16 (invalid; see ``EQT_LEAN_FP16_MESSAGE``)."""
    if model_parent != "EQTransformer":
        return False
    if backend_cfg.get("name") != "lean_pytorch":
        return False
    return backend_cfg.get("dtype") == "fp16"


def _driver_skipped_eqt_fp16_row(
    *,
    kind: str,
    key: CellKey,
    backend_cfg: Dict[str, Any],
    parent: str,
    child: str,
    label: str,
    device: str,
    n_stations: int,
    batch_size: int,
    overlap: int,
    repeat: int,
    dataset_dir: str,
    in_samples: int,
    n_cpu_workers: int = -1,
    n_cpu_workers_per_gpu: int = -1,
    infer_num_threads: int = -1,
) -> Dict[str, Any]:
    """Record the same logical ``BackendError`` as ``LeanPyTorch.load()`` without Ray.

    Uses the normal success ``kind`` (``single``, ``dual_gpu``, …) so
    ``_row_key`` matches and resume works. ``benchmark_status`` + ``error``
    let analysis treat the row as a tracked failure without ``ActorDiedError``.
    """
    from rapid.backends.lean_pytorch import EQT_LEAN_FP16_MESSAGE

    row: Dict[str, Any] = {
        "kind": kind,
        "schema_version": SCHEMA_VERSION,
        "trial_uid": _trial_uid(key),
        "backend": backend_cfg["name"],
        "dtype": backend_cfg.get("dtype", "fp32"),
        "backend_extra": {
            k: v for k, v in backend_cfg.items()
            if k not in ("name", "dtype", "device")
        },
        "model_parent": parent,
        "model_child": child,
        "model_label": label,
        "device": device,
        "dataset_dir": dataset_dir,
        "dataset_label": Path(dataset_dir).name if dataset_dir else None,
        "n_stations": n_stations,
        "n_windows": -1,
        "in_samples": in_samples,
        "batch_size": batch_size,
        "overlap_samples": overlap,
        "repeat": repeat,
        "benchmark_status": "skipped_incompatible",
        "skip_reason": "eqt_lean_fp16",
        "error": f"BackendError: {EQT_LEAN_FP16_MESSAGE}",
        "traceback": (
            "Driver-side skip: same error as LeanPyTorch.load(); trial not run "
            "so Ray actors are never spawned (no ActorDiedError)."
        ),
        "wall_time_s": None,
        "total_s": None,
        "stage_times_s": {},
        "end_to_end_wall_s": None,
        "throughput_stations_per_s": None,
        "throughput_windows_per_s": None,
        "throughput_samples_per_s": None,
        "timestamp_s": time.time(),
    }
    if n_cpu_workers != -1:
        row["n_cpu_workers"] = n_cpu_workers
    if n_cpu_workers_per_gpu != -1:
        row["n_cpu_workers_per_gpu"] = n_cpu_workers_per_gpu
    if device.startswith("cpu"):
        row["infer_num_threads"] = infer_num_threads
        row["infer_num_threads_actual"] = None
    return row


def _trial_uid(key: CellKey) -> str:
    """Short stable hash of a cell key, for easy row referencing in analysis."""
    return hashlib.sha1(json.dumps(key, default=str).encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _gpu_mem_reset(device: str) -> None:
    """Reset peak-memory counter for a single CUDA device (no-op on CPU)."""
    if not device or not device.startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
    except Exception:
        pass


def _gpu_mem_reset_all() -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device=i)
    except Exception:
        pass


def _gpu_mem_peak(device: str) -> Optional[int]:
    if not device or not device.startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.max_memory_allocated(device=device))
    except Exception:
        return None


def _gpu_mem_peak_all() -> Dict[str, int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            f"cuda:{i}": int(torch.cuda.max_memory_allocated(device=i))
            for i in range(torch.cuda.device_count())
        }
    except Exception:
        return {}


def _derive_throughput(*, total_s: float, n_stations: int, n_windows: int,
                       in_samples: Optional[int]) -> Dict[str, float]:
    """Standard derived throughput metrics. All return ``nan`` if not computable."""
    out: Dict[str, float] = {
        "throughput_stations_per_s": float("nan"),
        "throughput_windows_per_s": float("nan"),
        "throughput_samples_per_s": float("nan"),
    }
    if total_s is None or total_s <= 0:
        return out
    if n_stations and n_stations > 0:
        out["throughput_stations_per_s"] = n_stations / total_s
    if n_windows is not None and n_windows > 0:
        out["throughput_windows_per_s"] = n_windows / total_s
        if in_samples is not None and in_samples > 0:
            out["throughput_samples_per_s"] = (n_windows * in_samples) / total_s
    return out


def _model_in_samples(parent: str, child: str) -> Optional[int]:
    """Best-effort lookup of ``in_samples`` for a (parent, child) pair.

    Cached per-process so we don't re-instantiate SeisBench weights.
    """
    cache = _model_in_samples._cache  # type: ignore[attr-defined]
    key = (parent, child)
    if key in cache:
        return cache[key]
    try:
        import seisbench.models as sbm

        cls = getattr(sbm, parent)
        m = cls.from_pretrained(child)
        val = int(getattr(m, "in_samples", 0)) or None
    except Exception:
        val = None
    cache[key] = val
    return val


_model_in_samples._cache = {}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Environment snapshot
# ---------------------------------------------------------------------------


def _env_snapshot(cfg: MatrixConfig) -> Dict[str, Any]:
    """Capture hardware/software/config context for one matrix run."""
    info: Dict[str, Any] = {
        "kind": "env",
        "schema_version": SCHEMA_VERSION,
        "timestamp_s": time.time(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "dataset_dir": cfg.dataset_dir,
        "dataset_label": Path(cfg.dataset_dir).name,
        "config": asdict(cfg),
    }
    try:
        info["cpu_affinity"] = list(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            try:
                info["cudnn_version"] = int(torch.backends.cudnn.version() or 0)
            except Exception:
                info["cudnn_version"] = None
            info["gpus"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_mem_bytes": int(torch.cuda.get_device_properties(i).total_memory),
                    "capability": list(torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as e:
        info["torch_error"] = repr(e)
    try:
        import seisbench  # type: ignore

        info["seisbench_version"] = getattr(seisbench, "__version__", "unknown")
    except Exception:
        pass
    return info


def _short_dtype(b: Dict[str, Any]) -> str:
    return b.get("dtype", "fp32")


def _backend_key(b: Dict[str, Any]) -> str:
    d = _short_dtype(b)
    if b["name"] == "lean_pytorch" and b.get("compile"):
        return f"{b['name']}_{d}_compile"
    return f"{b['name']}_{d}"


def _backends_for_device(cfg: "MatrixConfig", device: str) -> List[Dict[str, Any]]:
    """Effective backend list for this device.

    Uses ``cpu_backends_override`` when ``device == "cpu"`` and the
    override is non-empty; otherwise falls back to ``cfg.backends``.
    This lets us, e.g., drop ``lean_pytorch/fp16`` on CPU (most CPUs
    have no native FP16 and the PyTorch CPU fallback is slower than
    FP32) without affecting GPU coverage.
    """
    if device == "cpu" and cfg.cpu_backends_override:
        return list(cfg.cpu_backends_override)
    return list(cfg.backends)


def _batch_sizes_for_device(cfg: "MatrixConfig", device: str) -> List[int]:
    """Effective batch-size axis for this device.

    Batch size barely moves CPU wall time once you're past the point
    where the megabatch amortises Python overhead, so trimming the CPU
    axis saves a lot of trial time without losing information. GPU
    trials still use the full ``cfg.batch_sizes`` list.
    """
    if device == "cpu" and cfg.cpu_batch_sizes_override:
        return list(cfg.cpu_batch_sizes_override)
    return list(cfg.batch_sizes)


def run_matrix(
    cfg: MatrixConfig,
    dry_run: bool = False,
) -> Path:
    """Execute the configured sweep and append JSONL rows to ``cfg.output_jsonl``.

    When ``cfg.resume`` is True (default) the existing output file is scanned
    for already-completed cells and those repeats are skipped. Set
    ``cfg.retry_errors`` to False to also treat prior error rows as done.
    """
    from .backends import available_backends, get_backend
    from .data import load_all_streams, select_stations
    from .runners.single_gpu import run_baseline_single, run_lean_single
    from .runners.cpu_worker_sweep import run_cpu_worker_sweep
    from .runners.dual_gpu import run_dual_gpu
    from .runners.pipelined import run_baseline_dual_gpu, run_pipelined_dual_gpu

    out_path = Path(cfg.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set = _load_completed(out_path, cfg.retry_errors) if cfg.resume else set()

    # Write one env-info row per invocation so every JSONL is self-describing.
    # ``kind="env"`` rows are ignored by ``_load_completed`` (they don't carry
    # a valid cell key) but are preserved in the file for post-hoc analysis.
    if not dry_run:
        env_row = _env_snapshot(cfg)
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(env_row, default=str) + "\n")

    # ------------------------------------------------------------------
    # Single upfront Ray init for the entire matrix.
    #
    # Rationale (see the hang we diagnosed): individual runners each call
    # ``ray.init(..., ignore_reinit_error=True)`` with their own resource
    # requests. The FIRST call wins. If a CPU-inference trial runs first
    # and initialises with ``num_gpus=0``, every subsequent GPU actor
    # request (dual-GPU baseline, pipelined, dual_gpu_serial) queues
    # forever as ``pending`` because the cluster has no GPU resources.
    #
    # Solution: init Ray once here with the *max* resources the full
    # matrix will ever need. Downstream ``ray.init(..., ignore_reinit_error=True)``
    # calls then become true no-ops but the cluster correctly reports
    # every resource every trial needs.
    # ------------------------------------------------------------------
    if not dry_run:
        try:
            import ray as _ray
            import torch as _torch

            if not _ray.is_initialized():
                _n_cpu = os.cpu_count() or 1
                _n_gpu = 0
                try:
                    _n_gpu = int(_torch.cuda.device_count())
                except Exception:
                    _n_gpu = 0
                LOG.info(
                    "Initialising Ray once for the whole matrix: "
                    "num_cpus=%d num_gpus=%d", _n_cpu, _n_gpu,
                )
                _ray.init(
                    num_cpus=_n_cpu,
                    num_gpus=_n_gpu,
                    ignore_reinit_error=True,
                    log_to_driver=False,
                )
        except Exception as _e:  # noqa: BLE001
            LOG.warning(
                "Failed to pre-initialise Ray (%s); individual runners will "
                "try to init it themselves. This may cause GPU scheduling "
                "deadlocks if a CPU-only trial runs first.", _e,
            )

    # Only load stream data for an (n_stations) slice if there's still work
    # to do for it — avoids reading 580 miniSEED files when every cell is done.
    stream_cache: Dict[Tuple[str, int], List] = {}

    def _get_streams(n: int):
        key = ("__shared__", n)
        if key not in stream_cache:
            stations = select_stations(cfg.dataset_dir, n)
            stream_cache[key] = load_all_streams(cfg.dataset_dir, stations)
        return stream_cache[key]

    all_rows: List[Dict[str, Any]] = []
    n_skipped = 0

    with out_path.open("a", encoding="utf-8") as fh:

        for model_spec in cfg.models:
            parent = model_spec["parent"]
            child = model_spec["child"]
            label = model_spec.get("label", f"{parent}/{child}")

            for n_stations in cfg.n_stations_list:
                streams = None  # load lazily on first cell that actually runs

                def _ensure_streams():
                    nonlocal streams
                    if streams is None:
                        streams = _get_streams(n_stations)
                    return streams

                # Iterate devices on the outside so we can pick a
                # device-specific backend / batch-size list without
                # extra bookkeeping inside the backend loop.
                for device in cfg.devices:
                    effective_backends = _backends_for_device(cfg, device)
                    effective_batch_sizes = _batch_sizes_for_device(cfg, device)

                    for backend_cfg in effective_backends:
                        if backend_cfg["name"] not in available_backends():
                            LOG.warning(
                                "Skipping unavailable backend: %s", backend_cfg["name"]
                            )
                            continue
                        # Baseline: one single-device pass per device, then
                        # fall through to the 2-GPU baseline block below if
                        # the current device is a GPU. No batch-size sweeps.
                        if backend_cfg["name"] == "baseline_annotate":
                            rows, sk = _bench_baseline(
                                backend_cfg, parent, child, label,
                                _ensure_streams, device, cfg, dry_run,
                                completed, n_stations,
                            )
                            n_skipped += sk
                            for r in rows:
                                fh.write(json.dumps(r) + "\n")
                                fh.flush()
                                all_rows.append(r)
                            # Fair 2-GPU baseline: SeisBench annotate() run in
                            # parallel on 2 GPUs via ``run_baseline_dual_gpu``.
                            # Only meaningful on a CUDA device.
                            if cfg.dual_gpu and device.startswith("cuda"):
                                rows, sk = _bench_dual_gpu(
                                    backend_cfg, parent, child, label,
                                    _ensure_streams, -1, cfg, dry_run,
                                    completed, n_stations,
                                    n_cpu_workers_per_gpu=-1,
                                )
                                n_skipped += sk
                                for r in rows:
                                    fh.write(json.dumps(r) + "\n")
                                    fh.flush()
                                    all_rows.append(r)
                            continue

                        for overlap in cfg.overlap_sweep:
                            for bs in effective_batch_sizes:
                                rows, sk = _bench_lean_single(
                                    backend_cfg, parent, child, label,
                                    _ensure_streams, device, bs, overlap,
                                    cfg, dry_run, completed, n_stations,
                                )
                                n_skipped += sk
                                for r in rows:
                                    fh.write(json.dumps(r) + "\n")
                                    fh.flush()
                                    all_rows.append(r)

                        # Dual-GPU only meaningful if device is a GPU. The
                        # baseline path is handled above (it ``continue``s
                        # out of the backend loop); these blocks are lean-only
                        # and emit two distinct kinds for traceable evolution:
                        #
                        #   - ``dual_gpu_serial``  : old path, one ``run_lean_single``
                        #     per shard (serial preprocess inside each actor).
                        #   - ``dual_gpu``         : new pipelined path, each shard
                        #     runs a CPU preprocess pool + megabatch inference
                        #     actor via ``run_pipelined_dual_gpu``.
                        if cfg.dual_gpu and device.startswith("cuda"):
                            if cfg.dual_gpu_serial:
                                for bs in effective_batch_sizes:
                                    rows, sk = _bench_dual_gpu_serial(
                                        backend_cfg, parent, child, label,
                                        _ensure_streams, bs, cfg, dry_run,
                                        completed, n_stations,
                                    )
                                    n_skipped += sk
                                    for r in rows:
                                        fh.write(json.dumps(r) + "\n")
                                        fh.flush()
                                        all_rows.append(r)

                            dg_sweep = list(cfg.dual_gpu_cpu_workers) or [8]
                            for n_cpu_g in dg_sweep:
                                for bs in effective_batch_sizes:
                                    rows, sk = _bench_dual_gpu(
                                        backend_cfg, parent, child, label,
                                        _ensure_streams, bs, cfg, dry_run,
                                        completed, n_stations,
                                        n_cpu_workers_per_gpu=n_cpu_g,
                                    )
                                    n_skipped += sk
                                    for r in rows:
                                        fh.write(json.dumps(r) + "\n")
                                        fh.flush()
                                        all_rows.append(r)

                        # CPU-worker sweep on GPU runs (original behavior).
                        if cfg.cpu_worker_sweep and device.startswith("cuda"):
                            for n_cpu in cfg.cpu_worker_sweep:
                                for bs in effective_batch_sizes:
                                    rows, sk = _bench_cpu_workers(
                                        backend_cfg, parent, child, label,
                                        _ensure_streams, bs, n_cpu, cfg,
                                        dry_run, completed, n_stations,
                                        device="cuda:0",
                                        infer_num_threads=-1,
                                    )
                                    n_skipped += sk
                                    for r in rows:
                                        fh.write(json.dumps(r) + "\n")
                                        fh.flush()
                                        all_rows.append(r)

                        # CPU-inference variant of the same sweep: parallel
                        # preprocess + megabatch forward but with the
                        # inference actor pinned to CPU (and a controlled
                        # BLAS thread count). New cell keys don't collide
                        # with existing GPU rows because ``device`` is part
                        # of the key.
                        if (
                            cfg.cpu_worker_sweep
                            and cfg.cpu_worker_sweep_on_cpu
                            and device == "cpu"
                        ):
                            # If an explicit thread-count axis is provided,
                            # cross it with ``n_cpu_workers``; otherwise use
                            # -1 (auto-pick). We skip ``baseline_annotate``
                            # because the runner only works with lean-style
                            # megabatch backends.
                            if backend_cfg["name"] == "baseline_annotate":
                                thread_sweep: List[int] = []
                            else:
                                thread_sweep = list(cfg.cpu_infer_threads) or [-1]
                            for n_cpu in cfg.cpu_worker_sweep:
                                for inf_th in thread_sweep:
                                    for bs in effective_batch_sizes:
                                        rows, sk = _bench_cpu_workers(
                                            backend_cfg, parent, child, label,
                                            _ensure_streams, bs, n_cpu, cfg,
                                            dry_run, completed, n_stations,
                                            device="cpu",
                                            infer_num_threads=int(inf_th),
                                        )
                                        n_skipped += sk
                                        for r in rows:
                                            fh.write(json.dumps(r) + "\n")
                                            fh.flush()
                                            all_rows.append(r)

    LOG.info(
        "Wrote %d new rows to %s (skipped %d already-done repeats)",
        len(all_rows), out_path, n_skipped,
    )
    return out_path


# ---------------------------------------------------------------------------
# Per-shape benchmark helpers
# ---------------------------------------------------------------------------


def _bench_baseline(backend_cfg, parent, child, label, streams_fn, device,
                    cfg, dry, completed, n_stations):
    from .backends import get_backend
    from .runners.single_gpu import run_baseline_single

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    if dry:
        return rows, n_skipped
    streams = None
    for repeat in range(cfg.repeats):
        key = _make_cell_key(
            kind="baseline", backend_cfg=backend_cfg, parent=parent,
            child=child, label=label, device=device,
            n_stations=n_stations, batch_size=-1, overlap=-1, repeat=repeat,
        )
        if key in completed:
            n_skipped += 1
            continue
        try:
            if streams is None:
                streams = streams_fn()
            # end_to_end_wall_s: total wall time from "I asked for inference"
            # through "all outputs back", including model load and any setup.
            # This is the number a user feels on a cold start. wall_time_s
            # below only covers the compute region — matches how we compare
            # GPU trials against each other.
            _t_e2e = time.perf_counter()
            cls = get_backend(backend_cfg["name"])
            bk = cls(parent_model=parent, child_model=child, device=device, dtype="fp32")
            bk.load()
            _gpu_mem_reset(device)
            from .memory import RSSPoller
            from .telemetry import GPUWatcher
            _rss = RSSPoller(); _rss.start()
            _gpuw = GPUWatcher(
                device_indices=[int(device.split(":")[1])] if device.startswith("cuda") else [],
                interval_s=0.2,
            )
            _gpuw.start()
            result = run_baseline_single(bk, streams)
            peak_mem = _gpu_mem_peak(device)
            _rss_stats = _rss.stop()
            _tel = _gpuw.stop()
            _end_to_end = time.perf_counter() - _t_e2e
            bk.close()
            rows.append(_row(
                kind="baseline",
                backend_cfg=backend_cfg, parent=parent, child=child, label=label,
                device=device, n_stations=result.n_stations,
                batch_size=-1, overlap=-1, repeat=repeat,
                stage_times=result.stage_times, total_s=result.total_s,
                n_windows=result.n_windows,
                dataset_dir=cfg.dataset_dir,
                peak_gpu_mem_bytes=peak_mem,
                peak_cpu_rss_bytes=_rss_stats.peak_rss_bytes,
                delta_cpu_rss_bytes=_rss_stats.delta_rss_bytes,
                end_to_end_wall_s=_end_to_end,
                nvml_telemetry=_tel.as_row_fields(),
                trial_uid=_trial_uid(key),
            ))
        except Exception as e:
            rows.append(_error_row(
                backend_cfg, parent, child, label, device, repeat, e,
                n_stations=n_stations, batch_size=-1, overlap=-1,
                dataset_dir=cfg.dataset_dir, trial_uid=_trial_uid(key),
            ))
    return rows, n_skipped


def _bench_lean_single(backend_cfg, parent, child, label, streams_fn, device,
                       bs, overlap, cfg, dry, completed, n_stations):
    from .backends import get_backend
    from .runners.single_gpu import run_lean_single

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    if dry:
        return rows, n_skipped
    streams = None
    for repeat in range(cfg.repeats):
        key = _make_cell_key(
            kind="single", backend_cfg=backend_cfg, parent=parent,
            child=child, label=label, device=device,
            n_stations=n_stations, batch_size=bs, overlap=overlap, repeat=repeat,
        )
        if key in completed:
            n_skipped += 1
            continue
        if _eqt_lean_fp16_incompatible(parent, backend_cfg):
            rows.append(_driver_skipped_eqt_fp16_row(
                kind="single", key=key, backend_cfg=backend_cfg,
                parent=parent, child=child, label=label, device=device,
                n_stations=n_stations, batch_size=bs, overlap=overlap, repeat=repeat,
                dataset_dir=cfg.dataset_dir,
                in_samples=_model_in_samples(parent, child),
            ))
            continue
        try:
            if streams is None:
                streams = streams_fn()
            cls = get_backend(backend_cfg["name"])
            init_kwargs = {k: v for k, v in backend_cfg.items() if k != "name"}
            init_kwargs.setdefault("device", device)
            _t_e2e = time.perf_counter()
            bk = cls(parent_model=parent, child_model=child, **init_kwargs)
            bk.load()
            _gpu_mem_reset(device)
            from .memory import RSSPoller
            from .telemetry import GPUWatcher
            _rss = RSSPoller(); _rss.start()
            _gpuw = GPUWatcher(
                device_indices=[int(device.split(":")[1])] if device.startswith("cuda") else [],
                interval_s=0.2,
            )
            _gpuw.start()
            result = run_lean_single(
                bk, streams, batch_size=bs, overlap_samples=overlap,
                warmup_iters=cfg.warmup_iters,
            )
            peak_mem = _gpu_mem_peak(device)
            _rss_stats = _rss.stop()
            _tel = _gpuw.stop()
            _end_to_end = time.perf_counter() - _t_e2e
            in_samples = getattr(bk, "in_samples", None)
            bk.close()
            rows.append(_row(
                kind="single",
                backend_cfg=backend_cfg, parent=parent, child=child, label=label,
                device=device, n_stations=result.n_stations,
                batch_size=bs, overlap=overlap, repeat=repeat,
                stage_times=result.stage_times, total_s=result.total_s,
                n_windows=result.n_windows,
                dataset_dir=cfg.dataset_dir,
                peak_gpu_mem_bytes=peak_mem,
                peak_cpu_rss_bytes=_rss_stats.peak_rss_bytes,
                delta_cpu_rss_bytes=_rss_stats.delta_rss_bytes,
                end_to_end_wall_s=_end_to_end,
                nvml_telemetry=_tel.as_row_fields(),
                in_samples=in_samples,
                trial_uid=_trial_uid(key),
            ))
        except Exception as e:
            rows.append(_error_row(
                backend_cfg, parent, child, label, device, repeat, e,
                n_stations=n_stations, batch_size=bs, overlap=overlap,
                dataset_dir=cfg.dataset_dir, trial_uid=_trial_uid(key),
            ))
    return rows, n_skipped


def _bench_dual_gpu(backend_cfg, parent, child, label, streams_fn, bs,
                    cfg, dry, completed, n_stations, *,
                    n_cpu_workers_per_gpu: int = -1):
    """Dual-GPU benchmark.

    Uses ``run_pipelined_dual_gpu`` for lean backends (each GPU shard runs
    its own ``n_cpu_workers_per_gpu`` preprocess pool + inference actor) and
    ``run_baseline_dual_gpu`` for ``baseline_annotate`` (SeisBench's internal
    single-threaded preprocess, one model per GPU; the fair 2-GPU baseline).
    """
    from .runners.pipelined import run_baseline_dual_gpu, run_pipelined_dual_gpu

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    if dry:
        return rows, n_skipped
    streams = None
    in_samples = _model_in_samples(parent, child)
    is_baseline = backend_cfg["name"] == "baseline_annotate"
    for repeat in range(cfg.repeats):
        key = _make_cell_key(
            kind="dual_gpu", backend_cfg=backend_cfg, parent=parent,
            child=child, label=label, device="cuda:0+cuda:1",
            n_stations=n_stations, batch_size=bs, overlap=0, repeat=repeat,
            n_cpu_workers_per_gpu=n_cpu_workers_per_gpu,
        )
        if key in completed:
            n_skipped += 1
            continue
        if (not is_baseline) and _eqt_lean_fp16_incompatible(parent, backend_cfg):
            rows.append(_driver_skipped_eqt_fp16_row(
                kind="dual_gpu", key=key, backend_cfg=backend_cfg,
                parent=parent, child=child, label=label, device="cuda:0+cuda:1",
                n_stations=n_stations, batch_size=bs, overlap=0, repeat=repeat,
                dataset_dir=cfg.dataset_dir, in_samples=in_samples,
                n_cpu_workers_per_gpu=n_cpu_workers_per_gpu,
            ))
            continue
        try:
            if streams is None:
                streams = streams_fn()
            _gpu_mem_reset_all()
            from .memory import RSSPoller
            from .telemetry import GPUWatcher
            _rss = RSSPoller(); _rss.start()
            _gpuw = GPUWatcher(device_indices=[0, 1], interval_s=0.2)
            _gpuw.start()
            if is_baseline:
                result = run_baseline_dual_gpu(
                    parent_model=parent, child_model=child,
                    streams=streams, num_gpus=2,
                )
            else:
                result = run_pipelined_dual_gpu(
                    parent_model=parent, child_model=child,
                    streams=streams,
                    n_cpu_workers_per_gpu=max(1, int(n_cpu_workers_per_gpu)),
                    batch_size=bs, overlap_samples=0,
                    dtype=backend_cfg.get("dtype", "fp32"),
                    backend_name=backend_cfg["name"],
                    num_gpus=2,
                )
            _rss_stats = _rss.stop()
            _tel = _gpuw.stop()
            # Driver-side cuda:0/cuda:1 peak reflects nothing meaningful (the
            # actual inference runs in Ray worker processes) but record it so
            # the column is never missing. NVML peak-mem (``nvml_peak_mem_used_bytes``)
            # *does* see actor allocations because it reads the global
            # device memory state, so prefer that for dual-GPU analysis.
            per_gpu_mem = _gpu_mem_peak_all()
            tput = _derive_throughput(
                total_s=result.wall_time_s,
                n_stations=result.sum_stations,
                n_windows=result.sum_windows,
                in_samples=in_samples,
            )
            row = {
                "kind": "dual_gpu",
                "schema_version": SCHEMA_VERSION,
                "trial_uid": _trial_uid(key),
                "backend": backend_cfg["name"],
                "dtype": backend_cfg.get("dtype", "fp32"),
                "backend_extra": {
                    k: v for k, v in backend_cfg.items()
                    if k not in ("name", "dtype", "device")
                },
                "model_parent": parent, "model_child": child, "model_label": label,
                "device": "cuda:0+cuda:1",
                "dataset_dir": cfg.dataset_dir,
                "dataset_label": Path(cfg.dataset_dir).name,
                "n_stations": result.sum_stations,
                "n_windows": result.sum_windows,
                "in_samples": in_samples,
                "batch_size": bs, "overlap_samples": 0,
                "repeat": repeat,
                # wall_time_s: critical-path compute time (apples-to-apples
                # with single-GPU cpu_worker_sweep). end_to_end_wall_s:
                # real first-call latency incl. actor setup / model load.
                "wall_time_s": result.wall_time_s,
                "end_to_end_wall_s": getattr(result, "end_to_end_wall_s", result.wall_time_s),
                "total_s": result.wall_time_s,
                "per_gpu": result.per_gpu,
                "n_cpu_workers_per_gpu": n_cpu_workers_per_gpu,
                "gpu_utilization_pct": result.gpu_utilization_pct(),
                "peak_gpu_mem_bytes_driver": per_gpu_mem,
                # Driver-process RSS. With Ray, the tensors live in
                # actor processes, so the driver RSS is dominated by
                # whatever the orchestration code holds (stream
                # dicts, concatenated result buffers). Still useful as
                # an upper-bound floor; actor RSS is not directly
                # reachable here without teaching the runners to
                # report it, which is a follow-up.
                "peak_cpu_rss_bytes": _rss_stats.peak_rss_bytes,
                "delta_cpu_rss_bytes": _rss_stats.delta_rss_bytes,
                # NVML reads global device state so these peaks *do*
                # include Ray-actor allocations, unlike the driver-
                # side torch peak above. Also gives per-GPU util %,
                # power draw, and integrated energy.
                **_tel.as_row_fields(),
                **tput,
                "timestamp_s": time.time(),
            }
            rows.append(row)
        except Exception as e:
            err_row = _error_row(
                backend_cfg, parent, child, label, "cuda:0+cuda:1", repeat, e,
                n_stations=n_stations, batch_size=bs, overlap=0,
                dataset_dir=cfg.dataset_dir, trial_uid=_trial_uid(key),
            )
            err_row["n_cpu_workers_per_gpu"] = n_cpu_workers_per_gpu
            rows.append(err_row)
    return rows, n_skipped


def _bench_dual_gpu_serial(backend_cfg, parent, child, label, streams_fn, bs,
                           cfg, dry, completed, n_stations):
    """Old serial-preprocess lean 2-GPU benchmark (``kind="dual_gpu_serial"``).

    Each GPU shard runs a single ``run_lean_single`` with single-threaded
    preprocess inside the actor — no CPU pool. This is the intermediate step
    between "lean 1-GPU serial" and "lean 2-GPU pipelined", useful for
    quantifying how much of the pipelined dual-GPU win comes from the 2nd
    GPU vs. from the per-shard CPU pool.

    Baseline backends are skipped here (they have their own ``kind="dual_gpu"``
    path via ``run_baseline_dual_gpu``).
    """
    from .runners.dual_gpu import run_dual_gpu

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    if dry or backend_cfg["name"] == "baseline_annotate":
        return rows, n_skipped
    streams = None
    in_samples = _model_in_samples(parent, child)
    for repeat in range(cfg.repeats):
        key = _make_cell_key(
            kind="dual_gpu_serial", backend_cfg=backend_cfg, parent=parent,
            child=child, label=label, device="cuda:0+cuda:1",
            n_stations=n_stations, batch_size=bs, overlap=0, repeat=repeat,
            n_cpu_workers_per_gpu=-1,
        )
        if key in completed:
            n_skipped += 1
            continue
        if _eqt_lean_fp16_incompatible(parent, backend_cfg):
            rows.append(_driver_skipped_eqt_fp16_row(
                kind="dual_gpu_serial", key=key, backend_cfg=backend_cfg,
                parent=parent, child=child, label=label, device="cuda:0+cuda:1",
                n_stations=n_stations, batch_size=bs, overlap=0, repeat=repeat,
                dataset_dir=cfg.dataset_dir, in_samples=in_samples,
            ))
            continue
        try:
            if streams is None:
                streams = streams_fn()
            bk_kwargs = {
                k: v for k, v in backend_cfg.items()
                if k not in ("name", "dtype", "device")
            }
            _gpu_mem_reset_all()
            from .memory import RSSPoller
            from .telemetry import GPUWatcher
            _rss = RSSPoller(); _rss.start()
            _gpuw = GPUWatcher(device_indices=[0, 1], interval_s=0.2)
            _gpuw.start()
            result = run_dual_gpu(
                parent_model=parent, child_model=child,
                streams=streams,
                backend_name=backend_cfg["name"],
                dtype=backend_cfg.get("dtype", "fp32"),
                batch_size=bs, overlap_samples=0,
                backend_kwargs=bk_kwargs,
            )
            _rss_stats = _rss.stop()
            _tel = _gpuw.stop()
            per_gpu_mem = _gpu_mem_peak_all()
            per_actor_stages = [r.stage_times for r in result.per_actor]
            tput = _derive_throughput(
                total_s=result.wall_time_s,
                n_stations=result.sum_stations,
                n_windows=result.sum_windows,
                in_samples=in_samples,
            )
            row = {
                "kind": "dual_gpu_serial",
                "schema_version": SCHEMA_VERSION,
                "trial_uid": _trial_uid(key),
                "backend": backend_cfg["name"],
                "dtype": backend_cfg.get("dtype", "fp32"),
                "backend_extra": {
                    k: v for k, v in backend_cfg.items()
                    if k not in ("name", "dtype", "device")
                },
                "model_parent": parent, "model_child": child, "model_label": label,
                "device": "cuda:0+cuda:1",
                "dataset_dir": cfg.dataset_dir,
                "dataset_label": Path(cfg.dataset_dir).name,
                "n_stations": result.sum_stations,
                "n_windows": result.sum_windows,
                "in_samples": in_samples,
                "batch_size": bs, "overlap_samples": 0,
                "repeat": repeat,
                "wall_time_s": result.wall_time_s,
                "total_s": result.wall_time_s,
                "per_actor_stage_times": per_actor_stages,
                "peak_gpu_mem_bytes_driver": per_gpu_mem,
                "peak_cpu_rss_bytes": _rss_stats.peak_rss_bytes,
                "delta_cpu_rss_bytes": _rss_stats.delta_rss_bytes,
                **_tel.as_row_fields(),
                **tput,
                "timestamp_s": time.time(),
            }
            rows.append(row)
        except Exception as e:
            err_row = _error_row(
                backend_cfg, parent, child, label, "cuda:0+cuda:1", repeat, e,
                n_stations=n_stations, batch_size=bs, overlap=0,
                dataset_dir=cfg.dataset_dir, trial_uid=_trial_uid(key),
            )
            # Tag the error row with the same kind for accurate resume.
            err_row["kind"] = "dual_gpu_serial"
            rows.append(err_row)
    return rows, n_skipped


def _bench_cpu_workers(backend_cfg, parent, child, label, streams_fn, bs,
                       n_cpu, cfg, dry, completed, n_stations,
                       *, device: str = "cuda:0",
                       infer_num_threads: int = -1):
    """Pipelined preprocess pool + megabatch inference on ``device``.

    ``device="cuda:0"``  → original GPU sweep (no thread-count axis).
    ``device="cpu"``     → CPU inference path. ``infer_num_threads`` gates
                            the BLAS thread count of the inference actor.
                            ``-1`` tells the runner to auto-split the box.
    """
    from .runners.cpu_worker_sweep import run_cpu_worker_sweep

    rows: List[Dict[str, Any]] = []
    n_skipped = 0
    if dry:
        return rows, n_skipped
    streams = None
    in_samples = _model_in_samples(parent, child)
    is_cpu = device.startswith("cpu")
    # The cell-key field for ``infer_num_threads`` stays ``-1`` on GPU rows so
    # legacy GPU entries (which never had this axis) continue to hash the same.
    key_infer_threads = int(infer_num_threads) if is_cpu else -1
    for repeat in range(cfg.repeats):
        key = _make_cell_key(
            kind="cpu_worker_sweep", backend_cfg=backend_cfg, parent=parent,
            child=child, label=label, device=device,
            n_stations=n_stations, batch_size=bs, overlap=0, repeat=repeat,
            n_cpu_workers=n_cpu,
            infer_num_threads=key_infer_threads,
        )
        if key in completed:
            n_skipped += 1
            continue
        if _eqt_lean_fp16_incompatible(parent, backend_cfg):
            rows.append(_driver_skipped_eqt_fp16_row(
                kind="cpu_worker_sweep", key=key, backend_cfg=backend_cfg,
                parent=parent, child=child, label=label, device=device,
                n_stations=n_stations, batch_size=bs, overlap=0, repeat=repeat,
                dataset_dir=cfg.dataset_dir, in_samples=in_samples,
                n_cpu_workers=n_cpu,
                infer_num_threads=key_infer_threads if is_cpu else -1,
            ))
            continue
        try:
            if streams is None:
                streams = streams_fn()
            # end_to_end_wall_s wraps the whole function including Ray
            # actor spin-up, model load, warm-up, and actor teardown —
            # the first-call latency a user actually feels. Compare
            # against ``wall_time_s`` (critical path only) to see how
            # much Ray setup cost there is.
            _t_e2e = time.perf_counter()
            from .telemetry import GPUWatcher
            _gpuw = GPUWatcher(
                device_indices=[int(device.split(":")[1])] if device.startswith("cuda") else [],
                interval_s=0.2,
            )
            _gpuw.start()
            result = run_cpu_worker_sweep(
                parent_model=parent, child_model=child,
                streams=streams,
                n_cpu_workers=n_cpu,
                batch_size=bs,
                overlap_samples=0,
                dtype=backend_cfg.get("dtype", "fp32"),
                backend_name=backend_cfg["name"],
                device=device,
                infer_num_threads=(
                    None if (not is_cpu or infer_num_threads == -1)
                    else int(infer_num_threads)
                ),
            )
            _tel = _gpuw.stop()
            _end_to_end = time.perf_counter() - _t_e2e
            tput = _derive_throughput(
                total_s=result.wall_time_s,
                n_stations=result.n_stations,
                n_windows=result.n_windows,
                in_samples=in_samples,
            )
            row = {
                "kind": "cpu_worker_sweep",
                "schema_version": SCHEMA_VERSION,
                "trial_uid": _trial_uid(key),
                "backend": backend_cfg["name"],
                "dtype": backend_cfg.get("dtype", "fp32"),
                "backend_extra": {
                    k: v for k, v in backend_cfg.items()
                    if k not in ("name", "dtype", "device")
                },
                "model_parent": parent, "model_child": child, "model_label": label,
                "device": device,
                "dataset_dir": cfg.dataset_dir,
                "dataset_label": Path(cfg.dataset_dir).name,
                "n_cpu_workers": n_cpu,
                # CPU rows only (GPU keeps these out so JSONL doesn't balloon
                # with ``-1`` noise). We persist TWO fields:
                #   ``infer_num_threads``  : the cell-key axis (what the user
                #                            asked for; ``-1`` means "auto").
                #                            MUST match what ``_make_cell_key``
                #                            used so resume round-trips.
                #   ``infer_num_threads_actual``: the runner's resolved value
                #                            after auto-split. Use for analysis.
                **(
                    {
                        "infer_num_threads": key_infer_threads,
                        "infer_num_threads_actual": getattr(
                            result, "infer_num_threads", -1
                        ),
                    }
                    if is_cpu
                    else {}
                ),
                "n_stations": result.n_stations,
                "n_windows": result.n_windows,
                "in_samples": in_samples,
                "batch_size": bs,
                "overlap_samples": 0,
                "repeat": repeat,
                "wall_time_s": result.wall_time_s,
                "total_s": result.wall_time_s,
                # Field name kept as gpu_forward_s / gpu_idle_s for schema
                # parity with GPU rows; on CPU rows they mean "inference
                # forward time" / "inference-idle time".
                "gpu_forward_s": result.gpu_forward_s,
                "gpu_idle_s": result.gpu_idle_s,
                "preprocess_total_s": result.preprocess_total_s,
                "n_gpu_submits": getattr(result, "n_gpu_submits", None),
                "gpu_utilization_pct": (
                    100.0 * result.gpu_forward_s / result.wall_time_s
                    if result.wall_time_s > 0 else 0.0
                ),
                # Memory metrics. Inference-actor RSS is the largest
                # driver of the overall memory footprint since it owns
                # the model weights + activations; worker_max tells you
                # how much each preprocess replica costs (so you can
                # size ``n_cpu_workers`` against available RAM). For
                # GPU inference, ``peak_gpu_mem_bytes`` comes from the
                # actor's own ``torch.cuda.max_memory_allocated`` — the
                # driver process's CUDA view would read ~0 since Ray
                # put the model in a separate worker process.
                "peak_cpu_rss_bytes": getattr(result, "peak_rss_bytes_driver", None),
                "peak_rss_bytes_infer": getattr(result, "peak_rss_bytes_infer", None),
                "peak_rss_bytes_worker_max": getattr(result, "peak_rss_bytes_worker_max", None),
                "peak_gpu_mem_bytes": getattr(result, "peak_gpu_mem_bytes_infer", None) or None,
                "end_to_end_wall_s": _end_to_end,
                **_tel.as_row_fields(),
                **tput,
                "timestamp_s": time.time(),
            }
            rows.append(row)
        except Exception as e:
            err_row = _error_row(
                backend_cfg, parent, child, label, device, repeat, e,
                n_stations=n_stations, batch_size=bs, overlap=0,
                n_cpu_workers=n_cpu,
                dataset_dir=cfg.dataset_dir, trial_uid=_trial_uid(key),
            )
            if is_cpu:
                err_row["infer_num_threads"] = key_infer_threads
            rows.append(err_row)
    return rows, n_skipped


def _row(*, kind, backend_cfg, parent, child, label, device, n_stations,
         batch_size, overlap, repeat, stage_times, total_s, n_windows=-1,
         dataset_dir: Optional[str] = None,
         peak_gpu_mem_bytes: Optional[int] = None,
         in_samples: Optional[int] = None,
         trial_uid: Optional[str] = None,
         # CPU / process memory metrics. All optional so older callers
         # continue to work; missing fields land as ``None`` in the row
         # and read as ``NaN`` when pandas ingests them.
         peak_cpu_rss_bytes: Optional[int] = None,
         delta_cpu_rss_bytes: Optional[int] = None,
         peak_rss_bytes_infer: Optional[int] = None,
         peak_rss_bytes_worker_max: Optional[int] = None,
         peak_gpu_mem_bytes_per_device: Optional[Dict[str, int]] = None,
         # End-to-end wall time (incl. model load / actor spawn / etc.).
         # ``wall_time_s`` stays the "critical path" number; this one is
         # what a first-time user feels when they invoke the function.
         end_to_end_wall_s: Optional[float] = None,
         # Flat dict of NVML-derived fields (mean/peak util, mean/peak
         # power, integrated energy, peak memory-used per device).
         nvml_telemetry: Optional[Dict[str, Any]] = None):
    # Prefer the backend-reported in_samples; otherwise fall back to lookup by
    # (parent, child) so derived throughput works even for the baseline path.
    if in_samples is None:
        in_samples = _model_in_samples(parent, child)
    tput = _derive_throughput(
        total_s=total_s, n_stations=n_stations,
        n_windows=n_windows, in_samples=in_samples,
    )
    return {
        "kind": kind,
        "schema_version": SCHEMA_VERSION,
        "trial_uid": trial_uid,
        "backend": backend_cfg["name"],
        "dtype": backend_cfg.get("dtype", "fp32"),
        "backend_extra": {
            k: v for k, v in backend_cfg.items()
            if k not in ("name", "dtype", "device")
        },
        "model_parent": parent, "model_child": child, "model_label": label,
        "device": device,
        "dataset_dir": dataset_dir,
        "dataset_label": Path(dataset_dir).name if dataset_dir else None,
        "n_stations": n_stations,
        "n_windows": n_windows,
        "in_samples": in_samples,
        "batch_size": batch_size,
        "overlap_samples": overlap,
        "repeat": repeat,
        "stage_times_s": stage_times,
        "total_s": total_s,
        "wall_time_s": total_s,  # alias so every kind exposes the same name
        "peak_gpu_mem_bytes": peak_gpu_mem_bytes,
        # CPU/process memory. ``peak_cpu_rss_bytes`` is the driver
        # process's RSS high-water during the trial. For Ray-actor
        # runs the driver mostly orchestrates — the infer/worker
        # actors hold the big allocations — so the ``_infer`` /
        # ``_worker_max`` fields are the ones to watch.
        "peak_cpu_rss_bytes": peak_cpu_rss_bytes,
        "delta_cpu_rss_bytes": delta_cpu_rss_bytes,
        "peak_rss_bytes_infer": peak_rss_bytes_infer,
        "peak_rss_bytes_worker_max": peak_rss_bytes_worker_max,
        "peak_gpu_mem_bytes_per_device": peak_gpu_mem_bytes_per_device,
        "end_to_end_wall_s": end_to_end_wall_s if end_to_end_wall_s is not None else total_s,
        # NVML fields flattened at the row's top level so pandas picks
        # them up as first-class columns. ``nvml_telemetry`` stays as a
        # nested dict for convenience; the individual scalars are what
        # analysis queries against.
        **(nvml_telemetry or {}),
        **tput,
        "timestamp_s": time.time(),
    }


def _error_row(backend_cfg, parent, child, label, device, repeat, err: Exception,
               *, n_stations: int = -1, batch_size: int = -1,
               overlap: int = -1, n_cpu_workers: int = -1,
               dataset_dir: Optional[str] = None,
               trial_uid: Optional[str] = None):
    return {
        "kind": "error",
        "schema_version": SCHEMA_VERSION,
        "trial_uid": trial_uid,
        "backend": backend_cfg["name"],
        "dtype": backend_cfg.get("dtype", "fp32"),
        "backend_extra": {
            k: v for k, v in backend_cfg.items()
            if k not in ("name", "dtype", "device")
        },
        "model_parent": parent, "model_child": child, "model_label": label,
        "device": device,
        "dataset_dir": dataset_dir,
        "dataset_label": Path(dataset_dir).name if dataset_dir else None,
        "n_stations": n_stations,
        "batch_size": batch_size,
        "overlap_samples": overlap,
        "n_cpu_workers": n_cpu_workers,
        "repeat": repeat,
        "error": f"{type(err).__name__}: {err}",
        "traceback": traceback.format_exc(),
        "timestamp_s": time.time(),
    }
