"""SeisBench dataset matrix: wall time + pick quality vs catalog across dtypes.

Writes a dedicated JSONL (separate from :mod:`rapid.matrix` miniSEED runs) with
``data_source="seisbench"`` (schema v6). Each row is one trial keyed by dataset
trace, model, backend, device, ``n_stations`` (duplicate pseudo-stations per
catalog trace), lean ``batch_size``, ``overlap_samples``, and repeat.

The single-GPU path duplicates the same reference trace ``N`` times (distinct
station ids), runs baseline ``annotate`` on the merged or first-duplicate stream,
and lean ``infer_batch`` with a sub-batch size sweep. Raw waveforms are cropped to
``cfg.n_samples`` samples centered on the catalog P pick (``n_samples//2`` samples
before P and the remainder after, clipped to the resampled trace); both baseline
and lean then use the same segment, with lean further cropped to ``in_samples``
for the model via ``cut_window``. ``pick_quality`` always
uses predictions for the **first duplicate** (``pick_quality_station_index=0``);
non-FP32 lean dtypes compare argmax drift vs the FP32 lean prediction from the same
``(overlap, batch_size, n_stations)`` cell when ``lean_pytorch`` **fp32** is included
in ``cfg.backends``; otherwise drift vs FP32 lean is not computed (FP32 quality
reference is still ``baseline_annotate`` on the merged or first-duplicate stream).

When ``dual_gpu`` is enabled and two CUDA devices exist, the **same**
``n_stations`` duplicate-stream list as the single-GPU path is split into two
contiguous halves (``cuda:0`` / ``cuda:1``) via
:func:`rapid.runners.dual_gpu_threaded.run_lean_two_gpu_even_halves` /
:func:`~rapid.runners.dual_gpu_threaded.run_baseline_two_gpu_even_halves` —
one driver thread per GPU, no Ray. Lean dual rows sweep ``batch_sizes`` and
``overlap_sweep`` like single-GPU; ``pick_quality`` uses the merged predictions
for station index 0. Each lean dual cell is **re-run once per CPU budget** in
``cfg.cpus``: the driver process is pinned to that many cores via
:func:`os.sched_setaffinity` for the duration of the trial (then restored), so
the threaded dispatch + CPU preprocessing share a fixed core budget. Pinned
trials emit ``runner="lean_pytorch_dual_pipelined"`` with
``n_cpu_workers_per_gpu`` = the CPU count and ``cpu_affinity_set`` = the actual
mask. When ``dual_gpu_serial=True`` an additional **unpinned** trial runs as
``runner="lean_pytorch_dual_serial"`` (``n_cpu_workers_per_gpu=-1``) for a
no-constraint reference. EQTransformer lean **fp16** (no ``torch.compile``) is
driver-skipped. Optional ``model_labels`` on a backend entry restricts that
backend to matching ``models[].label`` values (e.g. FP16+compile for PhaseNet
only). CPU trials can trim dtypes via ``cpu_backends_override`` (omit or leave
empty for a full sweep including FP16 on CPU).

Rows include **process RAM**, **PyTorch VRAM**, and when NVML is available the
same ``nvml_*`` fields as :mod:`rapid.matrix`.

Use :func:`run_seisbench_matrix` from ``scripts/run_seisbench_matrix.py``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import obspy
import seisbench.models as sbm

from rapid.backends.baseline import BaselineAnnotate
from rapid.backends.base import BackendError
from rapid.backends.lean_pytorch import EQT_LEAN_FP16_MESSAGE, LeanPyTorchBackend
from rapid.quality import TraceStats, as_dict, compare_probabilities, extract_picks_simple
from rapid.seisbench_precision_eval import (
    catalog_mask,
    catalog_pick_columns,
    cut_window,
    load_dataset,
    phase_indices,
    preprocess_array,
    waves_to_stream,
)
from rapid.timing import Timer
from rapid.memory import RSSPoller, gpu_mem_peak_all, gpu_mem_reset_all
from rapid.telemetry import GPUWatcher
from rapid.runners.dual_gpu_threaded import (
    run_baseline_two_gpu_even_halves,
    run_lean_two_gpu_even_halves,
)
from rapid.runners.dual_gpu_process import (
    DualGPUProcessResult,
    run_dual_gpu_process,
)
from rapid.runners.single_gpu import run_baseline_single, run_lean_single
from rapid.data import (
    WindowSpec,
    build_megabatch,
    preprocess_for_model,
    stream_to_3c_array,
)

LOG = logging.getLogger("rapid.seisbench_matrix")

SCHEMA_VERSION = 6


@dataclass
class SeisBenchMatrixConfig:
    seisbench_datasets: List[str]
    traces_per_dataset: int
    models: List[Dict[str, str]]
    backends: List[Dict[str, Any]]
    devices: List[str]
    repeats: int = 3
    warmup_iters: int = 1
    seed: int = 42
    output_jsonl: str = "results/seisbench_matrix.jsonl"
    # Extra JSONLs whose successful rows are merged into the resume ``completed``
    # set (same keys as ``output_jsonl``). Use when CPU-only / GPU-only phases
    # append to different files but prior trials live elsewhere—e.g. finished CPU
    # rows in ``seisbench_matrix.jsonl`` while new CPU work writes to another path.
    resume_include_jsonl: List[str] = field(default_factory=list)
    resume: bool = True
    require_catalog_s: bool = False
    include_baseline: bool = True
    prob_threshold: float = 0.3
    # P-centered raw window length (samples at model sampling rate). Baseline
    # ``annotate()`` sees this full window; lean preprocesses it then crops to
    # ``in_samples`` for inference. Default 6000 → ~3000 samples on each side of P
    # when the pick is not clipped by trace edges.
    n_samples: int = 6000
    # Mirror :class:`rapid.matrix.MatrixConfig`: trim CPU dtype sweep and run
    # the same dual-GPU shapes as the miniSEED matrix. The SeisBench dual-GPU
    # block uses thread-based runners (no Ray) since each shard typically
    # holds 1-2 stations and Ray actor setup dominates the work.
    cpu_backends_override: List[Dict[str, Any]] = field(default_factory=list)
    # Run the normal single-device sweep over ``devices`` (baseline + lean).
    # Must stay **True** for CPU-only or single-GPU cuda runs. Set ``False`` only for a
    # **dual-GPU-only** append run after 1-GPU results exist (otherwise nothing runs).
    run_single_gpu: bool = True
    dual_gpu: bool = False
    dual_gpu_serial: bool = True
    # When True, dual-GPU lean rows with ``compile=true`` are routed through
    # :func:`rapid.runners.dual_gpu_process.run_dual_gpu_process` (separate
    # Python interpreters + CUDA contexts via ``torch.multiprocessing``) so
    # ``torch.compile``'s CUDA-graph capture does not corrupt cross-thread
    # state. When False (legacy behaviour) those cells are skipped.
    dual_gpu_use_process_runner_for_compile: bool = False
    # CPU-affinity sweep: each value N pins the **whole driver process** to N
    # cores via ``os.sched_setaffinity`` while the trial runs (then restores
    # the original affinity). Recorded on dual-GPU lean rows as
    # ``n_cpu_workers_per_gpu`` for schema parity with miniSEED matrix outputs.
    cpus: List[int] = field(default_factory=lambda: [6, 8, 12])
    # Megabatch axis (same spirit as :class:`rapid.matrix.MatrixConfig`): duplicate
    # the same catalog trace N times as N ``stations``, then sweep sub-batch sizes.
    n_stations_list: List[int] = field(default_factory=lambda: [1])
    batch_sizes: List[int] = field(
        default_factory=lambda: [1, 32, 64, 128, 256, 512],
    )
    overlap_sweep: List[int] = field(default_factory=lambda: [0])
    cpu_batch_sizes_override: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SeisBenchMatrixConfig":
        clean = {k: v for k, v in d.items() if not str(k).startswith("_comment")}
        if "dual_gpu_cpu_workers" in clean and "cpus" not in clean:
            clean["cpus"] = clean.pop("dual_gpu_cpu_workers")
        else:
            clean.pop("dual_gpu_cpu_workers", None)
        clean.pop("dual_gpu_batch_size", None)
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in clean.items() if k in allowed})


CellKey = Tuple[Any, ...]

# Keys on a backend dict that are matrix driver metadata only (not passed to LeanPyTorch or stored in backend_extra).
_BACKEND_CFG_META_KEYS = frozenset({"name", "dtype", "device", "model_labels"})


def _backend_cfg_extra(backend_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: v
        for k, v in (backend_cfg or {}).items()
        if k not in _BACKEND_CFG_META_KEYS
    }


def _backend_applies_to_label(backend_cfg: Dict[str, Any], label: str) -> bool:
    allowed = backend_cfg.get("model_labels")
    if allowed is None:
        return True
    if isinstance(allowed, (list, tuple)):
        return label in allowed
    return str(label) == str(allowed)


def _backend_extra_sig(backend_cfg: Dict[str, Any]) -> str:
    return json.dumps(
        _backend_cfg_extra(backend_cfg), sort_keys=True, default=str
    )


def _trace_slot(row: Dict[str, Any]) -> Any:
    trs = row.get("sb_trace_rows")
    if isinstance(trs, (list, tuple)) and len(trs) > 0:
        return tuple(sorted(int(x) for x in trs))
    return row.get("sb_trace_row")


@contextlib.contextmanager
def _torch_thread_cap(n_cpus: Optional[int]) -> Iterator[Optional[Tuple[int, int]]]:
    """Cap ``torch.set_num_threads`` / ``set_num_interop_threads`` for the duration.

    This pairs with :func:`_pin_cpu_affinity`: pinning constrains the OS scheduler,
    but PyTorch's intra-op OpenMP pool size is set independently and defaults to
    the number of physical cores at import time. Capping it here keeps PyTorch's
    own CPU threads aligned with the affinity budget for preprocessing work.

    This does **not** affect MKL / OpenBLAS / NumPy thread pools, which are sized
    at process start from ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` env vars. For
    full BLAS-aware affinity control, launch the matrix via
    :mod:`scripts.run_all_affinity` (which sets those env vars before Python
    imports) instead of running the matrix entry point directly.
    """
    if n_cpus is None or int(n_cpus) <= 0:
        yield None
        return
    n = int(n_cpus)
    try:
        import torch  # local import to avoid cost on non-pinned paths
    except Exception:
        yield None
        return
    try:
        prev_intra = int(torch.get_num_threads())
    except Exception:
        prev_intra = -1
    try:
        prev_inter = int(torch.get_num_interop_threads())
    except Exception:
        prev_inter = -1
    try:
        torch.set_num_threads(n)
    except Exception as exc:
        LOG.warning("torch.set_num_threads(%d) failed: %s", n, exc)
    try:
        # set_num_interop_threads can only be called before any parallel work
        # has been dispatched. Skip silently if already locked in.
        torch.set_num_interop_threads(max(1, n))
    except RuntimeError:
        pass
    except Exception as exc:
        LOG.debug("torch.set_num_interop_threads(%d) failed: %s", n, exc)
    try:
        yield (prev_intra, prev_inter)
    finally:
        if prev_intra > 0:
            try:
                torch.set_num_threads(prev_intra)
            except Exception:
                pass


@contextlib.contextmanager
def _pin_cpu_affinity(n_cpus: Optional[int]) -> Iterator[Optional[List[int]]]:
    """Pin the calling process (and all its threads, on Linux) to ``n_cpus`` cores.

    Uses :func:`os.sched_setaffinity` so the dual-GPU driver thread plus its two
    GPU-shard worker threads — and any preprocessing they trigger on the CPU —
    are all confined to a fixed CPU budget for the duration of the trial.

    The original affinity mask is captured on entry and restored on exit, so the
    pin only lasts for one ``with`` block. Yields the sorted list of CPU ids the
    trial was actually pinned to (or ``None`` when no pin was applied).

    Pin is a no-op when:
      - ``n_cpus`` is ``None`` or ``<= 0`` (caller opted out of pinning),
      - :func:`os.sched_setaffinity` is unavailable (non-Linux),
      - the requested count is ``>=`` the currently available CPUs (already
        unconstrained), or
      - querying / setting affinity raises :class:`OSError` (logged then yielded
        as ``None``).
    """
    if n_cpus is None or int(n_cpus) <= 0:
        yield None
        return
    target_count = int(n_cpus)
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        LOG.warning(
            "os.sched_setaffinity unavailable on this platform; "
            "cannot pin trial to %d CPUs",
            target_count,
        )
        yield None
        return
    pid = 0
    try:
        prev = set(os.sched_getaffinity(pid))
    except OSError as exc:
        LOG.warning("sched_getaffinity failed (%s); skipping CPU pin", exc)
        yield None
        return
    avail = sorted(prev)
    if target_count >= len(avail):
        yield avail
        return
    target = set(avail[:target_count])
    try:
        os.sched_setaffinity(pid, target)
    except OSError as exc:
        LOG.warning(
            "sched_setaffinity to %d CPUs failed (%s); running unpinned",
            target_count,
            exc,
        )
        yield None
        return
    try:
        yield sorted(target)
    finally:
        try:
            os.sched_setaffinity(pid, prev)
        except OSError as exc:
            LOG.warning(
                "sched_setaffinity restore to %s failed (%s)",
                sorted(prev),
                exc,
            )


def _is_dual_ray_row(device: Any, runner: Any) -> bool:
    """True for 2-GPU rows (formerly Ray; now driver-thread).

    Name kept for resume-key parity with older JSONL outputs.
    """
    if device == "cuda:0+cuda:1":
        return True
    r = str(runner or "")
    return r in (
        "baseline_annotate_dual",
        "lean_pytorch_dual_serial",
        "lean_pytorch_dual_pipelined",
        "lean_pytorch_dual_process",
    )


def _batch_size_key_cell(row: Dict[str, Any]) -> int:
    bs_raw = row.get("batch_size")
    if bs_raw is None or bs_raw == -1:
        return -1
    try:
        return int(bs_raw)
    except (TypeError, ValueError):
        return -1


def _row_key(row: Dict[str, Any]) -> CellKey:
    extra = row.get("backend_extra") or {}
    base = (
        row.get("runner"),
        row.get("sb_dataset"),
        _trace_slot(row),
        row.get("model_label"),
        row.get("backend"),
        row.get("dtype"),
        _backend_extra_sig(extra if isinstance(extra, dict) else {}),
        row.get("device"),
    )
    rep = row.get("repeat", -1)
    if _is_dual_ray_row(row.get("device"), row.get("runner")):
        nw = row.get("n_cpu_workers_per_gpu")
        if nw is None:
            nw = -1
        bs_k = _batch_size_key_cell(row)
        ov = int(row.get("overlap_samples") or 0)
        ns_win = int(row.get("n_samples") or -1)
        # n_stations is REQUIRED in dual keys: earlier versions omitted it,
        # which made resume incorrectly mark N>64 cells as already done after
        # only the N=64 sweep had been written.
        ns = int(row.get("n_stations") or 1)
        return base + (ns_win, int(nw), bs_k, ov, ns, rep)
    ns = int(row.get("n_stations") or 1)
    bs_k = _batch_size_key_cell(row)
    ov = int(row.get("overlap_samples") or 0)
    ns_win = int(row.get("n_samples") or -1)
    pin_raw = row.get("n_cpus_pinned")
    pin = int(pin_raw) if pin_raw is not None else -1
    return base + (ns_win, ns, bs_k, ov, pin, rep)


def _dup_streams(
    raw_stream: obspy.Stream,
    n: int,
    trace_row: int,
) -> List[Tuple[str, obspy.Stream]]:
    """N independent copies of the same trace with unique trace-level IDs.

    Each duplicate stream is rewritten so that the inner ObsPy traces have a
    unique ``network.station`` code. Without this rewrite, ``stream.merge(-1)``
    inside ``model.annotate(stream)`` collapses identical-id duplicates down to
    a single station's worth of work, which makes annotate() appear to ignore
    ``n_stations`` while the lean path honestly processes every window.
    """
    out: List[Tuple[str, obspy.Stream]] = []
    for i in range(int(n)):
        traces = []
        # Encode the duplicate index into the station code. Station codes can
        # be up to 5 alphanumeric characters; we use a zero-padded base-36-ish
        # encoding to stay safely under that limit for n up to 46655 (36**3).
        # Practically: i in [0, 580) fits in 3 chars.
        sta_code = f"S{int(i):04d}"[:5]
        for tr in raw_stream:
            tc = tr.copy()
            tc.stats.network = "RP"
            tc.stats.station = sta_code
            tc.stats.location = ""
            traces.append(tc)
        st_copy = obspy.Stream(traces=traces)
        out.append((f"sb{trace_row}#{i}", st_copy))
    return out


def _effective_batch_sizes(cfg: "SeisBenchMatrixConfig", device: str) -> List[int]:
    if device == "cpu" and cfg.cpu_batch_sizes_override:
        return [int(x) for x in cfg.cpu_batch_sizes_override]
    return [int(x) for x in cfg.batch_sizes]


def _effective_overlap_list(cfg: "SeisBenchMatrixConfig") -> List[int]:
    ov = list(cfg.overlap_sweep) if cfg.overlap_sweep else [0]
    return [int(x) for x in ov]


def _trial_uid(key: CellKey) -> str:
    return hashlib.sha1(json.dumps(key, default=str).encode("utf-8")).hexdigest()[:12]


def _load_completed(path: Path) -> set:
    done: set = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("kind") == "env":
            continue
        if row.get("kind") == "error":
            continue
        if row.get("benchmark_status") == "skipped_incompatible":
            done.add(_row_key(row))
            continue
        if row.get("wall_time_s") is None and row.get("benchmark_status") != "skipped_incompatible":
            continue
        done.add(_row_key(row))
    return done


def _finite_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return int(round(v))


def _gpu_mem_reset(device: str) -> None:
    if not device or not device.startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)
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


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if not device.startswith("cuda:"):
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        idx = int(device.split(":")[1])
        return 0 <= idx < torch.cuda.device_count()
    except Exception:
        return False


def _lean_cfgs_for_device(
    cfg: SeisBenchMatrixConfig,
    lean_sorted: List[Dict[str, Any]],
    device: str,
) -> List[Dict[str, Any]]:
    if device == "cpu" and cfg.cpu_backends_override:
        ov = [b for b in cfg.cpu_backends_override if b.get("name") == "lean_pytorch"]
        return sorted(ov, key=_lean_backend_sort_key)
    return list(lean_sorted)


def _pick_quality(
    pred: np.ndarray,
    *,
    p_idx: int,
    s_idx: int,
    p_win: int,
    s_win: Optional[int],
    prob_threshold: float,
    ref_fp32: Optional[np.ndarray],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "argmax_p": int(np.argmax(pred[0, :, p_idx])),
        "argmax_s": int(np.argmax(pred[0, :, s_idx])),
        "delta_p_vs_catalog": int(np.argmax(pred[0, :, p_idx]) - p_win),
    }
    if s_win is not None:
        out["delta_s_vs_catalog"] = int(np.argmax(pred[0, :, s_idx]) - s_win)
    p_on = extract_picks_simple(pred[0, :, p_idx], threshold=prob_threshold)
    s_on = extract_picks_simple(pred[0, :, s_idx], threshold=prob_threshold)
    out["onset_p"] = int(p_on[0]) if p_on.size else None
    out["onset_s"] = int(s_on[0]) if s_on.size else None
    if out["onset_p"] is not None:
        out["onset_delta_p_vs_catalog"] = int(out["onset_p"] - p_win)
    if s_win is not None and out["onset_s"] is not None:
        out["onset_delta_s_vs_catalog"] = int(out["onset_s"] - s_win)
    if ref_fp32 is not None and ref_fp32.shape == pred.shape:
        st: TraceStats = compare_probabilities(ref_fp32, pred)
        out["vs_fp32_prob"] = as_dict(st)
        out["delta_argmax_p_vs_fp32"] = int(
            np.argmax(pred[0, :, p_idx]) - np.argmax(ref_fp32[0, :, p_idx])
        )
        out["delta_argmax_s_vs_fp32"] = int(
            np.argmax(pred[0, :, s_idx]) - np.argmax(ref_fp32[0, :, s_idx])
        )
    return out


def _cuda_indices_for_device(device: str) -> List[int]:
    if not device or not device.startswith("cuda:"):
        return []
    try:
        i = int(device.split(":")[1])
        return [i] if i >= 0 else []
    except ValueError:
        return []


def _memory_row_fields(
    rss_stats: Any,
    tel: Any,
    *,
    peak_gpu_torch: Optional[int],
    include_torch_all_devices: bool = False,
) -> Dict[str, Any]:
    """RSS + optional NVML fields + PyTorch allocator peak (per :mod:`rapid.matrix`)."""
    out: Dict[str, Any] = {
        "peak_cpu_rss_bytes": int(rss_stats.peak_rss_bytes),
        "start_cpu_rss_bytes": int(rss_stats.start_rss_bytes),
        "delta_cpu_rss_bytes": int(rss_stats.delta_rss_bytes),
        "peak_gpu_mem_bytes": peak_gpu_torch,
    }
    if include_torch_all_devices:
        try:
            allp = gpu_mem_peak_all()
            if allp:
                out["peak_gpu_mem_torch_all_devices_bytes"] = dict(allp)
        except Exception:
            pass
    if tel is not None and hasattr(tel, "as_row_fields"):
        try:
            nv = tel.as_row_fields()
            if nv:
                out.update(nv)
        except Exception:
            pass
    return out


def _annotate_stream_to_window_pred(
    ann: Any,
    sb_model: Any,
    *,
    preprocessed_T: int,
    window_start: int,
    in_samples: int,
) -> Optional[np.ndarray]:
    """Map SeisBench ``annotate()`` probability traces to ``(1, T_win, n_label)``.

    ``annotate()`` can yield a different time length than
    ``annotate_stream_pre`` (filter / reassembly). When lengths differ we
    linearly resample sample indices from the preprocessed grid to the
    probability grid so ``p_win`` / ``s_win`` stay comparable to the lean path.
    """
    try:
        labs = list(sb_model.labels)
    except Exception:
        return None
    cols: List[np.ndarray] = []
    for lab in labs:
        suf = "_" + str(lab).upper()
        tr_found = None
        for tr in ann:
            ch = str(getattr(tr.stats, "channel", "") or "").upper()
            if ch.endswith(suf):
                tr_found = tr
                break
        if tr_found is None:
            for tr in ann:
                ch = str(getattr(tr.stats, "channel", "") or "").upper()
                if str(lab).upper() in ch:
                    tr_found = tr
                    break
        if tr_found is None:
            return None
        cols.append(np.asarray(tr_found.data, dtype=np.float32))
    lengths = {int(c.shape[0]) for c in cols}
    if len(lengths) != 1:
        return None
    t_ann = int(next(iter(lengths)))
    mat = np.stack(cols, axis=-1)
    if t_ann <= 0 or preprocessed_T <= 0 or in_samples <= 0:
        return None
    if window_start < 0 or window_start + in_samples > preprocessed_T:
        return None

    if t_ann == preprocessed_T:
        sl = mat[window_start : window_start + in_samples, :]
    else:
        scale = (t_ann - 1) / max(preprocessed_T - 1, 1)
        idx = []
        for j in range(in_samples):
            ia = window_start + j
            ia = int(np.clip(ia, 0, preprocessed_T - 1))
            ib = int(round(ia * scale))
            ib = int(np.clip(ib, 0, t_ann - 1))
            idx.append(ib)
        sl = mat[np.array(idx, dtype=np.intp), :]

    out = sl[None, ...].astype(np.float32, copy=False)
    if out.ndim != 3 or out.shape[0] != 1 or out.shape[1] != in_samples:
        return None
    return out


def _cut_raw_window(
    waves: np.ndarray,
    *,
    n_samples: int,
    p_sample: int,
) -> Tuple[np.ndarray, int, int]:
    """Window raw (C, T) around catalog P to length ``n_samples``.

    Centering uses ``p_sample - n_samples//2`` (clipped), so for ``n_samples=6000``
    and an interior pick, there are ``3000`` samples at indices ``< p`` and
    ``n_samples - 3000 - 1`` samples at indices ``> p`` (``p`` is the
    ``(n_samples//2)``-th sample in the window when unclipped).

    This is the baseline/lean *alignment* window: both runners are fed the same
    raw waveform segment before any model-specific preprocessing/filtering.
    """
    if waves.ndim != 2 or int(n_samples) <= 0:
        raise ValueError("waves must be (C, T) and n_samples > 0")
    _, t_len = waves.shape
    if not (0 <= int(p_sample) < int(t_len)):
        raise ValueError("P pick outside raw trace")
    ns = int(n_samples)
    if t_len >= ns:
        start = int(np.clip(int(p_sample) - ns // 2, 0, t_len - ns))
        win = np.asarray(waves[:, start : start + ns], dtype=waves.dtype)
        p_win = int(p_sample) - start
        return win, start, p_win
    win = np.zeros((waves.shape[0], ns), dtype=waves.dtype)
    win[:, :t_len] = waves
    p_win = int(p_sample)
    return win, 0, p_win


def _skipped_eqt_fp16_row(
    *,
    key: CellKey,
    cfg: SeisBenchMatrixConfig,
    parent: str,
    child: str,
    label: str,
    device: str,
    repeat: int,
    sb_dataset: str,
    in_samples: int,
    n_samples: int,
    trace_row: Optional[int] = None,
    sb_trace_rows: Optional[Sequence[int]] = None,
    runner: str = "lean_pytorch",
    n_stations: int = 1,
    n_cpu_workers_per_gpu: Optional[int] = None,
    batch_size: int = 1,
    overlap_samples: int = 0,
) -> Dict[str, Any]:
    trs: Optional[List[int]] = None
    tr1: Optional[int] = None
    if sb_trace_rows is not None:
        trs = [int(x) for x in sb_trace_rows]
    elif trace_row is not None:
        tr1 = int(trace_row)
    return {
        "kind": "seisbench",
        "data_source": "seisbench",
        "runner": runner,
        "schema_version": SCHEMA_VERSION,
        "trial_uid": _trial_uid(key),
        "backend": "lean_pytorch",
        "dtype": "fp16",
        "backend_extra": {},
        "model_parent": parent,
        "model_child": child,
        "model_label": label,
        "device": device,
        "sb_dataset": sb_dataset,
        "sb_trace_row": tr1,
        "sb_trace_rows": trs,
        "dataset_dir": None,
        "dataset_label": sb_dataset,
        "n_stations": n_stations,
        "n_cpu_workers_per_gpu": n_cpu_workers_per_gpu,
        "n_windows": 1,
        "n_samples": int(n_samples),
        "in_samples": in_samples,
        "batch_size": int(batch_size),
        "overlap_samples": int(overlap_samples),
        "repeat": repeat,
        "benchmark_status": "skipped_incompatible",
        "skip_reason": "eqt_lean_fp16",
        "error": f"BackendError: {EQT_LEAN_FP16_MESSAGE}",
        "traceback": "Driver-side skip (same as main matrix).",
        "wall_time_s": None,
        "stage_times_s": {},
        "pick_quality": None,
        "timestamp_s": time.time(),
    }


_DUAL_COMPILE_SKIP_MESSAGE = (
    "torch.compile/torch.inductor CUDA graphs are not reliable when two GPU "
    "shards run concurrently in worker threads; skip dual-GPU compile trials "
    "(run compile variants on single-GPU cuda:0 only)."
)


def _skipped_dual_compile_row(
    *,
    key: CellKey,
    cfg: SeisBenchMatrixConfig,
    parent: str,
    child: str,
    label: str,
    device: str,
    repeat: int,
    sb_dataset: str,
    trace_row: int,
    in_samples: int,
    n_samples: int,
    runner: str,
    dtype: str,
    backend_extra: Dict[str, Any],
    n_stations: int,
    n_cpu_workers_per_gpu: Optional[int],
    batch_size: int,
    overlap_samples: int,
) -> Dict[str, Any]:
    return {
        "kind": "seisbench",
        "data_source": "seisbench",
        "runner": runner,
        "schema_version": SCHEMA_VERSION,
        "trial_uid": _trial_uid(key),
        "backend": "lean_pytorch",
        "dtype": dtype,
        "backend_extra": dict(backend_extra),
        "model_parent": parent,
        "model_child": child,
        "model_label": label,
        "device": device,
        "sb_dataset": sb_dataset,
        "sb_trace_row": trace_row,
        "dataset_dir": None,
        "dataset_label": sb_dataset,
        "n_stations": n_stations,
        "n_samples": int(n_samples),
        "n_cpu_workers_per_gpu": n_cpu_workers_per_gpu,
        "n_windows": 1,
        "in_samples": in_samples,
        "batch_size": int(batch_size),
        "overlap_samples": int(overlap_samples),
        "repeat": repeat,
        "benchmark_status": "skipped_incompatible",
        "skip_reason": "dual_gpu_torch_compile_threads",
        "error": _DUAL_COMPILE_SKIP_MESSAGE,
        "traceback": "Driver-side skip (torch.compile incompatible with threaded dual-GPU runners).",
        "wall_time_s": None,
        "stage_times_s": {},
        "pick_quality": None,
        "timestamp_s": time.time(),
    }


def _process_cpu_affinity() -> Optional[List[int]]:
    """Return the sorted CPU ids the current process is allowed to run on.

    Reflects whatever outer pinning context (e.g. ``taskset -c 0-11``) the
    matrix driver was launched under, so it captures the affinity budget that
    was actually enforced for the trial. Returns ``None`` on platforms without
    :func:`os.sched_getaffinity` (non-Linux) or if the syscall fails.
    """
    if not hasattr(os, "sched_getaffinity"):
        return None
    try:
        return sorted(int(x) for x in os.sched_getaffinity(0))
    except OSError:
        return None


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    # Stamp the process-wide CPU affinity onto every row so the JSONL is
    # self-describing about the core budget that produced it. Per-trial pin
    # info (``n_cpus_pinned`` / ``cpu_affinity_set``) is set by the dual-GPU
    # pipelined runner and is left intact; ``process_*`` always reflects the
    # outer affinity mask in effect when the row is appended.
    if row.get("kind") != "env":
        aff = _process_cpu_affinity()
        if aff is not None:
            row.setdefault("process_n_cpus", len(aff))
            row.setdefault("process_cpu_affinity", aff)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def _lean_backend_sort_key(cfg: Dict[str, Any]) -> Tuple[int, int, str]:
    dt = cfg.get("dtype", "fp32")
    comp = 1 if cfg.get("compile") else 0
    tier = {"fp32": 0, "fp16": 1, "bf16": 2}.get(dt, 9)
    return (tier, comp, dt)


def run_seisbench_matrix(cfg: SeisBenchMatrixConfig) -> None:
    out_path = Path(cfg.output_jsonl)
    completed = _load_completed(out_path) if cfg.resume else set()
    for inc_path in cfg.resume_include_jsonl:
        completed |= _load_completed(Path(inc_path))
    rng = np.random.default_rng(cfg.seed)
    n_win_cfg = int(cfg.n_samples)
    if n_win_cfg < 1:
        raise ValueError("n_samples must be >= 1")

    lean_cfgs_all = [b for b in cfg.backends if b.get("name") == "lean_pytorch"]
    lean_cfgs_all = sorted(lean_cfgs_all, key=_lean_backend_sort_key)

    if not cfg.run_single_gpu and not cfg.dual_gpu:
        LOG.warning(
            "run_single_gpu=False and dual_gpu=False: no SeisBench trials will run "
            "(single-device sweep is off and the dual-GPU block is off). "
            "Set run_single_gpu=True for CPU/cuda device sweeps, or dual_gpu=True "
            "for the 2-GPU block."
        )

    for dname in cfg.seisbench_datasets:
        dkey = dname.strip().lower()
        try:
            ds = load_dataset(dkey)
        except Exception as exc:
            LOG.warning("Skip dataset %s: %s", dkey, exc)
            continue

        mask = catalog_mask(ds, cfg.require_catalog_s)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            LOG.warning("No valid traces in %s", dkey)
            continue
        take = min(cfg.traces_per_dataset, idxs.size)
        chosen = rng.choice(idxs, size=take, replace=False)
        p_col, s_col = catalog_pick_columns(ds)

        for model_entry in cfg.models:
            parent = model_entry["parent"]
            child = model_entry["child"]
            label = model_entry.get("label", f"{parent}/{child}")

            try:
                sb_model = getattr(sbm, parent).from_pretrained(child)
            except Exception as exc:
                LOG.warning("Skip model %s: %s", label, exc)
                continue

            try:
                p_i, s_i = phase_indices(sb_model)
            except ValueError:
                LOG.warning("Model %s has no P/S labels; skip", label)
                continue

            sr = float(sb_model.sampling_rate)
            in_samples = int(sb_model.in_samples)

            trace_items: List[Dict[str, Any]] = []
            for trace_row in chosen:
                trace_row = int(trace_row)
                try:
                    waves, meta = ds.get_sample(trace_row, sampling_rate=sr)
                except Exception:
                    continue
                co = str(meta.get("trace_component_order") or "ZNE")
                if waves.ndim != 2:
                    continue
                p_cat = _finite_int(meta.get(p_col))
                if p_cat is None or not (0 <= p_cat < waves.shape[1]):
                    continue
                s_cat = _finite_int(meta.get(s_col)) if s_col else None
                if cfg.require_catalog_s:
                    if s_cat is None or not (0 <= s_cat < waves.shape[1]):
                        continue
                else:
                    if s_cat is not None and not (0 <= s_cat < waves.shape[1]):
                        s_cat = None

                # Align baseline annotate() and lean infer_batch(): same P-centered
                # raw window of ``n_samples``, then lean crops preprocessed data to
                # ``in_samples`` for the model.
                try:
                    waves_win, raw_start, p_idx_in_win = _cut_raw_window(
                        waves, n_samples=n_win_cfg, p_sample=p_cat
                    )
                except ValueError:
                    continue
                t_raw = int(waves_win.shape[1])
                arr_full = preprocess_array(sb_model, waves_win, sr, co)
                if arr_full is None:
                    continue
                t_pp = int(arr_full.shape[1])
                if t_pp != t_raw:
                    LOG.debug(
                        "Skip trace_row=%s: preprocess length %s != raw window %s",
                        trace_row,
                        t_pp,
                        t_raw,
                    )
                    continue
                try:
                    win, model_start, p_mod = cut_window(
                        arr_full, in_samples, int(p_idx_in_win)
                    )
                except ValueError:
                    continue
                s_mod: Optional[int] = None
                if s_cat is not None:
                    s_idx_arr = int(s_cat) - int(raw_start)
                    if 0 <= s_idx_arr < t_pp:
                        s_off = s_idx_arr - int(model_start)
                        if 0 <= s_off < int(in_samples):
                            s_mod = int(s_off)

                raw_stream = waves_to_stream(waves_win, sr, co)
                batch = win[None, ...].astype(np.float32, copy=False)
                trace_items.append({
                    "trace_row": trace_row,
                    "waves": waves_win,
                    "meta": meta,
                    "raw_stream": raw_stream,
                    "co": co,
                    "p_cat": p_cat,
                    "p_win": int(p_mod),
                    "s_win": s_mod,
                    "batch": batch,
                    "preprocessed_T": t_pp,
                    "win_start": int(model_start),
                    "raw_win_start": int(raw_start),
                    "n_samples": n_win_cfg,
                })

            if cfg.run_single_gpu:
                # Affinity sweep for single-GPU baseline + lean cells. Each
                # listed value pins the driver process to that many cores via
                # ``os.sched_setaffinity`` and caps PyTorch's intra-op pool via
                # ``torch.set_num_threads`` for the duration of one trial. An
                # empty ``cpus`` list keeps the legacy unpinned behaviour.
                #
                # NOTE: MKL / OpenBLAS / NumPy thread pools are sized from
                # ``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` at process startup
                # and cannot be resized per-trial from Python. For BLAS-aware
                # affinity control launch the matrix via
                # ``scripts/run_all_affinity.py``, which wraps each affinity
                # value in its own ``taskset`` subprocess with the appropriate
                # env vars set before Python imports.
                single_gpu_cpus_sweep: List[Optional[int]] = (
                    [int(x) for x in cfg.cpus] if cfg.cpus else [None]
                )

                for device in cfg.devices:
                    if "+" in device:
                        continue
                    if not _device_available(device):
                        LOG.debug("Skip device %s (unavailable)", device)
                        continue
    
                    lean_cfgs_dev = _lean_cfgs_for_device(cfg, lean_cfgs_all, device)
                    lean_cache: Dict[Tuple[str, bool], LeanPyTorchBackend] = {}
                    baseline: Optional[BaselineAnnotate] = None
    
                    def _close_dev_backends() -> None:
                        nonlocal baseline
                        for b in lean_cache.values():
                            try:
                                b.close()
                            except Exception:
                                pass
                        lean_cache.clear()
                        if baseline is not None:
                            try:
                                baseline.close()
                            except Exception:
                                pass
                            baseline = None
    
                    try:
                        ovs = _effective_overlap_list(cfg)
                        bss = _effective_batch_sizes(cfg, device)
                        for n_st in cfg.n_stations_list:
                            n_st = int(n_st)
                            for item in trace_items:
                                trace_row = int(item["trace_row"])
                                waves = item["waves"]
                                co = item["co"]
                                raw_stream = item["raw_stream"]
                                p_cat = item["p_cat"]
                                p_win = item["p_win"]
                                s_win = item["s_win"]
                                pre_T = int(item["preprocessed_T"])
                                w0 = int(item["win_start"])
                                ns_row = int(item["n_samples"])
                                streams_n = _dup_streams(raw_stream, n_st, trace_row)
    
                                for repeat in range(cfg.repeats):
                                    if cfg.include_baseline:
                                      for sg_n_cpus in single_gpu_cpus_sweep:
                                        key_b = _row_key({
                                            "runner": "baseline_annotate",
                                            "sb_dataset": dkey,
                                            "sb_trace_row": trace_row,
                                            "model_label": label,
                                            "backend": "baseline_annotate",
                                            "dtype": "fp32",
                                            "backend_extra": {},
                                            "device": device,
                                            "n_stations": n_st,
                                            "n_samples": ns_row,
                                            "batch_size": -1,
                                            "overlap_samples": 0,
                                            "n_cpus_pinned": sg_n_cpus if sg_n_cpus and sg_n_cpus > 0 else None,
                                            "repeat": repeat,
                                        })
                                        if key_b not in completed:
                                            try:
                                                if baseline is None or getattr(
                                                    baseline, "device", None
                                                ) != device:
                                                    if baseline is not None:
                                                        baseline.close()
                                                    baseline = BaselineAnnotate(
                                                        parent, child, device=device, dtype="fp32"
                                                    )
                                                    baseline.load()
                                                _gpu_mem_reset(device)
                                                rss_b = RSSPoller()
                                                rss_b.start()
                                                gpuw_b: Optional[GPUWatcher] = None
                                                tel_stats_b: Any = None
                                                idxs = _cuda_indices_for_device(device)
                                                if idxs:
                                                    gpuw_b = GPUWatcher(
                                                        device_indices=idxs,
                                                        interval_s=0.2,
                                                    )
                                                    gpuw_b.start()
                                                t = Timer(
                                                    device=device
                                                    if device.startswith("cuda")
                                                    else None
                                                )
                                                bl_pinned_ids: Optional[List[int]] = None
                                                try:
                                                    _pin_arg_b = sg_n_cpus if (sg_n_cpus and sg_n_cpus > 0) else None
                                                    with _pin_cpu_affinity(_pin_arg_b) as bl_pinned_ids, _torch_thread_cap(_pin_arg_b):
                                                        with t.stage("annotate_end_to_end"):
                                                            res_bl = run_baseline_single(
                                                                baseline,
                                                                streams_n,
                                                                merge_into_one_stream=True,
                                                            )
                                                finally:
                                                    rss_stats_b = rss_b.stop()
                                                    if gpuw_b is not None:
                                                        tel_stats_b = gpuw_b.stop()
                                                stages = dict(res_bl.stage_times)
                                                wall = float(res_bl.total_s)
                                                peak = _gpu_mem_peak(device)
                                                if n_st == 1:
                                                    ann_pb = res_bl.annotations_stream
                                                else:
                                                    ann_pb = baseline.annotate_stream(
                                                        streams_n[0][1]
                                                    )
                                                pred_b = _annotate_stream_to_window_pred(
                                                    ann_pb,
                                                    sb_model,
                                                    preprocessed_T=pre_T,
                                                    window_start=w0,
                                                    in_samples=in_samples,
                                                )
                                                pq_b: Optional[Dict[str, Any]] = None
                                                if pred_b is not None:
                                                    try:
                                                        pq_b = _pick_quality(
                                                            pred_b,
                                                            p_idx=p_i,
                                                            s_idx=s_i,
                                                            p_win=p_win,
                                                            s_win=s_win,
                                                            prob_threshold=cfg.prob_threshold,
                                                            ref_fp32=None,
                                                        )
                                                    except Exception:
                                                        pq_b = None
                                                mem_b = _memory_row_fields(
                                                    rss_stats_b,
                                                    tel_stats_b,
                                                    peak_gpu_torch=peak,
                                                    include_torch_all_devices=bool(idxs),
                                                )
                                                row_b = {
                                                    "kind": "seisbench",
                                                    "data_source": "seisbench",
                                                    "runner": "baseline_annotate",
                                                    "schema_version": SCHEMA_VERSION,
                                                    "trial_uid": _trial_uid(key_b),
                                                    "backend": "baseline_annotate",
                                                    "dtype": "fp32",
                                                    "backend_extra": {},
                                                    "model_parent": parent,
                                                    "model_child": child,
                                                    "model_label": label,
                                                    "device": device,
                                                    "sb_dataset": dkey,
                                                    "sb_trace_row": trace_row,
                                                    "catalog_p_column": p_col,
                                                    "catalog_s_column": s_col,
                                                    "p_catalog_in_window": p_win,
                                                    "s_catalog_in_window": s_win,
                                                    "dataset_dir": None,
                                                    "dataset_label": dkey,
                                                    "n_stations": n_st,
                                                    "n_windows": int(res_bl.n_windows),
                                                    "n_samples": ns_row,
                                                    "in_samples": in_samples,
                                                    "batch_size": -1,
                                                    "overlap_samples": 0,
                                                    "repeat": repeat,
                                                    "wall_time_s": wall,
                                                    "stage_times_s": stages,
                                                    "forward_only_s": res_bl.forward_only_s,
                                                    "forward_calls": res_bl.forward_calls,
                                                    "n_cpus_pinned": (
                                                        int(sg_n_cpus)
                                                        if sg_n_cpus and sg_n_cpus > 0
                                                        else None
                                                    ),
                                                    "cpu_affinity_set": (
                                                        list(bl_pinned_ids)
                                                        if bl_pinned_ids is not None
                                                        else None
                                                    ),
                                                    "pick_quality": pq_b,
                                                    "station_synthesis": "repeated_catalog_trace",
                                                    "pick_quality_trace": (
                                                        "merged_stream"
                                                        if n_st == 1
                                                        else "first_duplicate_only"
                                                    ),
                                                    "pick_quality_station_index": 0,
                                                    "timestamp_s": time.time(),
                                                    **mem_b,
                                                }
                                                _append_jsonl(out_path, row_b)
                                                completed.add(key_b)
                                            except Exception:
                                                err = {
                                                    "kind": "error",
                                                    "data_source": "seisbench",
                                                    "runner": "baseline_annotate",
                                                    "schema_version": SCHEMA_VERSION,
                                                    "trial_uid": _trial_uid(key_b),
                                                    "model_label": label,
                                                    "sb_dataset": dkey,
                                                    "sb_trace_row": trace_row,
                                                    "device": device,
                                                    "repeat": repeat,
                                                    "error": traceback.format_exc(),
                                                    "timestamp_s": time.time(),
                                                }
                                                _append_jsonl(out_path, err)
    
                                    for overlap in ovs:
                                        for bs in bss:
                                            fp32_ref = None
                                            for backend_cfg in lean_cfgs_dev:
                                                if not _backend_applies_to_label(
                                                    backend_cfg, label
                                                ):
                                                    continue
                                                dtype = backend_cfg.get(
                                                    "dtype", "fp32"
                                                )
                                                compile_flag = bool(
                                                    backend_cfg.get("compile")
                                                )
                                                extra = _backend_cfg_extra(
                                                    backend_cfg
                                                )
                                                if parent == "EQTransformer" and dtype == "fp16":
                                                    key_s = _row_key({
                                                        "runner": "lean_pytorch",
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "model_label": label,
                                                        "backend": "lean_pytorch",
                                                        "dtype": "fp16",
                                                        "backend_extra": extra,
                                                        "device": device,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "repeat": repeat,
                                                    })
                                                    if key_s not in completed:
                                                        _append_jsonl(
                                                            out_path,
                                                            _skipped_eqt_fp16_row(
                                                                key=key_s,
                                                                cfg=cfg,
                                                                parent=parent,
                                                                child=child,
                                                                label=label,
                                                                device=device,
                                                                repeat=repeat,
                                                                sb_dataset=dkey,
                                                                trace_row=trace_row,
                                                                in_samples=in_samples,
                                                                n_samples=ns_row,
                                                                n_stations=n_st,
                                                                batch_size=int(bs),
                                                                overlap_samples=int(overlap),
                                                            ),
                                                        )
                                                        completed.add(key_s)
                                                    continue
    
                                                for sg_n_cpus_l in single_gpu_cpus_sweep:
                                                    key_l = _row_key({
                                                        "runner": "lean_pytorch",
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "model_label": label,
                                                        "backend": "lean_pytorch",
                                                        "dtype": dtype,
                                                        "backend_extra": extra,
                                                        "device": device,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "n_cpus_pinned": sg_n_cpus_l if sg_n_cpus_l and sg_n_cpus_l > 0 else None,
                                                        "repeat": repeat,
                                                    })
                                                    if key_l in completed:
                                                        continue

                                                    try:
                                                        bc_k = (dtype, compile_flag)
                                                        if bc_k not in lean_cache:
                                                            b = LeanPyTorchBackend(
                                                                parent,
                                                                child,
                                                                device=device,
                                                                dtype=dtype,
                                                                compile=compile_flag,
                                                            )
                                                            b.load()
                                                            lean_cache[bc_k] = b
                                                        be = lean_cache[bc_k]

                                                        _gpu_mem_reset(device)
                                                        rss_l = RSSPoller()
                                                        rss_l.start()
                                                        gpuw_l: Optional[GPUWatcher] = None
                                                        tel_stats_l: Any = None
                                                        idxs_l = _cuda_indices_for_device(device)
                                                        if idxs_l:
                                                            gpuw_l = GPUWatcher(
                                                                device_indices=idxs_l,
                                                                interval_s=0.2,
                                                            )
                                                            gpuw_l.start()
                                                        ln_pinned_ids: Optional[List[int]] = None
                                                        try:
                                                            _pin_arg_l = sg_n_cpus_l if (sg_n_cpus_l and sg_n_cpus_l > 0) else None
                                                            with _pin_cpu_affinity(_pin_arg_l) as ln_pinned_ids, _torch_thread_cap(_pin_arg_l):
                                                                res_ln = run_lean_single(
                                                                    be,
                                                                    streams_n,
                                                                    batch_size=max(1, int(bs)),
                                                                    overlap_samples=int(overlap),
                                                                    warmup_iters=cfg.warmup_iters,
                                                                )
                                                        finally:
                                                            rss_stats_l = rss_l.stop()
                                                            if gpuw_l is not None:
                                                                tel_stats_l = gpuw_l.stop()
                                                        stages = dict(res_ln.stage_times)
                                                        wall = float(res_ln.total_s)
                                                        peak = _gpu_mem_peak(device)
                                                        pred = res_ln.predictions
                                                        pq: Optional[Dict[str, Any]] = None
                                                        if (
                                                            pred is not None
                                                            and pred.shape[0] > 0
                                                        ):
                                                            pred0 = pred[0:1]
                                                            ref_for_pick = None
                                                            if dtype == "fp32":
                                                                fp32_ref = pred0
                                                            else:
                                                                ref_for_pick = fp32_ref
                                                            pq = _pick_quality(
                                                                pred0,
                                                                p_idx=p_i,
                                                                s_idx=s_i,
                                                                p_win=p_win,
                                                                s_win=s_win,
                                                                prob_threshold=cfg.prob_threshold,
                                                                ref_fp32=ref_for_pick,
                                                            )
    
                                                        mem_l = _memory_row_fields(
                                                            rss_stats_l,
                                                            tel_stats_l,
                                                            peak_gpu_torch=peak,
                                                            include_torch_all_devices=bool(idxs_l),
                                                        )
                                                        row_l = {
                                                            "kind": "seisbench",
                                                            "data_source": "seisbench",
                                                            "runner": "lean_pytorch",
                                                            "schema_version": SCHEMA_VERSION,
                                                            "trial_uid": _trial_uid(key_l),
                                                            "backend": "lean_pytorch",
                                                            "dtype": dtype,
                                                            "backend_extra": extra,
                                                            "model_parent": parent,
                                                            "model_child": child,
                                                            "model_label": label,
                                                            "device": device,
                                                            "sb_dataset": dkey,
                                                            "sb_trace_row": trace_row,
                                                            "catalog_p_column": p_col,
                                                            "catalog_s_column": s_col,
                                                            "p_catalog_in_window": p_win,
                                                            "s_catalog_in_window": s_win,
                                                            "dataset_dir": None,
                                                            "dataset_label": dkey,
                                                            "n_stations": n_st,
                                                            "n_windows": int(res_ln.n_windows),
                                                            "n_samples": ns_row,
                                                            "in_samples": in_samples,
                                                            "batch_size": int(bs),
                                                            "overlap_samples": int(overlap),
                                                            "repeat": repeat,
                                                            "wall_time_s": wall,
                                                            "stage_times_s": stages,
                                                            "forward_only_s": res_ln.forward_only_s,
                                                            "forward_calls": res_ln.forward_calls,
                                                            "n_cpus_pinned": (
                                                                int(sg_n_cpus_l)
                                                                if sg_n_cpus_l and sg_n_cpus_l > 0
                                                                else None
                                                            ),
                                                            "cpu_affinity_set": (
                                                                list(ln_pinned_ids)
                                                                if ln_pinned_ids is not None
                                                                else None
                                                            ),
                                                            "pick_quality": pq,
                                                            "station_synthesis": "repeated_catalog_trace",
                                                            "pick_quality_station_index": 0,
                                                            "timestamp_s": time.time(),
                                                            **mem_l,
                                                        }
                                                        _append_jsonl(out_path, row_l)
                                                        completed.add(key_l)
                                                    except BackendError as exc:
                                                        row_e = {
                                                            "kind": "error",
                                                            "data_source": "seisbench",
                                                            "runner": "lean_pytorch",
                                                            "schema_version": SCHEMA_VERSION,
                                                            "trial_uid": _trial_uid(key_l),
                                                            "model_label": label,
                                                            "sb_dataset": dkey,
                                                            "sb_trace_row": trace_row,
                                                            "backend": "lean_pytorch",
                                                            "dtype": dtype,
                                                            "device": device,
                                                            "repeat": repeat,
                                                            "error": str(exc),
                                                            "timestamp_s": time.time(),
                                                        }
                                                        _append_jsonl(out_path, row_e)
                                                    except Exception:
                                                        row_e = {
                                                            "kind": "error",
                                                            "data_source": "seisbench",
                                                            "runner": "lean_pytorch",
                                                            "schema_version": SCHEMA_VERSION,
                                                            "trial_uid": _trial_uid(key_l),
                                                            "model_label": label,
                                                            "sb_dataset": dkey,
                                                            "sb_trace_row": trace_row,
                                                            "error": traceback.format_exc(),
                                                            "timestamp_s": time.time(),
                                                        }
                                                        _append_jsonl(out_path, row_e)
                    finally:
                        _close_dev_backends()

            if cfg.dual_gpu:
                try:
                    import torch
                    n_gpu = int(torch.cuda.device_count())
                except Exception:
                    n_gpu = 0
                if n_gpu < 2:
                    LOG.warning(
                        "dual_gpu enabled but fewer than 2 CUDA devices (%d); "
                        "skipping SeisBench 2-GPU block for %s / %s",
                        n_gpu,
                        dkey,
                        label,
                    )
                else:
                    lean_dual_cfgs = sorted(
                        [b for b in cfg.backends if b.get("name") == "lean_pytorch"],
                        key=_lean_backend_sort_key,
                    )
                    dual_dev = "cuda:0+cuda:1"
                    cpus_sweep: List[int] = [int(x) for x in (cfg.cpus or [])]
                    ovs_d = _effective_overlap_list(cfg)
                    bss_d = [int(x) for x in cfg.batch_sizes]

                    for n_st in cfg.n_stations_list:
                        n_st = int(n_st)
                        if n_st < 2:
                            continue
                        for item in trace_items:
                            trace_row = int(item["trace_row"])
                            raw_stream = item["raw_stream"]
                            p_win = item["p_win"]
                            s_win = item["s_win"]
                            pre_T = int(item["preprocessed_T"])
                            w0 = int(item["win_start"])
                            ns_row = int(item["n_samples"])
                            streams_n = _dup_streams(raw_stream, n_st, trace_row)

                            for repeat in range(cfg.repeats):
                                if cfg.include_baseline:
                                    key_bd = _row_key({
                                        "runner": "baseline_annotate_dual",
                                        "sb_dataset": dkey,
                                        "sb_trace_row": trace_row,
                                        "model_label": label,
                                        "backend": "baseline_annotate",
                                        "dtype": "fp32",
                                        "backend_extra": {},
                                        "device": dual_dev,
                                        "n_stations": n_st,
                                        "n_samples": ns_row,
                                        "n_cpu_workers_per_gpu": 0,
                                        "batch_size": -1,
                                        "overlap_samples": 0,
                                        "repeat": repeat,
                                    })
                                    if key_bd not in completed:
                                        try:
                                            gpu_mem_reset_all()
                                            rss_d = RSSPoller()
                                            rss_d.start()
                                            gw_d = GPUWatcher(
                                                device_indices=[0, 1],
                                                interval_s=0.2,
                                            )
                                            gw_d.start()
                                            try:
                                                res_b = run_baseline_two_gpu_even_halves(
                                                    parent_model=parent,
                                                    child_model=child,
                                                    streams=streams_n,
                                                    first_station_stream=streams_n[0][1],
                                                )
                                            finally:
                                                rss_sd = rss_d.stop()
                                                tel_sd = gw_d.stop()
                                            mem_d = _memory_row_fields(
                                                rss_sd,
                                                tel_sd,
                                                peak_gpu_torch=None,
                                                include_torch_all_devices=True,
                                            )
                                            ann_b = res_b.annotations_stream_first_station
                                            pred_b = _annotate_stream_to_window_pred(
                                                ann_b,
                                                sb_model,
                                                preprocessed_T=pre_T,
                                                window_start=w0,
                                                in_samples=in_samples,
                                            )
                                            pq_bd: Optional[Dict[str, Any]] = None
                                            if pred_b is not None:
                                                try:
                                                    pq_bd = _pick_quality(
                                                        pred_b,
                                                        p_idx=p_i,
                                                        s_idx=s_i,
                                                        p_win=p_win,
                                                        s_win=s_win,
                                                        prob_threshold=cfg.prob_threshold,
                                                        ref_fp32=None,
                                                    )
                                                except Exception:
                                                    pq_bd = None
                                            row_bd = {
                                                "kind": "seisbench",
                                                "data_source": "seisbench",
                                                "runner": "baseline_annotate_dual",
                                                "schema_version": SCHEMA_VERSION,
                                                "trial_uid": _trial_uid(key_bd),
                                                "backend": "baseline_annotate",
                                                "dtype": "fp32",
                                                "backend_extra": {},
                                                "model_parent": parent,
                                                "model_child": child,
                                                "model_label": label,
                                                "device": dual_dev,
                                                "sb_dataset": dkey,
                                                "sb_trace_row": trace_row,
                                                "catalog_p_column": p_col,
                                                "catalog_s_column": s_col,
                                                "p_catalog_in_window": p_win,
                                                "s_catalog_in_window": s_win,
                                                "dataset_dir": None,
                                                "dataset_label": dkey,
                                                "n_stations": n_st,
                                                "n_cpu_workers_per_gpu": 0,
                                                "n_windows": int(res_b.sum_windows),
                                                "n_samples": ns_row,
                                                "in_samples": in_samples,
                                                "batch_size": -1,
                                                "overlap_samples": 0,
                                                "repeat": repeat,
                                                "wall_time_s": float(res_b.wall_time_s),
                                                "end_to_end_wall_s": float(
                                                    res_b.end_to_end_wall_s
                                                ),
                                                "stage_times_s": {},
                                                "pick_quality": pq_bd,
                                                "station_synthesis": "repeated_catalog_trace",
                                                "pick_quality_trace": "first_duplicate_only",
                                                "pick_quality_station_index": 0,
                                                "dual_gpu_station_split": "even_halves_cuda0_cuda1",
                                                "timestamp_s": time.time(),
                                                **mem_d,
                                            }
                                            _append_jsonl(out_path, row_bd)
                                            completed.add(key_bd)
                                        except Exception:
                                            err = {
                                                "kind": "error",
                                                "data_source": "seisbench",
                                                "runner": "baseline_annotate_dual",
                                                "schema_version": SCHEMA_VERSION,
                                                "trial_uid": _trial_uid(key_bd),
                                                "model_label": label,
                                                "sb_dataset": dkey,
                                                "sb_trace_row": trace_row,
                                                "device": dual_dev,
                                                "repeat": repeat,
                                                "error": traceback.format_exc(),
                                                "timestamp_s": time.time(),
                                            }
                                            _append_jsonl(out_path, err)

                                for overlap in ovs_d:
                                    for bs in bss_d:
                                        fp32_ref_d: Optional[Any] = None
                                        for backend_cfg in lean_dual_cfgs:
                                            if not _backend_applies_to_label(
                                                backend_cfg, label
                                            ):
                                                continue
                                            dtype = backend_cfg.get("dtype", "fp32")
                                            extra = _backend_cfg_extra(backend_cfg)
                                            bk_kw = extra or None

                                            compile_dual = bool(backend_cfg.get("compile"))
                                            if compile_dual:
                                                # Route compile + dual-GPU through the process-based runner
                                                # if the operator asked for it. Each GPU gets its own Python
                                                # interpreter + CUDA context (via torch.multiprocessing.spawn),
                                                # which is the only configuration where torch.compile's CUDA
                                                # graphs do not corrupt cross-thread state.
                                                use_proc_dc = bool(
                                                    getattr(cfg, "dual_gpu_use_process_runner_for_compile", False)
                                                )
                                                if use_proc_dc:
                                                    key_dpp = _row_key({
                                                        "runner": "lean_pytorch_dual_process",
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "model_label": label,
                                                        "backend": "lean_pytorch",
                                                        "dtype": dtype,
                                                        "backend_extra": extra,
                                                        "device": dual_dev,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "n_cpu_workers_per_gpu": -2,
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "repeat": repeat,
                                                    })
                                                    if key_dpp not in completed:
                                                        try:
                                                            _baseline_obj = locals().get("baseline", None)
                                                            sb_model_dp = (
                                                                _baseline_obj._model
                                                                if (_baseline_obj is not None and getattr(_baseline_obj, "_model", None) is not None)
                                                                else getattr(sbm, parent).from_pretrained(child)
                                                            )
                                                            argdict_dp = {
                                                                "sampling_rate": getattr(
                                                                    sb_model_dp, "sampling_rate", None
                                                                )
                                                            }
                                                            t_prep_dp0 = time.perf_counter()
                                                            arrays_dp: List[Tuple[str, np.ndarray]] = []
                                                            for sta_id, st in streams_n:
                                                                pre = preprocess_for_model(
                                                                    sb_model_dp, st, argdict=argdict_dp
                                                                )
                                                                arr = stream_to_3c_array(
                                                                    pre,
                                                                    component_order=getattr(
                                                                        sb_model_dp, "component_order", None
                                                                    ) or "ZNE",
                                                                )
                                                                if arr is not None:
                                                                    arrays_dp.append((sta_id, arr))
                                                            spec_dp = WindowSpec(
                                                                in_samples=in_samples,
                                                                overlap_samples=int(overlap),
                                                            )
                                                            mb_dp = build_megabatch(arrays_dp, spec_dp)
                                                            prep_s_dp = time.perf_counter() - t_prep_dp0

                                                            rss_dp = RSSPoller()
                                                            rss_dp.start()
                                                            gw_dp = GPUWatcher(
                                                                device_indices=[0, 1], interval_s=0.2
                                                            )
                                                            gw_dp.start()
                                                            try:
                                                                r_dp = run_dual_gpu_process(
                                                                    parent_model=parent,
                                                                    child_model=child,
                                                                    dtype=dtype,
                                                                    megabatch=mb_dp,
                                                                    batch_size=max(1, int(bs)),
                                                                    compile_model=True,
                                                                    devices=("cuda:0", "cuda:1"),
                                                                    warmup_iters=cfg.warmup_iters,
                                                                )
                                                            finally:
                                                                rss_stats_dp = rss_dp.stop()
                                                                tel_stats_dp = gw_dp.stop()
                                                            mem_dp = _memory_row_fields(
                                                                rss_stats_dp,
                                                                tel_stats_dp,
                                                                peak_gpu_torch=None,
                                                                include_torch_all_devices=True,
                                                            )
                                                            row_dp = {
                                                                "kind": "seisbench",
                                                                "data_source": "seisbench",
                                                                "runner": "lean_pytorch_dual_process",
                                                                "schema_version": SCHEMA_VERSION,
                                                                "trial_uid": _trial_uid(key_dpp),
                                                                "backend": "lean_pytorch",
                                                                "dtype": dtype,
                                                                "backend_extra": extra,
                                                                "model_parent": parent,
                                                                "model_child": child,
                                                                "model_label": label,
                                                                "device": dual_dev,
                                                                "sb_dataset": dkey,
                                                                "sb_trace_row": trace_row,
                                                                "catalog_p_column": p_col,
                                                                "catalog_s_column": s_col,
                                                                "p_catalog_in_window": p_win,
                                                                "s_catalog_in_window": s_win,
                                                                "dataset_dir": None,
                                                                "dataset_label": dkey,
                                                                "n_stations": n_st,
                                                                "n_samples": ns_row,
                                                                "n_cpu_workers_per_gpu": -2,
                                                                "n_windows": int(r_dp.n_windows),
                                                                "in_samples": in_samples,
                                                                "batch_size": int(bs),
                                                                "overlap_samples": int(overlap),
                                                                "repeat": repeat,
                                                                "wall_time_s": float(r_dp.wall_time_s),
                                                                "end_to_end_wall_s": float(r_dp.wall_time_s),
                                                                "stage_times_s": {
                                                                    "driver_preprocess": float(prep_s_dp),
                                                                    "gpu0_forward": float(r_dp.gpu0_time_s),
                                                                    "gpu1_forward": float(r_dp.gpu1_time_s),
                                                                },
                                                                "forward_only_s": float(
                                                                    max(r_dp.gpu0_time_s, r_dp.gpu1_time_s)
                                                                ),
                                                                "forward_calls": None,
                                                                "pick_quality": None,
                                                                "station_synthesis": "repeated_catalog_trace",
                                                                "pick_quality_station_index": 0,
                                                                "dual_gpu_station_split": "megabatch_halves_cuda0_cuda1",
                                                                "timestamp_s": time.time(),
                                                                **mem_dp,
                                                            }
                                                            _append_jsonl(out_path, row_dp)
                                                            completed.add(key_dpp)
                                                        except Exception as exc:
                                                            _append_jsonl(
                                                                out_path,
                                                                {
                                                                    "kind": "error",
                                                                    "data_source": "seisbench",
                                                                    "runner": "lean_pytorch_dual_process",
                                                                    "schema_version": SCHEMA_VERSION,
                                                                    "trial_uid": _trial_uid(key_dpp),
                                                                    "model_label": label,
                                                                    "sb_dataset": dkey,
                                                                    "sb_trace_row": trace_row,
                                                                    "backend": "lean_pytorch",
                                                                    "dtype": dtype,
                                                                    "device": dual_dev,
                                                                    "repeat": repeat,
                                                                    "error": f"{exc}\n{traceback.format_exc()}",
                                                                    "timestamp_s": time.time(),
                                                                },
                                                            )
                                                    continue

                                                want_serial_dc = bool(cfg.dual_gpu_serial)
                                                if want_serial_dc:
                                                    key_dcs = _row_key({
                                                        "runner": "lean_pytorch_dual_serial",
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "model_label": label,
                                                        "backend": "lean_pytorch",
                                                        "dtype": dtype,
                                                        "backend_extra": extra,
                                                        "device": dual_dev,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "n_cpu_workers_per_gpu": -1,
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "repeat": repeat,
                                                    })
                                                    if key_dcs not in completed:
                                                        _append_jsonl(
                                                            out_path,
                                                            _skipped_dual_compile_row(
                                                                key=key_dcs,
                                                                cfg=cfg,
                                                                parent=parent,
                                                                child=child,
                                                                label=label,
                                                                device=dual_dev,
                                                                repeat=repeat,
                                                                sb_dataset=dkey,
                                                                trace_row=trace_row,
                                                                in_samples=in_samples,
                                                                n_samples=ns_row,
                                                                runner="lean_pytorch_dual_serial",
                                                                dtype=dtype,
                                                                backend_extra=extra,
                                                                n_stations=n_st,
                                                                n_cpu_workers_per_gpu=-1,
                                                                batch_size=int(bs),
                                                                overlap_samples=int(overlap),
                                                            ),
                                                        )
                                                        completed.add(key_dcs)
                                                for nw_dc in cpus_sweep:
                                                    key_dcp = _row_key({
                                                        "runner": "lean_pytorch_dual_pipelined",
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "model_label": label,
                                                        "backend": "lean_pytorch",
                                                        "dtype": dtype,
                                                        "backend_extra": extra,
                                                        "device": dual_dev,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "n_cpu_workers_per_gpu": int(nw_dc),
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "repeat": repeat,
                                                    })
                                                    if key_dcp not in completed:
                                                        _append_jsonl(
                                                            out_path,
                                                            _skipped_dual_compile_row(
                                                                key=key_dcp,
                                                                cfg=cfg,
                                                                parent=parent,
                                                                child=child,
                                                                label=label,
                                                                device=dual_dev,
                                                                repeat=repeat,
                                                                sb_dataset=dkey,
                                                                trace_row=trace_row,
                                                                in_samples=in_samples,
                                                                n_samples=ns_row,
                                                                runner="lean_pytorch_dual_pipelined",
                                                                dtype=dtype,
                                                                backend_extra=extra,
                                                                n_stations=n_st,
                                                                n_cpu_workers_per_gpu=int(nw_dc),
                                                                batch_size=int(bs),
                                                                overlap_samples=int(overlap),
                                                            ),
                                                        )
                                                        completed.add(key_dcp)
                                                continue

                                            if parent == "EQTransformer" and dtype == "fp16":
                                                key_ser = _row_key({
                                                    "runner": "lean_pytorch_dual_serial",
                                                    "sb_dataset": dkey,
                                                    "sb_trace_row": trace_row,
                                                    "model_label": label,
                                                    "backend": "lean_pytorch",
                                                    "dtype": "fp16",
                                                    "backend_extra": extra,
                                                    "device": dual_dev,
                                                    "n_stations": n_st,
                                                    "n_samples": ns_row,
                                                    "n_cpu_workers_per_gpu": -1,
                                                    "batch_size": int(bs),
                                                    "overlap_samples": int(overlap),
                                                    "repeat": repeat,
                                                })
                                                if key_ser not in completed:
                                                    _append_jsonl(
                                                        out_path,
                                                        _skipped_eqt_fp16_row(
                                                            key=key_ser,
                                                            cfg=cfg,
                                                            parent=parent,
                                                            child=child,
                                                            label=label,
                                                            device=dual_dev,
                                                            repeat=repeat,
                                                            sb_dataset=dkey,
                                                            trace_row=trace_row,
                                                            in_samples=in_samples,
                                                            n_samples=ns_row,
                                                            runner="lean_pytorch_dual_serial",
                                                            n_stations=n_st,
                                                            n_cpu_workers_per_gpu=-1,
                                                            batch_size=int(bs),
                                                            overlap_samples=int(overlap),
                                                        ),
                                                    )
                                                    completed.add(key_ser)
                                                for nw_skip in cpus_sweep:
                                                    key_pl = _row_key({
                                                        "runner": "lean_pytorch_dual_pipelined",
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "model_label": label,
                                                        "backend": "lean_pytorch",
                                                        "dtype": "fp16",
                                                        "backend_extra": extra,
                                                        "device": dual_dev,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "n_cpu_workers_per_gpu": int(nw_skip),
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "repeat": repeat,
                                                    })
                                                    if key_pl not in completed:
                                                        _append_jsonl(
                                                            out_path,
                                                            _skipped_eqt_fp16_row(
                                                                key=key_pl,
                                                                cfg=cfg,
                                                                parent=parent,
                                                                child=child,
                                                                label=label,
                                                                device=dual_dev,
                                                                repeat=repeat,
                                                                sb_dataset=dkey,
                                                                trace_row=trace_row,
                                                                in_samples=in_samples,
                                                                n_samples=ns_row,
                                                                runner="lean_pytorch_dual_pipelined",
                                                                n_stations=n_st,
                                                                n_cpu_workers_per_gpu=int(nw_skip),
                                                                batch_size=int(bs),
                                                                overlap_samples=int(overlap),
                                                            ),
                                                        )
                                                        completed.add(key_pl)
                                                continue

                                            want_serial = bool(cfg.dual_gpu_serial)
                                            trials: List[Tuple[int, CellKey, str]] = []
                                            if want_serial:
                                                key_ls = _row_key({
                                                    "runner": "lean_pytorch_dual_serial",
                                                    "sb_dataset": dkey,
                                                    "sb_trace_row": trace_row,
                                                    "model_label": label,
                                                    "backend": "lean_pytorch",
                                                    "dtype": dtype,
                                                    "backend_extra": extra,
                                                    "device": dual_dev,
                                                    "n_stations": n_st,
                                                    "n_samples": ns_row,
                                                    "n_cpu_workers_per_gpu": -1,
                                                    "batch_size": int(bs),
                                                    "overlap_samples": int(overlap),
                                                    "repeat": repeat,
                                                })
                                                if key_ls not in completed:
                                                    trials.append((-1, key_ls, "lean_pytorch_dual_serial"))
                                            for nw in cpus_sweep:
                                                key_lp = _row_key({
                                                    "runner": "lean_pytorch_dual_pipelined",
                                                    "sb_dataset": dkey,
                                                    "sb_trace_row": trace_row,
                                                    "model_label": label,
                                                    "backend": "lean_pytorch",
                                                    "dtype": dtype,
                                                    "backend_extra": extra,
                                                    "device": dual_dev,
                                                    "n_stations": n_st,
                                                    "n_samples": ns_row,
                                                    "n_cpu_workers_per_gpu": int(nw),
                                                    "batch_size": int(bs),
                                                    "overlap_samples": int(overlap),
                                                    "repeat": repeat,
                                                })
                                                if key_lp not in completed:
                                                    trials.append((int(nw), key_lp, "lean_pytorch_dual_pipelined"))
                                            if not trials:
                                                continue

                                            for n_cpus, key_t, runner_name in trials:
                                                pin_arg = n_cpus if n_cpus > 0 else None
                                                try:
                                                    gpu_mem_reset_all()
                                                    rss_ld = RSSPoller()
                                                    rss_ld.start()
                                                    gw_ld = GPUWatcher(
                                                        device_indices=[0, 1],
                                                        interval_s=0.2,
                                                    )
                                                    gw_ld.start()
                                                    pinned_ids: Optional[List[int]] = None
                                                    try:
                                                        with _pin_cpu_affinity(pin_arg) as pinned_ids:
                                                            r_ds = run_lean_two_gpu_even_halves(
                                                                parent_model=parent,
                                                                child_model=child,
                                                                streams=streams_n,
                                                                batch_size=max(1, int(bs)),
                                                                overlap_samples=int(overlap),
                                                                dtype=dtype,
                                                                backend_kwargs=bk_kw,
                                                                warmup_iters=cfg.warmup_iters,
                                                            )
                                                    finally:
                                                        rss_sld = rss_ld.stop()
                                                        tel_sld = gw_ld.stop()
                                                    mem_ld = _memory_row_fields(
                                                        rss_sld,
                                                        tel_sld,
                                                        peak_gpu_torch=None,
                                                        include_torch_all_devices=True,
                                                    )
                                                    pred = r_ds.predictions
                                                    pq_d: Optional[Dict[str, Any]] = None
                                                    if pred is not None and pred.shape[0] > 0:
                                                        pred0 = pred[0:1]
                                                        ref_for_pick = None
                                                        if dtype == "fp32":
                                                            if fp32_ref_d is None:
                                                                fp32_ref_d = pred0
                                                        else:
                                                            ref_for_pick = fp32_ref_d
                                                        pq_d = _pick_quality(
                                                            pred0,
                                                            p_idx=p_i,
                                                            s_idx=s_i,
                                                            p_win=p_win,
                                                            s_win=s_win,
                                                            prob_threshold=cfg.prob_threshold,
                                                            ref_fp32=ref_for_pick,
                                                        )

                                                    row_t = {
                                                        "kind": "seisbench",
                                                        "data_source": "seisbench",
                                                        "runner": runner_name,
                                                        "schema_version": SCHEMA_VERSION,
                                                        "trial_uid": _trial_uid(key_t),
                                                        "backend": "lean_pytorch",
                                                        "dtype": dtype,
                                                        "backend_extra": extra,
                                                        "model_parent": parent,
                                                        "model_child": child,
                                                        "model_label": label,
                                                        "device": dual_dev,
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "catalog_p_column": p_col,
                                                        "catalog_s_column": s_col,
                                                        "p_catalog_in_window": p_win,
                                                        "s_catalog_in_window": s_win,
                                                        "dataset_dir": None,
                                                        "dataset_label": dkey,
                                                        "n_stations": n_st,
                                                        "n_samples": ns_row,
                                                        "n_cpu_workers_per_gpu": int(n_cpus),
                                                        "n_cpus_pinned": (
                                                            int(n_cpus) if n_cpus > 0 else None
                                                        ),
                                                        "cpu_affinity_set": (
                                                            list(pinned_ids)
                                                            if pinned_ids is not None
                                                            else None
                                                        ),
                                                        "n_windows": int(r_ds.sum_windows),
                                                        "in_samples": in_samples,
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "repeat": repeat,
                                                        "wall_time_s": float(r_ds.wall_time_s),
                                                        "end_to_end_wall_s": float(
                                                            r_ds.end_to_end_wall_s
                                                        ),
                                                        "stage_times_s": {},
                                                        "pick_quality": pq_d,
                                                        "station_synthesis": "repeated_catalog_trace",
                                                        "pick_quality_station_index": 0,
                                                        "dual_gpu_station_split": "even_halves_cuda0_cuda1",
                                                        "timestamp_s": time.time(),
                                                        **mem_ld,
                                                    }
                                                    _append_jsonl(out_path, row_t)
                                                    completed.add(key_t)
                                                except BackendError as exc:
                                                    row_e = {
                                                        "kind": "error",
                                                        "data_source": "seisbench",
                                                        "runner": runner_name,
                                                        "schema_version": SCHEMA_VERSION,
                                                        "trial_uid": _trial_uid(key_t),
                                                        "model_label": label,
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "backend": "lean_pytorch",
                                                        "dtype": dtype,
                                                        "device": dual_dev,
                                                        "n_cpu_workers_per_gpu": int(n_cpus),
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "n_stations": n_st,
                                                        "repeat": repeat,
                                                        "error": str(exc),
                                                        "timestamp_s": time.time(),
                                                    }
                                                    _append_jsonl(out_path, row_e)
                                                except Exception:
                                                    row_e = {
                                                        "kind": "error",
                                                        "data_source": "seisbench",
                                                        "runner": runner_name,
                                                        "schema_version": SCHEMA_VERSION,
                                                        "trial_uid": _trial_uid(key_t),
                                                        "model_label": label,
                                                        "sb_dataset": dkey,
                                                        "sb_trace_row": trace_row,
                                                        "n_cpu_workers_per_gpu": int(n_cpus),
                                                        "batch_size": int(bs),
                                                        "overlap_samples": int(overlap),
                                                        "n_stations": n_st,
                                                        "repeat": repeat,
                                                        "error": traceback.format_exc(),
                                                        "timestamp_s": time.time(),
                                                    }
                                                    _append_jsonl(out_path, row_e)

    if not out_path.exists():
        # Resume can cover 100% of keys (e.g. continuation file + include of full
        # prior JSONL); we still create the path so runs are not "finished" with a
        # missing artifact. Downstream loaders skip kind=env (see _load_completed).
        _append_jsonl(
            out_path,
            {
                "kind": "env",
                "data_source": "seisbench",
                "schema_version": SCHEMA_VERSION,
                "driver_note": (
                    "No trial rows appended: every cell was already in the resume "
                    "completed set, or no work was scheduled for this output path."
                ),
                "output_jsonl": str(out_path),
                "timestamp_s": time.time(),
            },
        )
        LOG.info(
            "SeisBench matrix: created %s with env placeholder only (zero new trials).",
            out_path,
        )

    LOG.info("SeisBench matrix finished → %s", out_path)
