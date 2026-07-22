"""Simple entry points for running SeisBench pickers with RAPID orchestration.

Use this module when you want to pick a station network without wiring up
``RunEQCCTPro`` by hand. Choose an orchestration strategy (Model-Actor or
Ripper) and a forward path (SeisBench classify, or Slipstream at a chosen
precision). For single-process SeisBench ``annotate`` / ``classify`` without
Ray, see ``rapid.api``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .functionality import RunEQCCTPro

PathLike = Union[str, Path]

# Friendly names used in the paper and README.
MODELS: Dict[str, Dict[str, str]] = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"parent": "EQTransformer", "child": "nonconservative"},
}

STRATEGIES = ("modelactor", "ripper")
FORWARDS = ("classify", "slipstream")
DTYPES = ("fp32", "fp16", "bf16")


def _resolve_model(model: str) -> Dict[str, str]:
    if model in MODELS:
        return MODELS[model]
    # Allow "Parent/child" passthrough for custom SeisBench weights.
    if "/" in model:
        parent, child = model.split("/", 1)
        return {"parent": parent, "child": child}
    raise ValueError(
        f"Unknown model '{model}'. Choose one of {list(MODELS)} "
        "or pass 'ParentModel/child_weights'."
    )


def load_network_meta(input_dir: PathLike) -> Dict[str, Any]:
    """Load ``manifest.json`` written by ``examples/build_seisbench_network.py``.

    Returns the ``meta`` block (start/end times, station count, …). Raises if
    the manifest is missing so callers know to pass times explicitly.
    """
    path = Path(input_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No manifest.json under {input_dir}. Pass start_time, end_time, "
            "and timechunk_dt yourself, or build a network with "
            "examples/build_seisbench_network.py."
        )
    payload = json.loads(path.read_text())
    meta = payload.get("meta") or payload
    return meta


def _default_cpu_ids(n: Optional[int]) -> List[int]:
    if n is None:
        n = max(1, (os.cpu_count() or 1))
    return list(range(int(n)))


def pick(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    model: str = "PhaseNet",
    strategy: str = "modelactor",
    forward: str = "classify",
    dtype: str = "bf16",
    compile_model: bool = False,
    n_workers: Optional[int] = None,
    cpu_ids: Optional[Sequence[int]] = None,
    gpus: Optional[Sequence[int]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    timechunk_dt: Optional[int] = None,
    overlap_samples: int = 0,
    batch_size: int = 256,
    p_threshold: float = 0.001,
    s_threshold: float = 0.02,
    detection_threshold: float = 0.3,
    log_filepath: Optional[PathLike] = None,
    tmp_dir: Optional[PathLike] = None,
    pick_output_format: str = "ascii",
    ascii_station_pick_format: str = "csv",
    overwrite: bool = True,
    **extra: Any,
) -> RunEQCCTPro:
    """Pick P/S arrivals for every station under ``input_dir``.

    Parameters
    ----------
    input_dir
        Directory of per-station miniSEED (``<station>/*.mseed``), typically
        built with ``examples/build_seisbench_network.py``.
    output_dir
        Where picks and logs are written.
    model
        One of ``PhaseNet``, ``PhaseNetLight``, ``EQTransformer``, ``EQT-NC``,
        or ``Parent/child`` for other SeisBench weights.
    strategy
        ``modelactor`` keeps a pool of persistent workers (recommended).
        ``ripper`` starts a fresh worker task per station (slower cold start;
        useful as a control).
    forward
        ``classify`` uses SeisBench ``classify()`` inside each worker.
        ``slipstream`` uses RAPID's lean reduced-precision forward. Pick
        ``dtype`` when using Slipstream (``fp32``, ``fp16``, or ``bf16``).
    dtype
        Numerical precision for Slipstream. Ignored for ``forward="classify"``
        (SeisBench classify stays FP32). Prefer ``bf16`` for EQTransformer;
        FP16 can overflow its attention padding sentinel.
    n_workers
        How many concurrent station workers to run. Defaults to the number of
        CPUs in ``cpu_ids``.
    cpu_ids
        CPU affinity list for the run. Defaults to ``range(n_workers)`` or all
        visible CPUs.
    gpus
        Physical GPU indices to use, e.g. ``[0]`` or ``[0, 1]``. ``None``
        (default) runs on CPU only.
    start_time, end_time, timechunk_dt
        Time window for the picker. If omitted and ``input_dir/manifest.json``
        exists, values are taken from the synthetic-network manifest.
        ``timechunk_dt`` is in minutes (EQCCTPro convention).
    overlap_samples
        Window overlap in samples for SeisBench / Slipstream windowing.
    batch_size
        Lean batch size for Slipstream megabatches.
    """
    strategy = strategy.lower().replace("-", "").replace("_", "")
    if strategy == "modelactor":
        ripper = False
    elif strategy == "ripper":
        ripper = True
    else:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    forward = forward.lower()
    if forward not in FORWARDS:
        raise ValueError(
            f"forward must be one of {FORWARDS}, got {forward!r}. "
            "For single-process SeisBench annotate/classify without Ray, "
            "use rapid.api.annotate / rapid.api.classify."
        )

    dtype = dtype.lower()
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")

    m = _resolve_model(model)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if start_time is None or end_time is None or timechunk_dt is None:
        meta = load_network_meta(input_dir)
        start_time = start_time or meta["start_time"]
        end_time = end_time or meta["end_time"]
        # Synthetic networks store the window length in seconds under
        # timechunk_dt; EQCCTPro treats timechunk_dt as minutes. For a single
        # short window the chunk is still clamped to [start, end], so either
        # convention yields one chunk. Prefer an explicit minutes value when
        # the window is >= 60 s (common for 6000-sample @ 100 Hz networks).
        if timechunk_dt is None:
            raw = int(meta["timechunk_dt"])
            timechunk_dt = max(1, (raw + 59) // 60) if raw >= 60 else 1

    if cpu_ids is None:
        cpu_ids = _default_cpu_ids(n_workers)
    else:
        cpu_ids = list(cpu_ids)
    if n_workers is None:
        n_workers = len(cpu_ids)

    use_gpu = gpus is not None and len(list(gpus)) > 0
    selected_gpus = list(gpus) if use_gpu else None

    slipstream = forward == "slipstream"
    log_path = Path(log_filepath) if log_filepath else output_dir / "eqcctpro.log"

    runner = RunEQCCTPro(
        use_gpu=use_gpu,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        log_filepath=str(log_path),
        number_of_concurrent_station_predictions=int(n_workers),
        number_of_concurrent_timechunk_predictions=1,
        intra_threads=1,
        inter_threads=1,
        P_threshold=p_threshold,
        S_threshold=s_threshold,
        selected_gpus=selected_gpus,
        cpu_id_list=list(cpu_ids),
        start_time=start_time,
        end_time=end_time,
        timechunk_dt=int(timechunk_dt),
        waveform_overlap=0,
        tmp_dir=str(tmp_dir) if tmp_dir else None,
        model_type="seisbench",
        seisbench_parent_model=m["parent"],
        seisbench_child_model=m["child"],
        Detection_threshold=detection_threshold,
        ripper=ripper,
        slipstream_inference=slipstream,
        slipstream_dtype=dtype if slipstream else "fp32",
        slipstream_compile=bool(compile_model) if slipstream else False,
        slipstream_overlap_samples=int(overlap_samples),
        slipstream_batch_size=int(batch_size),
        seisbench_overlap_samples=int(overlap_samples),
        pick_output_format=pick_output_format,
        ascii_station_pick_format=ascii_station_pick_format,
        overwrite=overwrite,
        **extra,
    )
    runner.run_eqcctpro()
    return runner


def model_actor(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    forward: str = "classify",
    dtype: str = "bf16",
    **kwargs: Any,
) -> RunEQCCTPro:
    """Convenience wrapper: ``pick(..., strategy="modelactor")``."""
    return pick(
        input_dir,
        output_dir,
        strategy="modelactor",
        forward=forward,
        dtype=dtype,
        **kwargs,
    )


def ripper(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    forward: str = "classify",
    dtype: str = "bf16",
    **kwargs: Any,
) -> RunEQCCTPro:
    """Convenience wrapper: ``pick(..., strategy="ripper")``."""
    return pick(
        input_dir,
        output_dir,
        strategy="ripper",
        forward=forward,
        dtype=dtype,
        **kwargs,
    )
