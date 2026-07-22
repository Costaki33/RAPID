"""Single-process SeisBench picking helpers (no Ray).

These cover the native baselines used in the fair benchmark: ``annotate``,
``classify``, and RAPID Slipstream. For network-scale runs with Model-Actor or
Ripper, use ``eqcctpro.api.pick`` instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from obspy import Stream, read

PathLike = Union[str, Path]

MODELS: Dict[str, Dict[str, str]] = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"parent": "EQTransformer", "child": "nonconservative"},
}


def _resolve_model(model: str) -> Dict[str, str]:
    if model in MODELS:
        return MODELS[model]
    if "/" in model:
        parent, child = model.split("/", 1)
        return {"parent": parent, "child": child}
    raise ValueError(
        f"Unknown model '{model}'. Choose one of {list(MODELS)} "
        "or pass 'ParentModel/child_weights'."
    )


def _load_seisbench_model(model: str, device: str):
    import seisbench.models as sbm
    import torch

    m = _resolve_model(model)
    cls = getattr(sbm, m["parent"])
    sb_model = cls.from_pretrained(m["child"])
    sb_model.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        sb_model.to(torch.device(device))
    else:
        sb_model.to(torch.device("cpu"))
        device = "cpu"
    return sb_model, device


def _station_dirs(input_dir: Path) -> List[Path]:
    dirs = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name != "__pycache__")
    if not dirs:
        raise FileNotFoundError(f"No station directories under {input_dir}")
    return dirs


def _read_station_stream(sta_dir: Path) -> Stream:
    files = sorted(sta_dir.glob("*.mseed")) + sorted(sta_dir.glob("*.ms"))
    if not files:
        raise FileNotFoundError(f"No miniSEED in {sta_dir}")
    st = Stream()
    for f in files:
        st += read(str(f))
    return st


def _merge_network(input_dir: Path, n_stations: Optional[int] = None) -> Stream:
    merged = Stream()
    for i, sta_dir in enumerate(_station_dirs(input_dir)):
        if n_stations is not None and i >= n_stations:
            break
        merged += _read_station_stream(sta_dir)
    return merged


def _write_classify_picks(output_dir: Path, station: str, result: Any) -> None:
    """Best-effort dump of SeisBench classify picks to a simple CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{station}_picks.csv"
    picks = getattr(result, "picks", result)
    rows: List[str] = ["station,phase,peak_time,peak_value"]
    try:
        for p in picks:
            phase = getattr(p, "phase", getattr(p, "type", ""))
            t = getattr(p, "peak_time", getattr(p, "start_time", ""))
            v = getattr(p, "peak_value", getattr(p, "value", ""))
            rows.append(f"{station},{phase},{t},{v}")
    except TypeError:
        rows.append(f"{station},,,{picks!r}")
    path.write_text("\n".join(rows) + "\n")


def annotate(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    model: str = "PhaseNet",
    device: str = "cpu",
    batch_size: int = 256,
    n_stations: Optional[int] = None,
    torch_threads: int = 1,
    **annotate_kwargs: Any,
) -> Stream:
    """Run SeisBench ``model.annotate()`` on the merged network stream.

    Returns the probability streams SeisBench produces. This is the native
    single-process batched path — not Ray Model-Actor.
    """
    import torch

    torch.set_num_threads(int(torch_threads))
    sb_model, device = _load_seisbench_model(model, device)
    merged = _merge_network(Path(input_dir), n_stations=n_stations)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(batch_size=batch_size)
    kw.update(annotate_kwargs)
    probs = sb_model.annotate(merged, **kw)
    probs.write(str(out / "annotate_probs.mseed"), format="MSEED")
    return probs


def classify(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    model: str = "PhaseNet",
    device: str = "cpu",
    n_stations: Optional[int] = None,
    torch_threads: int = 1,
    batched: bool = False,
    **classify_kwargs: Any,
) -> List[Any]:
    """Run SeisBench ``model.classify()``.

    By default this is one station at a time (the serial twin of
    Model-Actor[classify]). Set ``batched=True`` to classify the merged
    network in one call (SeisBench's best single-process picking path).
    """
    import torch

    torch.set_num_threads(int(torch_threads))
    sb_model, device = _load_seisbench_model(model, device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: List[Any] = []

    if batched:
        merged = _merge_network(Path(input_dir), n_stations=n_stations)
        result = sb_model.classify(merged, **classify_kwargs)
        _write_classify_picks(out, "NETWORK", result)
        results.append(result)
        return results

    for i, sta_dir in enumerate(_station_dirs(Path(input_dir))):
        if n_stations is not None and i >= n_stations:
            break
        st = _read_station_stream(sta_dir)
        result = sb_model.classify(st, **classify_kwargs)
        _write_classify_picks(out, sta_dir.name, result)
        results.append(result)
    return results


def slipstream(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    model: str = "PhaseNet",
    device: str = "cpu",
    dtype: str = "bf16",
    compile_model: bool = False,
    batch_size: int = 256,
    n_stations: Optional[int] = None,
    torch_threads: int = 1,
    overlap_samples: int = 0,
    p_threshold: float = 0.3,
    s_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run RAPID's lean Slipstream forward on a synthetic or local network.

    Builds windowed batches the same way the fair benchmark does, runs the
    lean PyTorch backend at ``dtype`` (``fp32`` / ``fp16`` / ``bf16``), and
    writes picks under ``output_dir``. Prefer ``bf16`` for EQTransformer —
    FP16 can overflow its attention padding sentinel.
    """
    import torch

    from rapid.backends.lean_pytorch import LeanPyTorchBackend
    from rapid.benchmark.fairness import build_windowed_batch, windows_to_station_picks
    from rapid.seisbench_precision_eval import phase_indices

    torch.set_num_threads(int(torch_threads))
    m = _resolve_model(model)
    backend = LeanPyTorchBackend(
        parent_model=m["parent"],
        child_model=m["child"],
        device=device,
        dtype=dtype,
        compile=compile_model,
    )
    backend.load()

    try:
        streams: List[Tuple[str, Stream]] = []
        for i, sta_dir in enumerate(_station_dirs(Path(input_dir))):
            if n_stations is not None and i >= n_stations:
                break
            streams.append((sta_dir.name, _read_station_stream(sta_dir)))

        in_samples = int(getattr(backend._raw_model, "in_samples", 3001) or 3001)
        station_ids, windows, n_per, starts = build_windowed_batch(
            backend._raw_model,
            streams,
            in_samples,
            overlap_samples,
            component_order=getattr(backend, "component_order", None),
        )
        preds = backend.infer_chunked(windows, batch_size=max(1, int(batch_size)))
        p_idx, s_idx = phase_indices(backend._raw_model)
        picks = windows_to_station_picks(
            preds,
            station_ids,
            n_per,
            starts,
            p_idx=p_idx,
            s_idx=s_idx,
            p_threshold=p_threshold,
            s_threshold=s_threshold,
        )
    finally:
        backend.close()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": model,
        "dtype": dtype,
        "device": device,
        "n_stations": len(station_ids),
        "batch_shape": list(np.shape(windows)),
        "n_pick_stations": len(picks),
    }
    (out / "slipstream_summary.json").write_text(json.dumps(summary, indent=2))
    (out / "slipstream_picks.json").write_text(json.dumps(picks, indent=2, default=str))
    return {"summary": summary, "picks": picks, "preds": preds}
