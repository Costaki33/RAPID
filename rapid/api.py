"""Single-process SeisBench picking helpers (no Ray).

These cover the native baselines used in the fair benchmark: ``annotate``,
``classify``, and reduced-precision Annotate (``annotate_bf16`` /
``annotate_fp16``). For network-scale runs with Model-Actor or Ripper, use
``rapid.pick`` (or ``rapid.orchestration.pick``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from obspy import Stream, read

PathLike = Union[str, Path]

MODELS: Dict[str, Dict[str, str]] = {
    "PhaseNet": {"parent": "PhaseNet", "child": "original"},
    "PhaseNetLight": {"parent": "PhaseNetLight", "child": "stead"},
    "EQTransformer": {"parent": "EQTransformer", "child": "original"},
    "EQT-NC": {"parent": "EQTransformer", "child": "original_nonconservative"},
    # SeisBench main: EQCCT is two WaveformModels (P-branch + S-branch).
    "EQCCTP": {"parent": "EQCCTP", "child": "original"},
    "EQCCTS": {"parent": "EQCCTS", "child": "original"},
}

DTYPES = ("fp32", "fp16", "bf16")

EQT_FP16_MESSAGE = (
    "EQTransformer cannot run in fp16: it hard-codes -1e10 as a "
    "pooling pad sentinel, which overflows fp16. Use dtype='bf16' "
    "(same 16-bit storage, full fp32 exponent range) or dtype='fp32'."
)

_DTYPE_MAP = {
    "fp32": "float32",
    "fp16": "float16",
    "bf16": "bfloat16",
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


def _normalize_dtype(dtype: str) -> str:
    d = str(dtype).lower().strip()
    if d not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
    return d


def _wrap_forward_cast(model, torch_dtype):
    """Make SeisBench annotate's FP32 buffers compatible with cast weights.

    ``model.annotate()`` builds float32 tensors internally. After
    ``model.to(bf16/fp16)``, the first conv would otherwise fail with
    ``expected scalar type Float but found BFloat16``. Casting inputs in
    ``forward`` (and returning float32 outputs) keeps Annotate's pipeline
    intact while running the network at the requested precision.
    """
    import torch

    if getattr(model, "_rapid_dtype_wrapped", False):
        return model

    orig_forward = model.forward

    def forward(x, *args, **kwargs):
        if torch.is_tensor(x) and x.dtype != torch_dtype:
            x = x.to(dtype=torch_dtype)
        out = orig_forward(x, *args, **kwargs)
        if torch.is_tensor(out):
            return out.float()
        if isinstance(out, (tuple, list)):
            casted = []
            for o in out:
                casted.append(o.float() if torch.is_tensor(o) else o)
            return type(out)(casted)
        return out

    model.forward = forward  # type: ignore[method-assign]
    model._rapid_dtype_wrapped = True
    return model


def _load_seisbench_model(model: str, device: str, dtype: str = "fp32"):
    """Load a pretrained SeisBench model and optionally cast weights."""
    import seisbench.models as sbm
    import torch

    dtype = _normalize_dtype(dtype)
    m = _resolve_model(model)
    if m["parent"] == "EQTransformer" and dtype == "fp16":
        raise ValueError(EQT_FP16_MESSAGE)

    cls = getattr(sbm, m["parent"])
    sb_model = cls.from_pretrained(m["child"])
    sb_model.eval()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch_device = torch.device(device)
    else:
        torch_device = torch.device("cpu")
        device = "cpu"

    sb_model.to(torch_device)
    if dtype in ("fp16", "bf16"):
        torch_dtype = getattr(torch, _DTYPE_MAP[dtype])
        sb_model.to(torch_dtype)
        sb_model = _wrap_forward_cast(sb_model, torch_dtype)

    return sb_model, device, dtype


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
    dtype: str = "fp32",
    batch_size: int = 256,
    n_stations: Optional[int] = None,
    torch_threads: int = 1,
    **annotate_kwargs: Any,
) -> Stream:
    """Run SeisBench ``model.annotate()`` on the merged network stream.

    ``dtype`` selects numerical precision for the forward pass:

    * ``fp32`` — native SeisBench weights (default)
    * ``bf16`` — cast weights to bfloat16 before annotate
    * ``fp16`` — cast weights to float16 before annotate (not valid for EQTransformer)

    Returns the same ObsPy probability streams SeisBench Annotate produces.
    Pass those streams to :func:`classify_from_annotations` (or the model's
    ``classify_aggregate``) for discrete P/S picks.
    """
    import torch

    torch.set_num_threads(int(torch_threads))
    sb_model, device, dtype = _load_seisbench_model(model, device, dtype=dtype)
    merged = _merge_network(Path(input_dir), n_stations=n_stations)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(batch_size=batch_size)
    kw.update(annotate_kwargs)
    probs = sb_model.annotate(merged, **kw)
    suffix = "" if dtype == "fp32" else f"_{dtype}"
    probs.write(str(out / f"annotate{suffix}_probs.mseed"), format="MSEED")
    return probs


def annotate_bf16(
    input_dir: PathLike,
    output_dir: PathLike,
    **kwargs: Any,
) -> Stream:
    """Convenience wrapper: ``annotate(..., dtype="bf16")``."""
    kwargs.pop("dtype", None)
    return annotate(input_dir, output_dir, dtype="bf16", **kwargs)


def annotate_fp16(
    input_dir: PathLike,
    output_dir: PathLike,
    **kwargs: Any,
) -> Stream:
    """Convenience wrapper: ``annotate(..., dtype="fp16")``."""
    kwargs.pop("dtype", None)
    return annotate(input_dir, output_dir, dtype="fp16", **kwargs)


def classify_from_annotations(
    model: Any,
    annotations: Stream,
    **thresholds: Any,
) -> Any:
    """Turn Annotate probability streams into discrete picks.

    This is SeisBench's own ``classify_aggregate`` — the same step Classify
    runs after Annotate — so reduced-precision Annotate outputs stay in the
    Classify pick family.
    """
    argdict = dict(getattr(model, "default_args", {}) or {})
    argdict.update(thresholds)
    return model.classify_aggregate(annotations, argdict)


def classify(
    input_dir: PathLike,
    output_dir: PathLike,
    *,
    model: str = "PhaseNet",
    device: str = "cpu",
    dtype: str = "fp32",
    n_stations: Optional[int] = None,
    torch_threads: int = 1,
    batched: bool = False,
    **classify_kwargs: Any,
) -> List[Any]:
    """Run SeisBench picking, optionally at reduced Annotate precision.

    By default (``dtype="fp32"``) this calls ``model.classify()`` — one station
    at a time, or the merged network when ``batched=True``.

    When ``dtype`` is ``bf16`` or ``fp16``, this runs Annotate at that precision
    then ``classify_aggregate`` (same composition SeisBench uses internally),
    so discrete picks match the Classify extractor family.
    """
    import torch

    torch.set_num_threads(int(torch_threads))
    dtype = _normalize_dtype(dtype)
    sb_model, device, dtype = _load_seisbench_model(model, device, dtype=dtype)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: List[Any] = []

    use_precision_annotate = dtype in ("fp16", "bf16")
    annotate_kw = dict(batch_size=int(classify_kwargs.pop("batch_size", 256)))
    thr_kw = dict(classify_kwargs)

    def _one(stream: Stream) -> Any:
        if use_precision_annotate:
            annotations = sb_model.annotate(stream, **annotate_kw)
            return classify_from_annotations(sb_model, annotations, **thr_kw)
        return sb_model.classify(stream, **thr_kw)

    if batched:
        merged = _merge_network(Path(input_dir), n_stations=n_stations)
        result = _one(merged)
        _write_classify_picks(out, "NETWORK", result)
        return [result]

    for i, sta_dir in enumerate(_station_dirs(Path(input_dir))):
        if n_stations is not None and i >= n_stations:
            break
        result = _one(_read_station_stream(sta_dir))
        _write_classify_picks(out, sta_dir.name, result)
        results.append(result)
    return results


def slipstream(*args: Any, **kwargs: Any) -> None:
    """Removed. Use :func:`annotate` with ``dtype='bf16'`` or ``dtype='fp16'``.

    Reduced-precision inference is now SeisBench Annotate after a weight cast.
    Discrete picks come from :func:`classify_from_annotations` (SeisBench
    ``classify_aggregate``), not RAPID's old threshold extractor.
    """
    raise RuntimeError(
        "rapid.slipstream was removed. Use rapid.annotate(..., dtype='bf16') "
        "or rapid.annotate_fp16 / rapid.annotate_bf16. For discrete picks, pass "
        "the probability Stream to rapid.classify_from_annotations (SeisBench "
        "classify_aggregate)."
    )
