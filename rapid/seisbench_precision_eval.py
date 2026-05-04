"""Evaluate numeric precision (FP32 / FP16 / BF16) vs catalog P/S picks on SeisBench datasets.

Loads STEAD, TXED, GEOFON, ETHZ (and optionally others) from the SeisBench cache
(``SEISBENCH_CACHE_ROOT`` / ``~/.seisbench`` by default). For each trace with
catalog P (and optionally S) picks, runs the lean PyTorch backend at several
dtypes and reports:

- pick drift vs **FP32** (argmax and simple onset metrics, mirroring ``rapid.quality``)
- absolute error vs **catalog** samples in the model window
- probability-trace MAE vs FP32 (``rapid.quality.compare_probabilities``)

Waveforms are resampled with :meth:`seisbench.data.WaveformDataset.get_sample`,
then passed through each model's ``annotate_stream_pre`` so filtering matches
``annotate()`` / RAPID benchmarks.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import seisbench.data as sbd
import seisbench.models as sbm
from obspy import Stream, Trace, UTCDateTime

from rapid.backends.base import BackendError
from rapid.backends.lean_pytorch import LeanPyTorchBackend
from rapid.data import stream_to_3c_array
from rapid.quality import TraceStats, as_dict, compare_probabilities, extract_picks_simple


@dataclass(frozen=True)
class ModelSpec:
    parent: str
    child: str
    label: str


DEFAULT_MODELS: Tuple[ModelSpec, ...] = (
    ModelSpec("PhaseNet", "original", "PhaseNet"),
    ModelSpec("PhaseNetLight", "stead", "PhaseNetLight"),
    ModelSpec("EQTransformer", "original", "EQTransformer"),
    ModelSpec("EQTransformer", "original_nonconservative", "EQT-NC"),
)

# Case-insensitive tokens for CLI / RAPID_PRECISION_MODELS (values = canonical label).
MODEL_ARG_ALIASES: Dict[str, str] = {
    "phasenet": "PhaseNet",
    "pn": "PhaseNet",
    "phasenetlight": "PhaseNetLight",
    "pnl": "PhaseNetLight",
    "eqtransformer": "EQTransformer",
    "eqt": "EQTransformer",
    "eqt-nc": "EQT-NC",
    "eqt_nc": "EQT-NC",
    "eqnc": "EQT-NC",
}


def parse_models_arg(arg: str) -> Tuple[ModelSpec, ...]:
    """Parse ``'all'`` or comma-separated labels / aliases into :class:`ModelSpec` tuples."""
    s = arg.strip()
    if not s or s.lower() == "all":
        return DEFAULT_MODELS
    by_label = {m.label.lower(): m for m in DEFAULT_MODELS}
    chosen: List[ModelSpec] = []
    seen: set[str] = set()
    for raw in s.split(","):
        tok = raw.strip()
        if not tok:
            continue
        canon = MODEL_ARG_ALIASES.get(tok.lower(), tok)
        key = canon.strip().lower()
        if key not in by_label:
            allowed = ", ".join(m.label for m in DEFAULT_MODELS)
            raise ValueError(
                f"Unknown model {tok!r} (resolved {canon!r}). Choose from: {allowed}"
            )
        m = by_label[key]
        if m.label not in seen:
            seen.add(m.label)
            chosen.append(m)
    if not chosen:
        raise ValueError("Empty --models list after parsing")
    return tuple(chosen)


def list_model_choices() -> str:
    lines = [
        "Available models (use --models, or env RAPID_PRECISION_MODELS):",
        "  Aliases: pn PhaseNet | pnl PhaseNetLight | eqt EQTransformer | eqt-nc EQT-NC",
        "",
    ]
    for m in DEFAULT_MODELS:
        lines.append(f"  {m.label:16}  parent={m.parent!r}  child={m.child!r}")
    return "\n".join(lines)


DATASET_CLASSES = {
    "stead": sbd.STEAD,
    "txed": sbd.TXED,
    "geofon": sbd.GEOFON,
    "ethz": sbd.ETHZ,
    "instancecounts": sbd.InstanceCounts,
}


def _finite_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(round(v))


def phase_indices(model) -> Tuple[int, int]:
    """Return (P_channel, S_channel) indices in the probability tensor."""
    labs = list(model.labels)
    if "P" not in labs or "S" not in labs:
        raise ValueError(f"Model {model} has no P/S in labels: {labs}")
    return labs.index("P"), labs.index("S")


def waves_to_stream(waves: np.ndarray, sampling_rate: float, component_order: str) -> Stream:
    """``waves``: (C, T) in dataset component order."""
    st = Stream()
    for i, comp in enumerate(component_order):
        tr = Trace(data=np.asarray(waves[i], dtype=np.float64))
        tr.stats.starttime = UTCDateTime(0)
        tr.stats.sampling_rate = float(sampling_rate)
        tr.stats.channel = f"HH{comp}"
        tr.stats.network = "SB"
        tr.stats.station = "X"
        st += tr
    return st


def preprocess_array(
    model,
    waves: np.ndarray,
    sampling_rate: float,
    component_order: str,
) -> Optional[np.ndarray]:
    """Return (3, T) float32 after ``annotate_stream_pre`` + 3C stack."""
    st = waves_to_stream(waves, sampling_rate, component_order)
    try:
        st_f = model.annotate_stream_pre(st, {})
    except Exception:
        return None
    arr = stream_to_3c_array(st_f, model.component_order)
    return arr


def cut_window(
    arr: np.ndarray,
    in_samples: int,
    p_sample: int,
) -> Tuple[np.ndarray, int, int]:
    """Crop or pad ``arr`` (C, T) to (C, in_samples) centered on ``p_sample``.

    Returns ``(window, start_offset, p_in_window)`` where catalog sample index
    ``p_sample`` maps to ``p_in_window = p_sample - start_offset``.
    """
    _, t_len = arr.shape
    if t_len >= in_samples:
        start = int(np.clip(p_sample - in_samples // 2, 0, t_len - in_samples))
        win = np.asarray(arr[:, start : start + in_samples], dtype=np.float32)
        p_win = p_sample - start
        return win, start, p_win
    win = np.zeros((arr.shape[0], in_samples), dtype=np.float32)
    win[:, :t_len] = np.asarray(arr[:, :t_len], dtype=np.float32)
    if p_sample < 0 or p_sample >= t_len:
        raise ValueError("P pick outside trace after preprocess")
    p_win = p_sample
    return win, 0, p_win


def _dtype_runs_for_model(parent: str, dtypes: Sequence[str]) -> List[str]:
    out: List[str] = []
    for d in dtypes:
        if d == "fp16" and parent == "EQTransformer":
            continue
        out.append(d)
    return out


def evaluate_trace(
    batch: np.ndarray,
    *,
    parent: str,
    child: str,
    device: str,
    dtypes: Sequence[str],
    p_idx: int,
    s_idx: int,
    p_win: int,
    s_win: Optional[int],
    prob_threshold: float = 0.3,
    include_fp16_compile: bool,
    backend_cache: Optional[Dict[Tuple[str, bool], LeanPyTorchBackend]] = None,
) -> Dict[str, Any]:
    """Return nested dict with per-dtype metrics vs fp32 and vs catalog.

    When ``backend_cache`` is set, backends are reused (caller must ``close()``
    them). Otherwise each run loads and disposes its own backend.
    """
    ref_pred: Optional[np.ndarray] = None
    out: Dict[str, Any] = {"per_dtype": {}}

    filtered = _dtype_runs_for_model(parent, dtypes)
    # FP32 must run first so later dtypes compare to the same reference tensor.
    ordered = sorted(filtered, key=lambda d: (0 if d == "fp32" else 1, d))
    runs: List[Tuple[str, bool]] = [(d, False) for d in ordered]
    if include_fp16_compile and parent != "EQTransformer" and "fp16" in dtypes:
        runs.append(("fp16", True))

    for dtype, use_compile in runs:
        key = f"{dtype}_compile" if use_compile else dtype
        try:
            if backend_cache is not None:
                bc_key = (dtype, use_compile)
                if bc_key not in backend_cache:
                    b = LeanPyTorchBackend(
                        parent,
                        child,
                        device=device,
                        dtype=dtype,
                        compile=use_compile,
                    )
                    b.load()
                    backend_cache[bc_key] = b
                pred = backend_cache[bc_key].infer_batch(batch)
            else:
                be = LeanPyTorchBackend(
                    parent, child, device=device, dtype=dtype, compile=use_compile
                )
                be.load()
                pred = be.infer_batch(batch)
                be.close()
        except BackendError as e:
            out["per_dtype"][key] = {"error": str(e)}
            continue
        entry: Dict[str, Any] = {"shape": list(pred.shape)}
        # Argmax picks (samples)
        entry["argmax_p"] = int(np.argmax(pred[0, :, p_idx]))
        entry["argmax_s"] = int(np.argmax(pred[0, :, s_idx]))
        entry["delta_p_vs_catalog"] = entry["argmax_p"] - p_win
        if s_win is not None:
            entry["delta_s_vs_catalog"] = entry["argmax_s"] - s_win
        # Onset vs catalog (first rising edge)
        p_on = extract_picks_simple(pred[0, :, p_idx], threshold=prob_threshold)
        s_on = extract_picks_simple(pred[0, :, s_idx], threshold=prob_threshold)
        entry["onset_p"] = int(p_on[0]) if p_on.size else None
        entry["onset_s"] = int(s_on[0]) if s_on.size else None
        if entry["onset_p"] is not None:
            entry["onset_delta_p_vs_catalog"] = entry["onset_p"] - p_win
        if s_win is not None and entry["onset_s"] is not None:
            entry["onset_delta_s_vs_catalog"] = entry["onset_s"] - s_win

        if ref_pred is None:
            ref_pred = pred
            entry["vs_fp32"] = None
        else:
            st: TraceStats = compare_probabilities(ref_pred, pred)
            entry["vs_fp32"] = as_dict(st)
            # pick drift vs fp32
            entry["delta_argmax_p_vs_fp32"] = int(
                np.argmax(pred[0, :, p_idx]) - np.argmax(ref_pred[0, :, p_idx])
            )
            entry["delta_argmax_s_vs_fp32"] = int(
                np.argmax(pred[0, :, s_idx]) - np.argmax(ref_pred[0, :, s_idx])
            )
        out["per_dtype"][key] = entry
    return out


def load_dataset(name: str):
    cls = DATASET_CLASSES[name.lower()]
    return cls()


def catalog_pick_columns(ds) -> Tuple[str, Optional[str]]:
    """Return (p_col, s_col) CSV/metadata keys for catalog P and S sample columns."""
    cols = set(ds.metadata.columns)
    if "trace_p_arrival_sample" in cols:
        p_col = "trace_p_arrival_sample"
        s_col = "trace_s_arrival_sample" if "trace_s_arrival_sample" in cols else None
        return p_col, s_col
    if "trace_P_arrival_sample" in cols:
        p_col = "trace_P_arrival_sample"
        s_col = "trace_S_arrival_sample" if "trace_S_arrival_sample" in cols else None
        return p_col, s_col
    raise ValueError(
        "Dataset metadata has no recognized P pick column "
        "(expected trace_p_arrival_sample or trace_P_arrival_sample)."
    )


def catalog_mask(
    ds,
    require_s: bool,
) -> np.ndarray:
    """Boolean mask over row indices with valid P (and optionally S) picks."""
    meta = ds.metadata
    p_col, s_col = catalog_pick_columns(ds)
    p = meta[p_col]
    ok = p.notna() & np.isfinite(p.astype(float))
    if require_s:
        if s_col is None or s_col not in meta.columns:
            return np.zeros(len(ds), dtype=bool)
        s = meta[s_col]
        ok &= s.notna() & np.isfinite(s.astype(float))
    return ok.to_numpy()


def run_evaluation(
    *,
    datasets: Sequence[str],
    models: Sequence[ModelSpec],
    max_per_dataset: int,
    device: str,
    dtypes: Sequence[str],
    require_both_ps: bool,
    seed: int,
    include_fp16_compile: bool,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    results: List[Dict[str, Any]] = []

    dtypes_eff: List[str] = []
    for d in dtypes:
        if d not in dtypes_eff:
            dtypes_eff.append(d)
    if "fp32" not in dtypes_eff:
        dtypes_eff.insert(0, "fp32")

    for dname in datasets:
        try:
            ds = load_dataset(dname)
        except Exception as exc:
            # e.g. partial download, missing chunk — skip on shared machines.
            print(f"[skip] dataset {dname!r}: {exc}", flush=True)
            continue
        mask = catalog_mask(ds, require_both_ps)
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            continue
        take = min(max_per_dataset, idxs.size)
        chosen = rng.choice(idxs, size=take, replace=False)

        p_col, s_col = catalog_pick_columns(ds)
        for spec in models:
            backend_cache: Dict[Tuple[str, bool], LeanPyTorchBackend] = {}
            try:
                cls = getattr(sbm, spec.parent)
                model = cls.from_pretrained(spec.child)
                p_i, s_i = phase_indices(model)
            except Exception:
                continue

            for row_idx in chosen:
                row_idx = int(row_idx)
                sr = float(model.sampling_rate)
                try:
                    waves, meta = ds.get_sample(row_idx, sampling_rate=sr)
                except Exception:
                    continue
                co = str(meta.get("trace_component_order") or "ZNE")
                if waves.ndim != 2:
                    continue
                arr = preprocess_array(model, waves, sr, co)
                if arr is None:
                    continue
                p_cat = _finite_int(meta.get(p_col))
                if p_cat is None or not (0 <= p_cat < arr.shape[1]):
                    continue
                s_cat = _finite_int(meta.get(s_col)) if s_col else None
                if require_both_ps:
                    if s_cat is None or not (0 <= s_cat < arr.shape[1]):
                        continue
                else:
                    if s_cat is not None and not (0 <= s_cat < arr.shape[1]):
                        s_cat = None

                try:
                    win, start, p_win = cut_window(arr, model.in_samples, p_cat)
                except ValueError:
                    continue
                if s_cat is not None:
                    s_win = s_cat - start
                    if not (0 <= s_win < model.in_samples):
                        s_win = None
                else:
                    s_win = None

                batch = win[None, ...].astype(np.float32, copy=False)

                metrics = evaluate_trace(
                    batch,
                    parent=spec.parent,
                    child=spec.child,
                    device=device,
                    dtypes=dtypes_eff,
                    p_idx=p_i,
                    s_idx=s_i,
                    p_win=p_win,
                    s_win=s_win,
                    include_fp16_compile=include_fp16_compile,
                    backend_cache=backend_cache,
                )
                rec = {
                    "dataset": dname.lower(),
                    "trace_row": row_idx,
                    "catalog_p_column": p_col,
                    "catalog_s_column": s_col,
                    "model_label": spec.label,
                    "model_parent": spec.parent,
                    "model_child": spec.child,
                    "device": device,
                    "in_samples": model.in_samples,
                    "p_catalog_in_window": p_win,
                    "s_catalog_in_window": s_win,
                    "metrics": metrics,
                }
                results.append(rec)

            for b in backend_cache.values():
                b.close()
    return results


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
