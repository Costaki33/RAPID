"""Waveform loading and windowing utilities shared by every backend.

We deliberately delegate the *preprocessing* (taper, bandpass, resample) to
SeisBench's own ``annotate_stream_pre`` via the pretrained model's filter
configuration, so all backends see the exact same inputs. RAPID's only job
here is:

1. Discover the first ``n_stations`` stations under a dataset root.
2. Load their mSEED files into ObsPy Streams once (shared across repeats).
3. Run SeisBench-style preprocessing (filter + resample) to get a 3C array per
   station at the model's expected sampling rate.
4. Cut those arrays into ``(in_samples,)`` windows with a caller-specified
   overlap, returning a ``(N_total_windows, C, in_samples)`` float32 array plus
   metadata that lets us reassemble predictions per station.
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import obspy


LOG = logging.getLogger("rapid.data")


# ---------------------------------------------------------------------------
# Station discovery
# ---------------------------------------------------------------------------


def list_stations(dataset_dir: str | os.PathLike) -> List[str]:
    """Return all station directory names under a timechunk directory.

    Matches the nested layout ``<dataset_dir>/<station>/*.mseed``.
    """
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {root}")
    stations = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and any(p.glob("*.mseed"))
    )
    return stations


def select_stations(dataset_dir: str | os.PathLike, n: int) -> List[str]:
    all_s = list_stations(dataset_dir)
    if n > len(all_s):
        raise ValueError(f"Requested {n} stations; only {len(all_s)} available in {dataset_dir}")
    return all_s[:n]


# ---------------------------------------------------------------------------
# Stream loading
# ---------------------------------------------------------------------------


def load_station_stream(dataset_dir: str | os.PathLike, station: str) -> obspy.Stream:
    """Read a single station's mSEED files, merge, demean. No filter/resample yet."""
    files = glob.glob(str(Path(dataset_dir) / station / "*.mseed"))
    if not files:
        return obspy.Stream()
    st = obspy.Stream()
    for f in files:
        try:
            tmp = obspy.read(f)
            st += tmp
        except Exception as e:
            LOG.warning("Failed to read %s: %s", f, e)
    if len(st) == 0:
        return st
    try:
        st.merge(method=1, fill_value=0)
    except Exception:
        pass
    st.detrend("demean")
    return st


def load_all_streams(
    dataset_dir: str | os.PathLike, stations: Sequence[str]
) -> List[Tuple[str, obspy.Stream]]:
    out: List[Tuple[str, obspy.Stream]] = []
    for sta in stations:
        st = load_station_stream(dataset_dir, sta)
        if len(st) > 0:
            out.append((sta, st))
    return out


# ---------------------------------------------------------------------------
# SeisBench-parity preprocessing (uses the model's own annotate_stream_pre)
# ---------------------------------------------------------------------------


def preprocess_for_model(model, stream: obspy.Stream, argdict: Optional[dict] = None) -> obspy.Stream:
    """Run the same preprocessing SeisBench would do inside annotate().

    This is ``WaveformModel.annotate_stream_pre`` — bandpass/filter (if the
    pretrained weights configured any), then resample to ``model.sampling_rate``.
    """
    if argdict is None:
        argdict = {}
    st = stream.copy()
    try:
        st.merge(-1)
    except Exception:
        pass
    return model.annotate_stream_pre(st, argdict)


# ---------------------------------------------------------------------------
# 3C array extraction per station
# ---------------------------------------------------------------------------


_ENZ_PRIMARY = ("E", "N", "Z")
_ENZ_ALT = ("1", "2", "Z")


def stream_to_3c_array(
    stream: obspy.Stream, component_order: str = "ZNE"
) -> Optional[np.ndarray]:
    """Extract a ``(3, T)`` float32 array from a 3C stream in a given component order.

    Returns None if fewer than 3 components or time ranges can't be aligned.
    """
    if len(stream) < 3:
        return None

    by_last: dict[str, obspy.Trace] = {}
    for tr in stream:
        ch = (tr.stats.channel or "").strip()
        if not ch:
            continue
        by_last.setdefault(ch[-1].upper(), tr)

    def _pick(comp: str) -> Optional[obspy.Trace]:
        if comp in by_last:
            return by_last[comp]
        if comp == "N" and "2" in by_last:
            return by_last["2"]
        if comp == "E" and "1" in by_last:
            return by_last["1"]
        return None

    traces = [_pick(c) for c in component_order]
    if any(t is None for t in traces):
        return None

    t0 = max(tr.stats.starttime for tr in traces)  # type: ignore[union-attr]
    t1 = min(tr.stats.endtime for tr in traces)  # type: ignore[union-attr]
    if t1 <= t0:
        return None

    arrs = []
    for tr in traces:
        t = tr.copy().trim(t0, t1, pad=False)  # type: ignore[union-attr]
        arrs.append(np.asarray(t.data, dtype=np.float32))
    min_len = min(a.shape[0] for a in arrs)
    return np.stack([a[:min_len] for a in arrs], axis=0)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


@dataclass
class WindowSpec:
    in_samples: int
    overlap_samples: int = 0


def window_from_array(arr: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Slide a ``(C, T)`` array into ``(n_windows, C, in_samples)`` chunks."""
    C, T = arr.shape
    step = max(1, spec.in_samples - spec.overlap_samples)
    if T < spec.in_samples:
        pad = np.zeros((C, spec.in_samples), dtype=arr.dtype)
        pad[:, :T] = arr
        return pad[None, ...]
    starts = list(range(0, T - spec.in_samples + 1, step))
    # Always include a tail window so we cover to the end
    if starts[-1] + spec.in_samples < T:
        starts.append(T - spec.in_samples)
    windows = np.stack(
        [arr[:, s : s + spec.in_samples] for s in starts], axis=0
    )
    return windows


@dataclass
class Megabatch:
    """A single contiguous tensor batch of every station's windows.

    Attributes
    ----------
    windows : np.ndarray
        Shape ``(N_total_windows, C, in_samples)``.
    station_of_window : np.ndarray
        Shape ``(N_total_windows,)``, integer station index for each window.
    station_ids : list[str]
        Station codes in order (``station_of_window`` indexes into this).
    n_windows_per_station : list[int]
        How many windows each station contributed.
    in_samples : int
    overlap_samples : int
    """

    windows: np.ndarray
    station_of_window: np.ndarray
    station_ids: List[str]
    n_windows_per_station: List[int]
    in_samples: int
    overlap_samples: int

    @property
    def total_windows(self) -> int:
        return int(self.windows.shape[0])


def build_megabatch(
    arrays: List[Tuple[str, np.ndarray]], spec: WindowSpec
) -> Megabatch:
    """Concatenate per-station ``(C, T)`` arrays into a single tensor of windows."""
    all_windows = []
    owners = []
    station_ids: List[str] = []
    n_per: List[int] = []
    for i, (sta, arr) in enumerate(arrays):
        w = window_from_array(arr, spec)
        all_windows.append(w)
        owners.extend([i] * w.shape[0])
        station_ids.append(sta)
        n_per.append(w.shape[0])
    windows = np.concatenate(all_windows, axis=0).astype(np.float32, copy=False)
    return Megabatch(
        windows=windows,
        station_of_window=np.asarray(owners, dtype=np.int32),
        station_ids=station_ids,
        n_windows_per_station=n_per,
        in_samples=spec.in_samples,
        overlap_samples=spec.overlap_samples,
    )
