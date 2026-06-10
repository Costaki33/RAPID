"""Export detected picks for JSON storage and orchestration-compatible CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from obspy import UTCDateTime

SR = 100.0


def _pick_time_to_sample(t: Any, t0: UTCDateTime) -> Optional[float]:
    if t is None:
        return None
    try:
        return float((UTCDateTime(t) - t0) * SR)
    except Exception:
        return None


def picks_from_classify_output(picks: Iterable[Any], t0: UTCDateTime) -> Dict[str, List[float]]:
    p_samples: List[float] = []
    s_samples: List[float] = []
    for pick in picks or []:
        phase = getattr(pick, "phase", "P")
        phase = str(phase).upper() if phase else "P"
        t = getattr(pick, "peak_time", getattr(pick, "start_time", getattr(pick, "time", None)))
        samp = _pick_time_to_sample(t, t0)
        if samp is None:
            continue
        if phase == "S":
            s_samples.append(samp)
        else:
            p_samples.append(samp)
    return {"p": p_samples, "s": s_samples}


def picks_from_lean_preds(
    preds: np.ndarray,
    *,
    window_starts: List[int],
    stream_start: UTCDateTime,
    t0: UTCDateTime,
    sampling_rate: float,
    p_idx: int,
    s_idx: int,
    p_threshold: float,
    s_threshold: float,
) -> Dict[str, List[float]]:
    from rapid.quality import extract_picks_simple

    p_samples: List[float] = []
    s_samples: List[float] = []
    n_win = preds.shape[0]
    for wi in range(n_win):
        w_start = window_starts[wi] if wi < len(window_starts) else 0
        for phase, idx, thr in (("P", p_idx, p_threshold), ("S", s_idx, s_threshold)):
            trace = preds[wi, :, idx]
            onsets = extract_picks_simple(trace, threshold=thr, min_separation=50)
            for o in onsets:
                abs_sample = int(w_start) + int(o)
                t = stream_start + float(abs_sample) / float(sampling_rate)
                samp = _pick_time_to_sample(t, t0)
                if samp is None:
                    continue
                if phase == "S":
                    s_samples.append(samp)
                else:
                    p_samples.append(samp)
    return {"p": p_samples, "s": s_samples}


def merge_station_picks(
    acc: Dict[str, Dict[str, List[float]]],
    station: str,
    picks: Dict[str, List[float]],
) -> None:
    if station not in acc:
        acc[station] = {"p": [], "s": []}
    acc[station]["p"].extend(picks.get("p", []))
    acc[station]["s"].extend(picks.get("s", []))


def write_picks_json(path: Path, picks_by_station: Dict[str, Dict[str, List[float]]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(picks_by_station, indent=2))


def write_orchestration_csv_tree(
    picks_dir: Path,
    picks_by_station: Dict[str, Dict[str, List[float]]],
    *,
    t0: UTCDateTime,
    timechunk_id: str = "20240101T000000Z_20240101T000100Z",
) -> None:
    """Write ``<picks_dir>/<chunk>/<station>_outputs/X_prediction_results.csv`` files."""
    chunk_dir = picks_dir / timechunk_id
    header = [
        "fname", "t0", "station", "it", "lat", "lon", "elv",
        "p_time", "p_prob", "s_time", "s_prob",
    ]
    for station, picks in picks_by_station.items():
        out_dir = chunk_dir / f"{station}_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "X_prediction_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            p_list = picks.get("p", [])
            s_list = picks.get("s", [])
            if not p_list and not s_list:
                w.writerow([timechunk_id, "", station, "", "", "", "", "", "", "", ""])
                continue
            for i, ps in enumerate(p_list):
                p_time = t0 + float(ps) / SR
                s_time_str = s_prob = ""
                if i < len(s_list):
                    st = t0 + float(s_list[i]) / SR
                    s_time_str = st.strftime("%Y-%m-%d %H:%M:%S.%f")
                    s_prob = "0.500000"
                w.writerow([
                    timechunk_id, "", station, 0, 0, 0, 0,
                    p_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "0.500000",
                    s_time_str,
                    s_prob,
                ])
            for j in range(len(p_list), len(s_list)):
                st = t0 + float(s_list[j]) / SR
                w.writerow([
                    timechunk_id, "", station, 0, 0, 0, 0,
                    "", "", st.strftime("%Y-%m-%d %H:%M:%S.%f"), "0.500000",
                ])
