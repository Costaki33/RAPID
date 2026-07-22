#!/usr/bin/env python3
"""Materialize a STEAD/TXED catalog subset into a synthetic mSEED station network.

RAPID's orchestration benchmarks (Ripper, Model-Actor, Model-Actor+Slipstream)
read per-station miniSEED from ``<input_dir>/<station>/*.mseed``. This script
builds such a network from a SeisBench dataset so that:

  * the orchestration timing sweep can scale to 250 / 580 "stations", and
  * the picks produced by each station can be compared back to the dataset's
    labeled P/S arrivals (the pick-quality validation Camilo requested).

Each station is one labeled catalog trace, written at 100 Hz with a fixed,
known start time ``T0``. The catalog P/S sample indices (at 100 Hz) are stored
in ``manifest.json`` so a detected pick's absolute time can be converted back to
a sample offset and compared against the catalog. When ``--n-stations`` exceeds
the number of distinct labeled traces, traces are tiled (duplicated under new
station codes) to reach the requested network size; the manifest records the
source trace for every duplicate so pick quality is still well defined.

Example::

    python examples/build_seisbench_network.py \\
        --dataset stead --n-stations 580 \\
        --out-root data/seisbench_networks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from obspy import Stream, Trace, UTCDateTime

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rapid.seisbench_precision_eval import (  # noqa: E402
    catalog_mask,
    catalog_pick_columns,
    load_dataset,
)

# Fixed start time for every synthetic trace. Catalog sample p (at 100 Hz) maps
# to absolute time T0 + p / 100. The comparator inverts this.
T0_ISO = "2024-01-01T00:00:00.000000Z"
TARGET_SR = 100.0


def _finite_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return int(round(v))


def _station_code(i: int) -> str:
    # 5-char alnum station code (well within mSEED limits). i in [0, 580).
    return f"S{i:04d}"


def build_network(
    *,
    dataset: str,
    n_stations: int,
    out_root: Path,
    n_unique: Optional[int],
    seed: int,
    require_s: bool,
    min_pick_sample: int = 0,
    max_pick_sample: Optional[int] = None,
    trim_samples: Optional[int] = None,
    net_suffix: str = "",
) -> Path:
    """Build a synthetic station network.

    Fairness windowing controls
    ---------------------------
    ``min_pick_sample`` / ``max_pick_sample`` constrain BOTH the P and S catalog
    samples (at 100 Hz) to ``[min_pick_sample, max_pick_sample)``. This guarantees
    every station's P and S arrival lives inside the single-window regimes (e.g.
    ``max_pick_sample=2951`` keeps both picks comfortably within a 3001-sample
    window).

    ``trim_samples`` writes only the first ``trim_samples`` of each trace (used to
    materialize the 3001-sample regime-B network from the SAME stations as the
    6000-sample network, so native and orchestration feed byte-identical windows).
    The deterministic mask+seed+filters make station S0000 the same source trace
    across the trimmed and untrimmed builds.
    """
    ds = load_dataset(dataset)
    p_col, s_col = catalog_pick_columns(ds)
    mask = catalog_mask(ds, require_s)
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        raise SystemExit(f"No catalog traces with valid picks in {dataset}.")

    def _pick_in_range(v: Optional[int]) -> bool:
        if v is None:
            return False
        if v < min_pick_sample:
            return False
        if max_pick_sample is not None and v >= max_pick_sample:
            return False
        return True

    # Fairness requirement: every station must be a DISTINCT labeled trace with
    # BOTH P and S picks (no tiling/duplication). We oversample the qualifying
    # candidate pool in a deterministic shuffled order and extract until we have
    # exactly ``n_stations`` unique valid payloads, failing loudly if a dataset
    # cannot supply that many. ``n_unique`` is kept for CLI compatibility but is
    # forced to ``n_stations`` here so the network has no duplicates.
    if not require_s:
        print(
            "[build] WARNING: require_s is False; for the fairness benchmark every "
            "station should have both P and S. Pass --require-s.",
        )
    rng = np.random.default_rng(seed)
    target_unique = n_stations if n_unique is None else min(n_unique, n_stations)
    if target_unique < n_stations:
        raise SystemExit(
            f"n-unique ({target_unique}) < n-stations ({n_stations}); tiling is "
            "disabled for the fairness benchmark. Increase --n-unique or omit it."
        )
    shuffled = idxs.copy()
    rng.shuffle(shuffled)

    net_dir = out_root / f"{dataset.lower()}_{n_stations}st{net_suffix}"
    net_dir.mkdir(parents=True, exist_ok=True)

    t0 = UTCDateTime(T0_ISO)
    manifest: Dict[str, Dict] = {}
    # Extract distinct trace payloads (one per station) until we reach n_stations.
    unique_payloads: List[Dict] = []
    scanned = 0
    for trace_row in shuffled:
        if len(unique_payloads) >= n_stations:
            break
        scanned += 1
        trace_row = int(trace_row)
        try:
            waves, meta = ds.get_sample(trace_row, sampling_rate=TARGET_SR)
        except Exception:
            continue
        if waves.ndim != 2:
            continue
        p_cat = _finite_int(meta.get(p_col))
        if p_cat is None or not (0 <= p_cat < waves.shape[1]):
            continue
        s_cat = _finite_int(meta.get(s_col)) if s_col else None
        if s_cat is not None and not (0 <= s_cat < waves.shape[1]):
            s_cat = None
        if require_s and s_cat is None:
            continue
        # Fairness: keep BOTH picks inside the single-window regimes.
        if not _pick_in_range(p_cat):
            continue
        if require_s and not _pick_in_range(s_cat):
            continue
        co = str(meta.get("trace_component_order") or "ZNE")
        unique_payloads.append(
            dict(
                trace_row=trace_row,
                waves=np.asarray(waves, dtype=np.float64),
                component_order=co,
                p_sample=p_cat,
                s_sample=s_cat,
                npts=int(waves.shape[1]),
            )
        )

    if len(unique_payloads) < n_stations:
        raise SystemExit(
            f"{dataset}: only found {len(unique_payloads)} usable unique traces "
            f"(needed {n_stations}) after scanning {scanned} qualifying candidates. "
            "Cannot build a no-tiling network of this size."
        )

    max_npts = 0
    for i in range(n_stations):
        payload = unique_payloads[i]
        station = _station_code(i)
        sta_dir = net_dir / station
        sta_dir.mkdir(parents=True, exist_ok=True)

        st = Stream()
        waves = payload["waves"]
        if trim_samples is not None:
            waves = waves[:, :trim_samples]
        npts_written = int(waves.shape[1])
        for ci, comp in enumerate(payload["component_order"]):
            if ci >= waves.shape[0]:
                break
            tr = Trace(data=np.asarray(waves[ci], dtype=np.float64))
            tr.stats.starttime = t0
            tr.stats.sampling_rate = TARGET_SR
            tr.stats.network = "SB"
            tr.stats.station = station
            tr.stats.channel = f"HH{comp}"
            st += tr
        st.write(str(sta_dir / f"{station}.mseed"), format="MSEED")
        max_npts = max(max_npts, npts_written)

        manifest[station] = dict(
            dataset=dataset.lower(),
            source_trace_row=payload["trace_row"],
            p_sample=payload["p_sample"],
            s_sample=payload["s_sample"],
            sampling_rate=TARGET_SR,
            npts=npts_written,
            t0=T0_ISO,
        )

    window_seconds = int(np.ceil(max_npts / TARGET_SR))
    end = t0 + window_seconds
    manifest_meta = dict(
        dataset=dataset.lower(),
        n_stations=n_stations,
        n_unique=len(unique_payloads),
        tiled=False,
        unique_stations=True,
        seed=seed,
        require_s=require_s,
        min_pick_sample=min_pick_sample,
        max_pick_sample=max_pick_sample,
        trim_samples=trim_samples,
        t0=T0_ISO,
        start_time=t0.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        timechunk_dt=window_seconds,
        target_sampling_rate=TARGET_SR,
        match_tolerance_samples=50,
    )
    out = dict(meta=manifest_meta, stations=manifest)
    (net_dir / "manifest.json").write_text(json.dumps(out, indent=2))

    print(f"Network written: {net_dir}")
    print(f"  stations={n_stations}  unique catalog traces={len(unique_payloads)}")
    print(f"  start_time='{manifest_meta['start_time']}'  end_time='{manifest_meta['end_time']}'  timechunk_dt={window_seconds}")
    print(f"  manifest: {net_dir / 'manifest.json'}")
    return net_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["stead", "txed"])
    ap.add_argument("--n-stations", type=int, required=True)
    ap.add_argument("--out-root", type=Path, default=_ROOT / "data" / "seisbench_networks")
    ap.add_argument(
        "--n-unique",
        type=int,
        default=None,
        help="Distinct catalog traces to sample before tiling (default min(n_stations, available)).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--require-s",
        action="store_true",
        help="Only use traces that have both a P and an S catalog pick.",
    )
    ap.add_argument("--min-pick-sample", type=int, default=0,
                    help="Both P and S catalog samples must be >= this (100 Hz).")
    ap.add_argument("--max-pick-sample", type=int, default=None,
                    help="Both P and S catalog samples must be < this (100 Hz). "
                         "Use 2951 to keep both picks inside a 3001-sample window.")
    ap.add_argument("--trim-samples", type=int, default=None,
                    help="Write only the first N samples of each trace (e.g. 3001 for regime-B).")
    ap.add_argument("--net-suffix", default="",
                    help="Suffix appended to the network dir name (e.g. _w3001).")
    args = ap.parse_args()

    build_network(
        dataset=args.dataset,
        n_stations=args.n_stations,
        out_root=args.out_root,
        n_unique=args.n_unique,
        seed=args.seed,
        require_s=args.require_s,
        min_pick_sample=args.min_pick_sample,
        max_pick_sample=args.max_pick_sample,
        trim_samples=args.trim_samples,
        net_suffix=args.net_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
