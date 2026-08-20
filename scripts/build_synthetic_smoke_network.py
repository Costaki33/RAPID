#!/usr/bin/env python3
"""Build a synthetic mSEED network for XPS smoke without waiting on full STEAD.

Smoke (Phase A) only needs valid manifests + per-station miniSEED so the runner
can exercise imports, CUDA, affinity, and one PhaseNet cell. Pilot/primary
should still use the STEAD-built networks from examples/build_seisbench_network.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from obspy import Stream, Trace, UTCDateTime

T0_ISO = "2024-01-01T00:00:00.000000Z"
TARGET_SR = 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-stations", type=int, default=250)
    ap.add_argument("--trim-samples", type=int, default=3001)
    ap.add_argument("--net-suffix", default="_w3001")
    ap.add_argument("--out-root", type=Path, default=Path("data/seisbench_networks"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    net_dir = args.out_root / f"stead_{args.n_stations}st{args.net_suffix}"
    net_dir.mkdir(parents=True, exist_ok=True)
    t0 = UTCDateTime(T0_ISO)
    stations = {}
    npts = int(args.trim_samples)
    p0 = max(50, npts // 5)
    s0 = min(npts - 50, p0 + 150)

    for i in range(args.n_stations):
        station = f"S{i:04d}"
        sta_dir = net_dir / station
        sta_dir.mkdir(parents=True, exist_ok=True)
        st = Stream()
        for comp in ("Z", "N", "E"):
            # Low-amplitude noise + simple Gaussian pulses near catalog P/S.
            wave = 1e-6 * rng.standard_normal(npts).astype(np.float64)
            t = np.arange(npts, dtype=np.float64)
            wave += 5e-5 * np.exp(-0.5 * ((t - p0) / 8.0) ** 2)
            wave += 4e-5 * np.exp(-0.5 * ((t - s0) / 12.0) ** 2)
            tr = Trace(data=wave)
            tr.stats.starttime = t0
            tr.stats.sampling_rate = TARGET_SR
            tr.stats.network = "SB"
            tr.stats.station = station
            tr.stats.channel = f"HH{comp}"
            st += tr
        st.write(str(sta_dir / f"{station}.mseed"), format="MSEED")
        stations[station] = dict(
            dataset="stead",
            source_trace_row=i,
            p_sample=p0,
            s_sample=s0,
            sampling_rate=TARGET_SR,
            npts=npts,
            t0=T0_ISO,
            synthetic_smoke=True,
        )

    window_seconds = int(np.ceil(npts / TARGET_SR))
    end = t0 + window_seconds
    meta = dict(
        dataset="stead",
        n_stations=args.n_stations,
        n_unique=args.n_stations,
        tiled=False,
        unique_stations=True,
        seed=args.seed,
        require_s=True,
        min_pick_sample=0,
        max_pick_sample=npts - 50,
        trim_samples=npts,
        t0=T0_ISO,
        start_time=t0.strftime("%Y-%m-%d %H:%M:%S"),
        end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
        timechunk_dt=window_seconds,
        target_sampling_rate=TARGET_SR,
        match_tolerance_samples=50,
        synthetic_smoke=True,
    )
    (net_dir / "manifest.json").write_text(json.dumps({"meta": meta, "stations": stations}, indent=2))
    print(f"Synthetic smoke network written: {net_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
