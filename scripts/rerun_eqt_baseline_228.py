"""Re-run a single matrix cell and patch ``results/matrix.jsonl`` in place.

Default target: EQTransformer / original, ``kind=baseline``, ``n_stations=228``,
``device=cuda:0`` (the historical ~180s outlier).

Usage (from ``RAPID/``)::

    export PYTHONPATH=\"$PWD:$PWD/..:$PYTHONPATH\"
    python scripts/rerun_eqt_baseline_228.py --dry-run   # only show match counts
    python scripts/rerun_eqt_baseline_228.py            # rerun + patch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def _match_row(r: dict, *, parent: str, child: str, n_stations: int, device: str) -> bool:
    return (
        r.get("kind") == "baseline"
        and r.get("backend") == "baseline_annotate"
        and r.get("model_parent") == parent
        and r.get("model_child") == child
        and r.get("n_stations") == n_stations
        and r.get("device") == device
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/full_matrix.json")
    ap.add_argument("--jsonl", default="results/matrix.jsonl")
    ap.add_argument("--parent", default="EQTransformer")
    ap.add_argument("--child", default="original")
    ap.add_argument("--label", default="EQTransformer")
    ap.add_argument("--n-stations", type=int, default=228)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from rapid.data import load_all_streams, select_stations
    from rapid.matrix import MatrixConfig, _bench_baseline

    cfg = MatrixConfig.from_dict(
        json.loads(Path(args.config).read_text(encoding="utf-8"))
    )
    jsonl_path = _HERE / args.jsonl
    if not jsonl_path.is_file():
        print(f"ERROR: {jsonl_path} not found", file=sys.stderr)
        return 1

    original_text = jsonl_path.read_text(encoding="utf-8")
    lines = original_text.splitlines()
    match_idx: list[int] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _match_row(
            r,
            parent=args.parent,
            child=args.child,
            n_stations=args.n_stations,
            device=args.device,
        ):
            match_idx.append(i)

    print(f"Matched {len(match_idx)} JSONL lines to replace (indices {match_idx[:5]}{'...' if len(match_idx) > 5 else ''})")
    if args.dry_run:
        return 0
    if len(match_idx) != cfg.repeats:
        print(
            f"WARNING: expected {cfg.repeats} baseline repeats, found {len(match_idx)}; aborting patch.",
            file=sys.stderr,
        )
        return 1

    stations = select_stations(cfg.dataset_dir, args.n_stations)
    streams = load_all_streams(cfg.dataset_dir, stations)

    def _streams_fn():
        return streams

    backend_cfg = {"name": "baseline_annotate", "dtype": "fp32"}
    new_rows, _ = _bench_baseline(
        backend_cfg,
        args.parent,
        args.child,
        args.label,
        _streams_fn,
        args.device,
        cfg,
        dry=False,
        completed=set(),
        n_stations=args.n_stations,
    )
    if len(new_rows) != cfg.repeats:
        print(f"ERROR: bench returned {len(new_rows)} rows", file=sys.stderr)
        return 1

    by_repeat = {r["repeat"]: r for r in new_rows}
    for i in match_idx:
        old = json.loads(lines[i])
        rep = old.get("repeat")
        if rep not in by_repeat:
            print(f"ERROR: line {i} repeat={rep} not in new rows", file=sys.stderr)
            return 1
        lines[i] = json.dumps(by_repeat[rep], default=str)

    backup = jsonl_path.with_suffix(jsonl_path.suffix + ".bak_rerun228")
    backup.write_text(original_text, encoding="utf-8")
    jsonl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"Backup: {backup}")
    print(f"Patched {jsonl_path} with {len(new_rows)} fresh baseline rows.")
    for r in new_rows:
        print(
            f"  repeat={r['repeat']} wall_time_s={r.get('wall_time_s')} "
            f"end_to_end_wall_s={r.get('end_to_end_wall_s')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
