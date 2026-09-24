#!/usr/bin/env python3
"""Dump quality-safe merged annotate() winners from the native config sweep.

Writes ``winners_merged_quality_safe.json`` under the native results root.
Prefers the aggregated row cache if present; otherwise walks result.json files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAPID_ROOT = Path(__file__).resolve().parents[1]
if str(RAPID_ROOT) not in sys.path:
    sys.path.insert(0, str(RAPID_ROOT))
sys.path.insert(0, str(RAPID_ROOT / "benchmarks" / "fair"))

from native_winner_orch import (  # noqa: E402
    MODELS,
    NATIVE_ROOT_DEFAULT,
    QS_THRESH,
    WINNERS_NAME,
    DTYPE_TO_METHOD,
    winner_key,
)

ROWS_CACHE = Path("/tmp/annotate_config_rows.json")


def _safe(dtype: str, p_f1_vs32) -> bool:
    if dtype == "fp32":
        return True
    return p_f1_vs32 is not None and float(p_f1_vs32) >= QS_THRESH


def winners_from_rows(rows) -> dict:
    best = {}
    for r in rows:
        if r.get("pkg") != "merged" or not r.get("ok") or r.get("timed"):
            continue
        if r.get("inf") is None:
            continue
        if not _safe(r["dtype"], r.get("p_f1_vs32")):
            continue
        k = winner_key(r["model"], r["st"], r["dev"], r["cpus"])
        cur = best.get(k)
        if cur is None or float(r["inf"]) < float(cur["inference_s_mean"]):
            best[k] = {
                "model": r["model"],
                "n_stations": int(r["st"]),
                "device": r["dev"],
                "n_cpus": int(r["cpus"]),
                "dtype": r["dtype"],
                "method": DTYPE_TO_METHOD[r["dtype"]],
                "batch_size": int(r["bs"]),
                "thread_mult": float(r["mult"]),
                "torch_threads": int(r["thr"]),
                "inference_s_mean": float(r["inf"]),
                "p_f1": r.get("p_f1"),
                "s_f1": r.get("s_f1"),
                "p_f1_vs_fp32": r.get("p_f1_vs32"),
            }
    return best


def winners_from_tree(native_root: Path) -> dict:
    from run_annotate_precision_trial import config_result_relpath, iter_config_matrix

    best = {}
    for cell in iter_config_matrix():
        if cell.get("packaging") != "merged":
            continue
        p = native_root / config_result_relpath(cell)
        if not p.is_file():
            continue
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        timing = r.get("timing") or {}
        if timing.get("timed_out") or float(timing.get("success_rate") or 0) < 1.0:
            continue
        inf = timing.get("inference_s_mean")
        if inf is None:
            continue
        dtype = (r.get("meta") or {}).get("dtype") or cell["method"].split("_")[-1]
        vs = r.get("pick_quality_vs_fp32") or {}
        p_f1_vs = vs.get("P.f1_mean")
        if not _safe(dtype, p_f1_vs):
            continue
        meta = r.get("meta") or {}
        k = winner_key(cell["model"], cell["n_stations"], cell["device"], cell["n_cpus"])
        cur = best.get(k)
        if cur is None or float(inf) < float(cur["inference_s_mean"]):
            cat = r.get("pick_quality") or {}
            best[k] = {
                "model": cell["model"],
                "n_stations": int(cell["n_stations"]),
                "device": cell["device"],
                "n_cpus": int(cell["n_cpus"]),
                "dtype": dtype,
                "method": DTYPE_TO_METHOD[dtype],
                "batch_size": int(meta.get("batch_size") or cell["batch_size"]),
                "thread_mult": float(meta.get("thread_mult") or cell["thread_mult"]),
                "torch_threads": int(meta.get("torch_threads") or cell["torch_threads"]),
                "inference_s_mean": float(inf),
                "p_f1": cat.get("P.f1_mean"),
                "s_f1": cat.get("S.f1_mean"),
                "p_f1_vs_fp32": p_f1_vs,
            }
    return best


def main() -> int:
    native_root = Path(sys.argv[1]) if len(sys.argv) > 1 else NATIVE_ROOT_DEFAULT
    if ROWS_CACHE.is_file():
        rows = json.loads(ROWS_CACHE.read_text())
        best = winners_from_rows(rows)
        source = str(ROWS_CACHE)
    else:
        sys.path.insert(0, str(RAPID_ROOT / "benchmarks" / "fair"))
        best = winners_from_tree(native_root)
        source = str(native_root)

    expected = len(MODELS) * 2 * 2 * 4
    if len(best) != expected:
        print(
            f"WARNING: expected {expected} winners, got {len(best)}",
            file=sys.stderr,
        )
    winners = [best[k] for k in sorted(best)]
    out = {
        "source": source,
        "native_root": str(native_root),
        "packaging": "merged",
        "quality_safe_rule": "fp32, or P F1 vs matched fp32 >= 0.99",
        "n_winners": len(winners),
        "winners": winners,
    }
    dest = native_root / WINNERS_NAME
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest} n={len(winners)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
