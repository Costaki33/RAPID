"""JSON result recording and repeat aggregation for benchmark trials."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def write_json_result(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out.setdefault("schema_version", SCHEMA_VERSION)
    out.setdefault("recorded_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    path.write_text(json.dumps(out, indent=2, default=_json_default))


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def aggregate_repeats(
    repeats: List[Dict[str, Any]],
    *,
    timing_keys: List[str],
    memory_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Summarize N repeat dicts (min/mean/std for timing and memory fields)."""
    memory_keys = memory_keys or ["peak_ram_mb", "process_tree_ram_mb"]
    agg: Dict[str, Any] = {"n_repeats": len(repeats), "repeats": repeats}
    if not repeats:
        return agg

    for key in timing_keys:
        vals = [float(r[key]) for r in repeats if r.get(key) is not None]
        if vals:
            agg[f"{key}_min"] = min(vals)
            agg[f"{key}_mean"] = float(statistics.mean(vals))
            agg[f"{key}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0

    for key in memory_keys:
        vals = [float(r[key]) for r in repeats if r.get(key) is not None]
        if vals:
            agg[f"{key}_min"] = min(vals)
            agg[f"{key}_mean"] = float(statistics.mean(vals))
            agg[f"{key}_std"] = float(statistics.stdev(vals)) if len(vals) > 1 else 0.0

    successes = sum(1 for r in repeats if r.get("success"))
    agg["success_rate"] = successes / len(repeats)
    return agg
