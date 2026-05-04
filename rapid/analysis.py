"""Analysis helpers for the matrix JSONL.

Once the sweep finishes (or even while it's running), load ``results/matrix.jsonl``
into a ``pandas.DataFrame`` via :func:`load_results`, then use the aggregation
helpers for the common questions:

- What's the wall-time speedup vs baseline for each (model, n_stations)?
- Where's the knee on the CPU-worker sweep?
- Does ``torch.compile`` help once the preprocess bottleneck is removed?
- How much GPU memory do the fast paths actually use?

Every helper keeps intermediate data in a DataFrame so you can chain additional
pandas filters / groupbys for ad-hoc analysis. ``env`` rows are dropped on
load; ``error`` rows are kept (with a ``is_error`` boolean) so you can audit
failures without them contaminating numerical aggregates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:  # pandas is optional; analysis helpers require it
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_results(
    path: str | Path,
    *,
    include_errors: bool = True,
    drop_env: bool = True,
) -> "pd.DataFrame":
    """Load a matrix JSONL into a DataFrame.

    One row per benchmark trial. ``stage_times_s`` is expanded into flat
    ``stage_<name>_s`` columns for easy group-by / plotting.
    """
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = r.get("kind")
        if drop_env and kind == "env":
            continue
        if not include_errors and kind == "error":
            continue
        rows.append(r)

    df = pd.DataFrame(rows)

    # Expand stage_times_s dict into columns.
    if "stage_times_s" in df.columns:
        stages = df["stage_times_s"].dropna().apply(lambda d: d if isinstance(d, dict) else {})
        stage_df = pd.json_normalize(stages).add_prefix("stage_").add_suffix("_s")
        stage_df.index = stages.index
        df = df.drop(columns=["stage_times_s"]).join(stage_df)

    # Canonical wall_time column: prefer wall_time_s, then total_s.
    if "wall_time_s" not in df.columns:
        df["wall_time_s"] = df.get("total_s")
    else:
        df["wall_time_s"] = df["wall_time_s"].fillna(df.get("total_s"))

    df["is_error"] = df.get("kind") == "error"
    if "benchmark_status" in df.columns:
        df["is_skipped_incompatible"] = df["benchmark_status"] == "skipped_incompatible"
    else:
        df["is_skipped_incompatible"] = False

    # Canonicalize a flat "variant" label that's nice for plotting legends.
    # Distinct method families end up visible in the sweep — the label
    # below makes each one identifiable by string alone:
    #
    #   baseline/fp32                          -> baseline 1-device
    #   baseline_annotate/fp32/2gpu_baseline   -> baseline 2-GPU fair
    #   lean_pytorch/<dt>                      -> lean 1-GPU serial
    #   lean_pytorch/<dt>/cpu<N>               -> lean 1-GPU pooled (on GPU)
    #   lean_pytorch/<dt>/2gpu_serial          -> lean 2-GPU serial
    #   lean_pytorch/<dt>/2gpu_cpu<N>          -> lean 2-GPU pipelined
    #   lean_pytorch/<dt>/cpu_infer_pool<N>[_t<T>]
    #                                          -> lean CPU-inference pooled
    #                                             (``cpu_worker_sweep`` with
    #                                             ``device="cpu"`` and
    #                                             optional explicit thread
    #                                             count ``T``)
    def _safe_int(x) -> Optional[int]:
        """Cast to int, returning None for None/NaN/invalid inputs.

        Pandas stores missing values in numeric columns as NaN, and
        ``int(float('nan'))`` raises ValueError — so every int cast in a
        row-wise apply must go through here.
        """
        try:
            if x is None:
                return None
            xf = float(x)
            if xf != xf:  # NaN check
                return None
            return int(xf)
        except (TypeError, ValueError):
            return None

    def _variant(row: pd.Series) -> str:
        parts = [str(row.get("backend", "")), str(row.get("dtype", ""))]
        extra = row.get("backend_extra") or {}
        if isinstance(extra, dict) and extra.get("compile"):
            parts.append("compile")
        kind = row.get("kind")
        n_cpu_g = _safe_int(row.get("n_cpu_workers_per_gpu"))
        if kind == "dual_gpu":
            if row.get("backend") == "baseline_annotate":
                parts.append("2gpu_baseline")
            elif n_cpu_g is not None and n_cpu_g > 0:
                parts.append(f"2gpu_cpu{n_cpu_g}")
            else:
                # Legacy serial-preprocess dual_gpu rows (pre-split).
                parts.append("2gpu_serial")
        elif kind == "dual_gpu_serial":
            parts.append("2gpu_serial")
        elif kind == "cpu_worker_sweep":
            n_cpu = _safe_int(row.get("n_cpu_workers")) or 0
            device = str(row.get("device") or "")
            if device.startswith("cpu"):
                # CPU-inference variant. Include the thread count only when
                # it was explicitly pinned (>= 0); -1 means "runner auto-picked".
                t = _safe_int(row.get("infer_num_threads"))
                if t is not None and t > 0:
                    parts.append(f"cpu_infer_pool{n_cpu}_t{t}")
                else:
                    parts.append(f"cpu_infer_pool{n_cpu}")
            else:
                parts.append(f"cpu{n_cpu}")
        return "/".join(p for p in parts if p)

    df["variant"] = df.apply(_variant, axis=1)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


GROUP_AXES_DEFAULT: Tuple[str, ...] = (
    "kind", "variant", "model_label", "device",
    "n_stations", "batch_size", "overlap_samples",
    "n_cpu_workers", "n_cpu_workers_per_gpu", "infer_num_threads",
)


def aggregate(
    df: "pd.DataFrame",
    *,
    axes: Iterable[str] = GROUP_AXES_DEFAULT,
    metrics: Iterable[str] = (
        "wall_time_s",
        "throughput_stations_per_s",
        "throughput_windows_per_s",
        "throughput_samples_per_s",
        "gpu_forward_s",
        "gpu_idle_s",
        "preprocess_total_s",
        "gpu_utilization_pct",
        "peak_gpu_mem_bytes",
    ),
) -> "pd.DataFrame":
    """Collapse repeats: median / min / max / std over each (axes) group.

    Non-existent axes and metrics are silently skipped, so the same call works
    whether or not the sweep included CPU-worker or dual-GPU rows.
    """
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    axes = [a for a in axes if a in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    work = df[~df["is_error"]].copy()
    if not axes or not metrics:
        return work
    agg_map = {m: ["median", "min", "max", "std", "count"] for m in metrics}
    g = work.groupby(list(axes), dropna=False).agg(agg_map)
    g.columns = [f"{m}_{stat}" for m, stat in g.columns]
    return g.reset_index()


def summarize_with_ci(
    df: "pd.DataFrame",
    *,
    axes: Iterable[str] = GROUP_AXES_DEFAULT,
    metrics: Iterable[str] = (
        "wall_time_s",
        "end_to_end_wall_s",
        "throughput_stations_per_s",
        "throughput_windows_per_s",
    ),
    ci: float = 0.95,
) -> "pd.DataFrame":
    """Collapse repeats with mean ± half-width confidence intervals.

    For each ``(axes)`` group, emits columns::

        <metric>_mean, <metric>_std, <metric>_n,
        <metric>_ci_low, <metric>_ci_high, <metric>_ci_half

    ``ci_half`` is ``t_{n-1, 1-alpha/2} * std / sqrt(n)`` — so with three
    repeats (the matrix default) and the 95 % CI, the width is about
    ``2.48 * std / sqrt(3)``. Rows with ``n <= 1`` get ``NaN`` for the
    CI columns since the t-distribution isn't defined. Useful for
    answering "is this 5 % speedup actually real, or is it within the
    repeat-to-repeat noise?".
    """
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    try:
        from scipy.stats import t as _student_t
        _have_scipy = True
    except ImportError:
        _have_scipy = False

    import math

    axes = [a for a in axes if a in df.columns]
    metrics = [m for m in metrics if m in df.columns]
    work = df[~df["is_error"]].copy()
    if not axes or not metrics:
        return work

    rows: List[Dict[str, Any]] = []
    for key, grp in work.groupby(list(axes), dropna=False):
        rec: Dict[str, Any] = dict(zip(axes, key if isinstance(key, tuple) else (key,)))
        for m in metrics:
            vals = grp[m].dropna().astype(float).to_numpy()
            n = int(vals.size)
            rec[f"{m}_n"] = n
            if n == 0:
                rec[f"{m}_mean"] = float("nan")
                rec[f"{m}_std"] = float("nan")
                rec[f"{m}_ci_low"] = float("nan")
                rec[f"{m}_ci_high"] = float("nan")
                rec[f"{m}_ci_half"] = float("nan")
                continue
            mean = float(vals.mean())
            rec[f"{m}_mean"] = mean
            if n <= 1:
                rec[f"{m}_std"] = float("nan")
                rec[f"{m}_ci_low"] = mean
                rec[f"{m}_ci_high"] = mean
                rec[f"{m}_ci_half"] = float("nan")
                continue
            std = float(vals.std(ddof=1))
            rec[f"{m}_std"] = std
            # t critical value with n-1 dof. Fall back to the normal z
            # (1.96 for 95 %) when scipy isn't installed — good enough
            # for n >= 10; conservative-ish at n=3.
            if _have_scipy:
                t_crit = float(_student_t.ppf(1.0 - (1.0 - ci) / 2.0, df=n - 1))
            else:
                t_crit = 1.96 if abs(ci - 0.95) < 1e-6 else 2.0
            half = t_crit * std / math.sqrt(n)
            rec[f"{m}_ci_half"] = half
            rec[f"{m}_ci_low"] = mean - half
            rec[f"{m}_ci_high"] = mean + half
        rows.append(rec)
    return pd.DataFrame(rows)


def significant_speedup(
    df: "pd.DataFrame",
    *,
    baseline_kind: str = "baseline",
    metric: str = "wall_time_s",
    match_on: Iterable[str] = ("model_label", "n_stations", "device"),
    ci: float = 0.95,
) -> "pd.DataFrame":
    """Speedup ratios tagged ``is_significant`` when CIs don't overlap.

    For each ``(variant, match_on)`` cell, joins against the baseline's
    mean & CI for the same cell. Emits::

        speedup_mean         mean(baseline_metric) / mean(variant_metric)
        baseline_mean
        variant_mean
        speedup_ci_low       lower bound using CI endpoints
        speedup_ci_high      upper bound using CI endpoints
        is_significant       True iff the variant's metric CI is fully
                             below the baseline's metric CI (for a
                             speedup) or fully above it (for a slowdown)

    A 5 % nominal speedup with ``is_significant=False`` means you don't
    actually have a speedup yet — noise can explain it.
    """
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    summary = summarize_with_ci(
        df, axes=GROUP_AXES_DEFAULT, metrics=[metric], ci=ci,
    )
    match_on = list(match_on)
    base = summary[summary["kind"] == baseline_kind][
        match_on + [f"{metric}_mean", f"{metric}_ci_low", f"{metric}_ci_high"]
    ].copy()
    base = base.rename(columns={
        f"{metric}_mean": "baseline_mean",
        f"{metric}_ci_low": "baseline_ci_low",
        f"{metric}_ci_high": "baseline_ci_high",
    })
    variants = summary[summary["kind"] != baseline_kind].copy()
    merged = variants.merge(base, on=match_on, how="left")
    merged = merged.rename(columns={f"{metric}_mean": "variant_mean"})
    merged["speedup_mean"] = merged["baseline_mean"] / merged["variant_mean"]
    # Best-case speedup: fastest plausible variant vs slowest plausible baseline.
    merged["speedup_ci_high"] = (
        merged["baseline_ci_high"] / merged[f"{metric}_ci_low"]
    )
    # Worst-case: slowest plausible variant vs fastest plausible baseline.
    merged["speedup_ci_low"] = (
        merged["baseline_ci_low"] / merged[f"{metric}_ci_high"]
    )
    # "Significant speedup" iff variant upper CI < baseline lower CI,
    # i.e. no overlap and variant is clearly faster.
    merged["is_significant"] = (
        merged[f"{metric}_ci_high"] < merged["baseline_ci_low"]
    )
    return merged


# ---------------------------------------------------------------------------
# Speedup tables
# ---------------------------------------------------------------------------


def speedup_vs_baseline(
    df: "pd.DataFrame",
    *,
    baseline_kind: str = "baseline",
    baseline_variant: Optional[str] = None,
    match_on: Iterable[str] = ("model_label", "n_stations", "device"),
    metric: str = "wall_time_s",
) -> "pd.DataFrame":
    """Compare every non-baseline variant against the appropriate baseline
    and emit a ``speedup_{median,best}`` column.

    Two baseline families are recognized so dual-GPU rows actually get a
    speedup number (they previously always got NaN because their device is
    ``cuda:0+cuda:1`` and no 1-GPU baseline exists there):

    - **Single-device baseline** – ``kind == baseline_kind`` (typically
      ``"baseline"``, i.e. 1-GPU/CPU ``baseline_annotate``). Used for every
      non-baseline row whose ``device`` does not contain a ``+``.
    - **Dual-device baseline** – ``kind == "dual_gpu"`` with
      ``backend == "baseline_annotate"`` (the fair 2-GPU ``annotate()``
      comparison). Used for every non-baseline row on a ``cuda:a+cuda:b``
      device.

    Match keys (``model_label``, ``n_stations``, ``device``) are shared
    across both families, so the merge just lines up each row with whatever
    baseline exists on the same device string.
    """
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    work = df[~df["is_error"]].copy()
    match_on = list(match_on)

    # Single-device baseline (1-GPU / CPU baseline_annotate).
    single_base_mask = work["kind"] == baseline_kind
    if baseline_variant is not None:
        single_base_mask &= work["variant"] == baseline_variant

    # Dual-device baseline (the 2-GPU baseline_annotate fair comparison).
    # Lives under kind="dual_gpu" with backend="baseline_annotate" and
    # device="cuda:0+cuda:1".
    dual_base_mask = (
        (work["kind"] == "dual_gpu")
        & (work.get("backend") == "baseline_annotate")
    )

    base_mask = single_base_mask | dual_base_mask
    base = (
        work[base_mask]
        .groupby(match_on, dropna=False)[metric]
        .median()
        .rename(f"{metric}_baseline")
        .reset_index()
    )

    other = (
        work[~base_mask]
        .groupby(match_on + ["kind", "variant"], dropna=False)[metric]
        .agg(["median", "min", "count"])
        .reset_index()
        .rename(columns={"median": f"{metric}_median", "min": f"{metric}_min",
                         "count": "n_repeats"})
    )

    out = other.merge(base, on=match_on, how="left")
    out["speedup_median"] = out[f"{metric}_baseline"] / out[f"{metric}_median"]
    out["speedup_best"] = out[f"{metric}_baseline"] / out[f"{metric}_min"]
    return out.sort_values(["model_label", "n_stations", "speedup_median"],
                           ascending=[True, True, False])


def best_variant_per_group(
    df: "pd.DataFrame",
    *,
    match_on: Iterable[str] = ("model_label", "n_stations", "device"),
    metric: str = "wall_time_s",
) -> "pd.DataFrame":
    """Return the single fastest variant per group, by median ``metric``."""
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    work = df[~df["is_error"]].copy()
    match_on = list(match_on)
    g = (
        work.groupby(match_on + ["kind", "variant"], dropna=False)[metric]
        .median()
        .reset_index()
        .sort_values(metric)
    )
    return g.groupby(match_on, dropna=False).first().reset_index()


# ---------------------------------------------------------------------------
# CPU-worker sweep knee + GPU utilization summary
# ---------------------------------------------------------------------------


def cpu_worker_knee(
    df: "pd.DataFrame",
    *,
    model_label: Optional[str] = None,
    n_stations: Optional[int] = None,
) -> "pd.DataFrame":
    """Per (model, n_stations, batch_size, variant), emit the best ``n_cpu_workers``.

    Useful for answering "how many CPUs should I use on this box?".
    """
    if pd is None:
        raise ImportError("pandas is required for analysis helpers")
    work = df[(df["kind"] == "cpu_worker_sweep") & ~df["is_error"]].copy()
    if model_label is not None:
        work = work[work["model_label"] == model_label]
    if n_stations is not None:
        work = work[work["n_stations"] == n_stations]
    g = (
        work.groupby(
            ["model_label", "n_stations", "batch_size", "variant"], dropna=False
        )
        .apply(
            lambda d: d.loc[d["wall_time_s"].idxmin()][
                ["n_cpu_workers", "wall_time_s", "gpu_utilization_pct"]
            ]
        )
        .reset_index()
    )
    return g


def env_rows(path: str | Path) -> List[Dict[str, Any]]:
    """Return every ``kind=="env"`` row from a JSONL (ordered as written)."""
    out: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "env":
            out.append(r)
    return out
