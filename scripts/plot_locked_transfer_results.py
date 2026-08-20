#!/usr/bin/env python3
"""Build tables + interactive HTML canvas for locked-recipe transfer results."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path.home() / "RAPID" / "results" / "locked_recipe_transfer" / "BEGE-TEXA75535L_2026-08-19"
OUT_DIR = ROOT / "figures"
OUT_CSV = ROOT / "transfer_summary.csv"
OUT_HTML = OUT_DIR / "transfer_canvas.html"


def _pq_f1(d: dict) -> tuple[float | None, float | None]:
    """Extract catalog P/S F1 from pick_quality_vs_catalog.

    RAPID result.json stores flat means as ``P.f1_mean`` / ``S.f1_mean``,
    and per-repeat blocks under ``repeats[].P|S.f1``. Older shapes may put
    ``P`` / ``S`` dicts at the top level.
    """
    pq = d.get("pick_quality_vs_catalog") or {}
    if not isinstance(pq, dict):
        return None, None

    p_f1 = pq.get("P.f1_mean")
    s_f1 = pq.get("S.f1_mean")

    if isinstance(pq.get("P"), dict) and p_f1 is None:
        p_f1 = pq["P"].get("f1")
    if isinstance(pq.get("S"), dict) and s_f1 is None:
        s_f1 = pq["S"].get("f1")

    repeats = pq.get("repeats")
    if isinstance(repeats, list) and repeats:
        if p_f1 is None:
            vals = [
                r["P"]["f1"]
                for r in repeats
                if isinstance(r, dict) and isinstance(r.get("P"), dict) and r["P"].get("f1") is not None
            ]
            if vals:
                p_f1 = sum(vals) / len(vals)
        if s_f1 is None:
            vals = [
                r["S"]["f1"]
                for r in repeats
                if isinstance(r, dict) and isinstance(r.get("S"), dict) and r["S"].get("f1") is not None
            ]
            if vals:
                s_f1 = sum(vals) / len(vals)

    # alternate key shapes
    for k, v in pq.items():
        if not isinstance(v, dict):
            continue
        kl = str(k).lower()
        if p_f1 is None and ("p_f1" in kl or kl in {"p", "phase_p"}):
            p_f1 = v.get("f1")
        if s_f1 is None and ("s_f1" in kl or kl in {"s", "phase_s"}):
            s_f1 = v.get("f1")

    def _f(x):
        try:
            return None if x is None else float(x)
        except (TypeError, ValueError):
            return None

    return _f(p_f1), _f(s_f1)


def collect_rows() -> pd.DataFrame:
    rows = []
    seen_rel: set[str] = set()
    for p in sorted(ROOT.rglob("result.json")):
        # Skip nested handoff mirrors / accidental copies under the run root
        if "handoff" in p.parts:
            continue
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        # Prefer canonical trees only
        if not (rel.startswith("annotate_bf16/") or rel.startswith("ma/")):
            continue
        d = json.loads(p.read_text())
        t = d.get("timing") or {}
        sr = float(t.get("success_rate") or 0.0)
        if sr < 1.0:
            continue
        parts = p.parts
        if "stead" not in parts:
            continue
        i = parts.index("stead")
        n_stations = int(str(parts[i + 1]).replace("st", ""))
        model = parts[i + 2]
        device = parts[i + 3]

        # Dedup if the same cell appears twice under annotate vs ma layout quirks
        dedup_key = rel
        if dedup_key in seen_rel:
            continue
        seen_rel.add(dedup_key)

        if "ma/" in rel or "/ma/" in p.as_posix():
            layer = "playback" if "/playback/" in p.as_posix() else "staggered"
            k_tok = next(x for x in parts if x.startswith("kma"))
            k = int(k_tok.split("_")[0].replace("kma", ""))
            # n_cpus is not in path; infer from K + caps for display
            # Prefer meta if present
            meta = d.get("meta") or {}
            n_cpus = meta.get("n_cpus") or meta.get("n_cpus_requested")
            if n_cpus is None:
                # reconstruct: orch cells with this K were run at the smallest
                # core budget that yields this K under the caps used.
                n_cpus = k  # affinity width matched k on first run
            orch = d.get("orch") or {}
            lat = ((d.get("latency") or {}).get("pooled_across_repeats") or {}).get(
                "e2e_finish_minus_ready"
            ) or {}
            if layer == "playback":
                runtime = orch.get("makespan_s_mean")
                metric = "makespan_s"
            else:
                runtime = lat.get("p95")
                metric = "p95_finish_minus_ready_s"
        else:
            layer = "native"
            cpus_tok = next(x for x in parts if x.startswith("cpus"))
            n_cpus = int(cpus_tok.replace("cpus", ""))
            k = None  # native has no actors
            runtime = t.get("inference_s_mean")
            metric = "inference_s"

        p_f1, s_f1 = _pq_f1(d)
        rows.append(
            dict(
                method=layer,
                model=model,
                n_stations=n_stations,
                device=device,
                n_gpus=1 if device == "gpu" else 0,
                n_cpus=int(n_cpus) if n_cpus is not None else None,
                K=k,
                runtime_s=None if runtime is None else float(runtime),
                metric=metric,
                success_rate=sr,
                path=rel,
                p_f1=p_f1,
                s_f1=s_f1,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = collect_rows()
    if df.empty:
        raise SystemExit(f"No successful result.json under {ROOT}")

    df = df.sort_values(["method", "model", "n_stations", "device", "n_cpus", "K"])
    df.to_csv(OUT_CSV, index=False)

    # Console breakdown
    print(f"rows={len(df)}  written {OUT_CSV}")
    print("\n=== counts by method × device ===")
    print(df.groupby(["method", "device"]).size().to_string())
    print("\n=== runtime summary (s) ===")
    show = df.copy()
    show["label"] = show.apply(
        lambda r: f"K={r['K']}" if pd.notna(r["K"]) else f"cpus={r['n_cpus']}",
        axis=1,
    )
    for method in ["native", "playback", "staggered"]:
        sub = df[df["method"] == method]
        print(f"\n--- {method} ({sub['metric'].iloc[0] if len(sub) else ''}) ---")
        cols = ["model", "n_stations", "device", "n_cpus", "K", "runtime_s"]
        print(sub[cols].to_string(index=False))

    # Interactive canvas
    df_plot = df.copy()
    df_plot["K_display"] = df_plot["K"].apply(lambda x: "native" if pd.isna(x) else str(int(x)))
    df_plot["config"] = df_plot.apply(
        lambda r: (
            f"{r['model']} | {r['n_stations']}st | {r['device']} | "
            f"cpus={r['n_cpus']}" + (f" | K={int(r['K'])}" if pd.notna(r["K"]) else "")
        ),
        axis=1,
    )
    df_plot["facet"] = df_plot["method"] + " · " + df_plot["metric"]

    fig = px.bar(
        df_plot,
        x="runtime_s",
        y="config",
        color="device",
        facet_col="method",
        facet_col_wrap=1,
        orientation="h",
        hover_data={
            "model": True,
            "n_stations": True,
            "n_cpus": True,
            "n_gpus": True,
            "K": True,
            "runtime_s": ":.3f",
            "metric": True,
            "p_f1": ":.4f",
            "s_f1": ":.4f",
            "config": False,
        },
        title="Locked-recipe transfer — BEGE-TEXA75535L (runtime by method / model / CPUs / GPU / K)",
        color_discrete_map={"cpu": "#2a6f97", "gpu": "#c1121f"},
        height=max(900, 28 * len(df_plot)),
    )
    fig.update_layout(
        template="plotly_white",
        font=dict(family="IBM Plex Sans, Segoe UI, sans-serif", size=12),
        margin=dict(l=40, r=40, t=80, b=40),
        legend_title_text="device",
        bargap=0.15,
    )
    fig.update_yaxes(automargin=True, title="")
    fig.update_xaxes(title="runtime (s)")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Second canvas: grouped by model for 580st highlight cells
    key = df_plot[df_plot["n_stations"] == 580].copy()
    fig2 = px.scatter(
        key,
        x="runtime_s",
        y="model",
        color="device",
        symbol="method",
        size="n_cpus",
        hover_data=["n_cpus", "n_gpus", "K", "metric", "runtime_s"],
        title="580-station slice — method × model (marker size = n_cpus)",
        color_discrete_map={"cpu": "#2a6f97", "gpu": "#c1121f"},
        height=520,
    )
    fig2.update_layout(template="plotly_white", font=dict(family="IBM Plex Sans, Segoe UI, sans-serif"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Locked-recipe transfer canvas</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --ink: #1b1b1b;
      --muted: #5a564c;
      --card: #fffdf8;
    }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 600px at 10% -10%, #e7f0f5 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #f7e8d8 0%, transparent 50%),
        var(--bg);
      color: var(--ink);
    }}
    header {{
      padding: 28px 32px 8px;
      max-width: 1400px;
      margin: 0 auto;
    }}
    h1 {{
      font-family: "Fraunces", Georgia, serif;
      font-weight: 600;
      font-size: 2rem;
      margin: 0 0 8px;
    }}
    p {{ color: var(--muted); max-width: 70ch; line-height: 1.45; }}
    .note {{
      display: inline-block;
      background: var(--card);
      border: 1px solid #ddd4c2;
      border-radius: 10px;
      padding: 10px 14px;
      margin-top: 8px;
      font-size: 0.95rem;
    }}
    section {{ max-width: 1400px; margin: 0 auto; padding: 8px 16px 32px; }}
  </style>
</head>
<body>
  <header>
    <h1>Locked-recipe transfer</h1>
    <p>
      Host BEGE-TEXA75535L · RTX 4050 Laptop 6&nbsp;GB · WSL 24&nbsp;GB ·
      first matrix <code>CPU_K_CAP=10</code>, then uncapped CPU&nbsp;K=20 follow-up ·
      <code>GPU_K_CAP=4</code> (PhaseNet@250 → K=2). Native metric = inference_s;
      playback = makespan_s; staggered = p95 finish−ready.
      Catalog P/S F1 in CSV hover via source table.
    </p>
    <div class="note">Rows: {len(df)} successful cells · CSV: {OUT_CSV.name} · p_f1/s_f1 filled from catalog</div>
  </header>
  <section>{fig.to_html(full_html=False, include_plotlyjs="cdn")}</section>
  <section>{fig2.to_html(full_html=False, include_plotlyjs=False)}</section>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\ncanvas: {OUT_HTML}")


if __name__ == "__main__":
    main()
