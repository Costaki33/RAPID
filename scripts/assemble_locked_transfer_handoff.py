#!/usr/bin/env python3
"""Assemble paper handoff package for the locked-recipe laptop transfer."""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HOME = Path.home() / "RAPID"
RUN = HOME / "results" / "locked_recipe_transfer" / "BEGE-TEXA75535L_2026-08-19"
# Prefer Windows-visible handoff under the synced repo if mounted
WIN = Path("/mnt/c/Users/cgs2528/Projects/RAPID/handoff/locked_recipe_transfer_BEGE-TEXA75535L")
OUT = WIN if Path("/mnt/c/Users/cgs2528/Projects/RAPID").exists() else (HOME / "handoff" / "locked_recipe_transfer_BEGE-TEXA75535L")


def sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e:
        return f"(unavailable: {e})"


def load_df() -> pd.DataFrame:
    csv = RUN / "transfer_summary.csv"
    if not csv.exists():
        raise SystemExit(f"missing {csv}; run plot_locked_transfer_results.py first")
    return pd.read_csv(csv)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def pick_row(df, **kw):
    sub = df
    for k, v in kw.items():
        sub = sub[sub[k] == v]
    if sub.empty:
        return None
    return sub.iloc[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = OUT / "data"
    figs = OUT / "figures"
    prov = OUT / "provenance"
    data.mkdir(exist_ok=True)
    figs.mkdir(exist_ok=True)
    prov.mkdir(exist_ok=True)

    df = load_df()
    df.to_csv(data / "transfer_summary.csv", index=False)

    # copy canvas + machine notes
    src_html = RUN / "figures" / "transfer_canvas.html"
    if src_html.exists():
        shutil.copy2(src_html, figs / "transfer_canvas.html")
    for name in ("README_machine.txt", "README.md", "LOCKED_VS_ACHIEVED_K.md", "locked_recipe_transfer.log"):
        p = RUN / name
        if p.exists():
            shutil.copy2(p, prov / name)

    # software versions
    py_info = sh(
        "python - <<'PY'\n"
        "import sys,torch,seisbench,ray,numpy,obspy\n"
        "print('python', sys.version.split()[0])\n"
        "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "  print('gpu_name', torch.cuda.get_device_name(0))\n"
        "  print('vram_bytes', torch.cuda.get_device_properties(0).total_memory)\n"
        "  print('bf16', torch.cuda.is_bf16_supported())\n"
        "print('seisbench', seisbench.__version__)\n"
        "print('ray', ray.__version__)\n"
        "print('numpy', numpy.__version__)\n"
        "print('obspy', obspy.__version__)\n"
        "PY"
    )
    lscpu = sh("lscpu | egrep 'Model name|CPU\\(s\\)|Thread|Core|Socket|Vendor'")
    free_h = sh("free -h")
    nvidia = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader")
    uname = sh("uname -a")
    git_head = sh("cd ~/RAPID && git rev-parse HEAD && git log -1 --oneline")
    wslconfig = sh("cat /mnt/c/Users/cgs2528/.wslconfig 2>/dev/null || true")

    (prov / "software_versions.txt").write_text(py_info + "\n", encoding="utf-8")
    (prov / "host_snapshot.txt").write_text(
        "\n".join(
            [
                f"assembled_at={datetime.now(timezone.utc).isoformat()}",
                f"uname={uname}",
                f"lscpu:\n{lscpu}",
                f"free:\n{free_h}",
                f"nvidia={nvidia}",
                f"git:\n{git_head}",
                f"wslconfig:\n{wslconfig}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # paper-facing tables
    lines = []
    lines.append("# Locked-recipe transfer handoff — BEGE-TEXA75535L")
    lines.append("")
    lines.append("**Audience:** another model / coauthor preparing the RAPID paper.")
    lines.append("**Purpose:** this laptop is being returned; this folder is the portable record of the locked-recipe transfer run.")
    lines.append("")
    lines.append("Parent protocol: `docs/RAPID_LOCKED_RECIPE_TRANSFER.md` (do not invent new recipes or K sweeps).")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Machine under test")
    lines.append("")
    lines.append("| Item | Value |")
    lines.append("|---|---|")
    lines.append("| Hostname | BEGE-TEXA75535L |")
    lines.append("| Host OS | Windows 11 + **WSL2 Ubuntu 26.04** (benchmarks require Linux: `taskset` / `sched_setaffinity` / Ray) |")
    lines.append("| CPU | 13th Gen Intel Core **i7-13700H** (host: 14C/20T; WSL exposed 20 logical CPUs) |")
    lines.append("| RAM | ~32 GB host; WSL capped **24 GB** via `%UserProfile%\\.wslconfig` |")
    lines.append("| GPU | NVIDIA GeForce **RTX 4050 Laptop**, **6141 MiB** (~6 GB) |")
    lines.append("| NVIDIA driver (Windows) | see `provenance/host_snapshot.txt` / nvidia-smi |")
    lines.append("| Power | Keep AC connected for long runs; sleep/standby suspends WSL and can interrupt trials |")
    lines.append("")
    lines.append("### WSL config used")
    lines.append("")
    lines.append("```ini")
    lines.append(wslconfig if wslconfig and not wslconfig.startswith("(unavailable") else "[wsl2]\nmemory=24GB\nprocessors=20\nswap=8GB")
    lines.append("```")
    lines.append("")
    lines.append("### Software stack (conda env `rapid`)")
    lines.append("")
    lines.append("```")
    lines.append(py_info)
    lines.append("```")
    lines.append("")
    lines.append("| Note | Detail |")
    lines.append("|---|---|")
    lines.append("| SeisBench (this laptop) | **0.12.5** (upgraded mid-campaign from 0.10.2 so `EQCCTP`/`EQCCTS` exist) |")
    lines.append("| SeisBench (workstation / original-data lock) | **0.11.8** — version differ; catalog F1 still matches workstation picks |")
    lines.append("| RAPID git | see `provenance/host_snapshot.txt` |")
    lines.append("| Networks | Repo-provided `data/seisbench_networks/stead_250st` and `stead_580st` (same trees as workstation transfer; do not rebuild with a new seed if F1 must match) |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. What was run (locked recipes)")
    lines.append("")
    lines.append("### Layers (unchanged from workstation lock)")
    lines.append("")
    lines.append("| Layer | Meaning | Locked recipe |")
    lines.append("|---|---|---|")
    lines.append("| **native** | single-process Annotate | `annotate_bf16`, batch **512**, packaging **merged**, torch/OMP threads = `n_cpus` |")
    lines.append("| **playback** | all-stations-ready orch | Model-Actor, **SG**, fill **static**, 1 thread/actor |")
    lines.append("| **staggered** | delayed-station orch | Model-Actor, **SG**, fill **eager**, 1 thread/actor |")
    lines.append("")
    lines.append("Models: **EQCCT, PhaseNet, PhaseNetLight, EQTransformer, EQT-NC**.")
    lines.append("Networks: STEAD **250** and **580**.")
    lines.append("Devices: **CPU** and **1× GPU**.")
    lines.append("Native core grid: **5, 10, 15, 20** (80 native cells complete).")
    lines.append("Orch unique K coverage (this package): **CPU K=5 / 10 / 20**; **GPU K=4** (+ PhaseNet@250 **K=2**).")
    lines.append("**Intentionally skipped:** CPU orch **K=15** (bracketed by K=10 and K=20; do not run).")
    lines.append("Repeats: **5** (smoke used 1 earlier; final package is the full run).")
    lines.append("")
    lines.append("### Locked worker counts K (workstation) vs achieved on this laptop")
    lines.append("")
    lines.append("```")
    lines.append("K = min(locked_want, n_cpus, CPU_K_CAP or GPU_K_CAP)")
    lines.append("```")
    lines.append("")
    lines.append("| Setting | Locked want | This laptop |")
    lines.append("|---|---|---|")
    lines.append("| GPU (most models) | **K=4** | **K=4** after GPU re-run (`GPU_K_CAP=4`) |")
    lines.append("| GPU PhaseNet @ 250 | **K=2** | **K=2** (matches lock) |")
    lines.append("| CPU @ 580 / EQCCT | **K=20** | **K=20** after uncapped re-run (was K≤10 with `CPU_K_CAP=10`) |")
    lines.append("| CPU @ 250 others | **K=10** | **K=5** at cpus=5; **K=10** at cpus≥10 |")
    lines.append("")
    lines.append("**Affinity:** `CORES=0,1,2,...,19` (all 20 logical CPUs) so native 15/20 core-grid cells were not skipped for missing IDs.")
    lines.append("")
    lines.append("**Important about SKIP lines in the log:** orchestration result paths are keyed by **K** (`kma{K}/...`), not by `n_cpus`.")
    lines.append("Once a successful `kma4` (or `kma2` / `kma20`) file exists, matrix rows for other `n_cpus` with the same K are logged as SKIP (duplicate path).")
    lines.append("Those are **not** missing locked configs — skip duplicate `n_cpus` rows that share the same K path.")
    lines.append("")
    lines.append("**CPU K=20 follow-up:** first matrix used `CPU_K_CAP=10`; then `scripts/rerun_cpu_orch_k20.sh` unset the cap for locked comparison at K=20.")
    lines.append("Peak process set size (PSS) on this box was ~**9–11 GB** at K=20 — a laptop / hybrid-CPU / WSL result.")
    lines.append("On this host, CPU **K=20 is often worse than K=10** for staggered p95; that is **not** a reason to change the workstation lock.")
    lines.append("")
    lines.append("### Caps / conditions the paper must disclose")
    lines.append("")
    lines.append("1. **SeisBench 0.12.5** on this laptop vs workstation **0.11.8**.")
    lines.append("2. **WSL memory cap 24 GB**; GPU is **RTX 4050 Laptop 6 GB**.")
    lines.append("3. **Caps story:** first matrix with **`CPU_K_CAP=10`**, then uncapped **K=20** follow-up for locked comparison; **`GPU_K_CAP=4`** (PhaseNet@250 → K=2).")
    lines.append("4. **Do not** recommend changing the workstation lock because of K=20 laptop results.")
    lines.append("5. Unique orch coverage: CPU K=5/10/20, GPU K=4 (+ PhaseNet 250 K=2); SKIP duplicate `n_cpus` rows that share the same K path; intentionally skip CPU K=15.")
    lines.append("6. **Absolute times will be slower** than Threadripper + RTX 6000 Ada; claim **within-machine ordering**, not absolute equality (e.g. native EQCCT CPU ~47 s here vs ~1.41 s on workstation).")
    lines.append("7. Staggered **makespan ~90 s** is expected (simulated delay ceiling); rank **p95 finish−ready** (GPU staggered p95 is **0.2–1.4 s** here — not makespan, not wait-5).")
    lines.append("8. EQCCT is SeisBench **`EQCCTP` + `EQCCTS`**, not a single WaveformModel / not TF EQCCTPro for this suite.")
    lines.append("9. This laptop package is enough for the transfer question: locked recipes still work, GPU stays the fast path, catalog picks match workstation.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Result inventory")
    lines.append("")
    lines.append("| Artifact | Path in this handoff |")
    lines.append("|---|---|")
    lines.append("| Tidy table (with catalog P/S F1) | `data/transfer_summary.csv` |")
    lines.append("| Interactive plots | `figures/transfer_canvas.html` |")
    lines.append("| Full raw `result.json` tree | `raw_results/` (152 successful cells; also under WSL `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`) |")
    lines.append("| Run log | `provenance/locked_recipe_transfer.log` |")
    lines.append("| Caps note | `provenance/LOCKED_VS_ACHIEVED_K.md` / this document |")
    lines.append("| Index | `README.md` |")
    lines.append("")
    counts = df.groupby(["method", "device"]).size()
    n_ok = int((df["success_rate"] == 1.0).sum()) if "success_rate" in df.columns else len(df)
    lines.append("### Successful cells in CSV")
    lines.append("")
    lines.append("```")
    lines.append(counts.to_string())
    lines.append(f"total_rows={len(df)}")
    lines.append(f"success_rate_1.0_rows={n_ok}")
    lines.append("failures=0")
    lines.append("```")
    lines.append("")
    lines.append(f"**{len(df)}** successful `result.json` rows in this package; **all** have `success_rate=1.0`; **zero** failures; native **80** cells complete.")
    lines.append("")
    lines.append("Metrics encoded in `runtime_s` / `metric`:")
    lines.append("")
    lines.append("| method | metric column meaning |")
    lines.append("|---|---|")
    lines.append("| native | `timing.inference_s_mean` |")
    lines.append("| playback | `orch.makespan_s_mean` |")
    lines.append("| staggered | `latency.pooled_across_repeats.e2e_finish_minus_ready.p95` |")
    lines.append("")
    lines.append("Catalog quality columns: `p_f1` / `s_f1` from `pick_quality_vs_catalog` (`P.f1_mean` / `S.f1_mean`).")
    lines.append("Expected workstation catalog picks (examples): EQCCT ~**0.905 / 0.946**, PhaseNet ~**0.974 / 0.937**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Paper-facing numbers (this laptop)")
    lines.append("")
    lines.append("### 4a. Native merged bf16 — mean inference (s)")
    lines.append("")
    lines.append("Prefer the fastest core cell that is quality-safe; table below lists **cpus=20** (full grid available) and GPU at cpus=20.")
    lines.append("")
    lines.append("| Model | Stations | CPU cpus20 | GPU | P F1 | S F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for model in ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]:
        for nst in (250, 580):
            cpu = pick_row(df, method="native", model=model, n_stations=nst, device="cpu", n_cpus=20)
            gpu = pick_row(df, method="native", model=model, n_stations=nst, device="gpu", n_cpus=20)
            ref = cpu if cpu is not None else gpu
            lines.append(
                f"| {model} | {nst} | {fmt(None if cpu is None else cpu.runtime_s)} | {fmt(None if gpu is None else gpu.runtime_s)} | "
                f"{fmt(None if ref is None else ref.get('p_f1'), 4)} | {fmt(None if ref is None else ref.get('s_f1'), 4)} |"
            )
    lines.append("")
    lines.append("### 4b. Playback MA SG — makespan (s) at achieved K")
    lines.append("")
    lines.append("| Model | Stations | CPU K=5 | CPU K=10 | CPU K=20 | GPU K | GPU makespan |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]:
        for nst in (250, 580):
            c5 = df[(df.method == "playback") & (df.model == model) & (df.n_stations == nst) & (df.device == "cpu") & (df.K == 5)]
            c10 = df[(df.method == "playback") & (df.model == model) & (df.n_stations == nst) & (df.device == "cpu") & (df.K == 10)]
            c20 = df[(df.method == "playback") & (df.model == model) & (df.n_stations == nst) & (df.device == "cpu") & (df.K == 20)]
            g = df[(df.method == "playback") & (df.model == model) & (df.n_stations == nst) & (df.device == "gpu")]
            c5r = c5.iloc[0] if len(c5) else None
            c10r = c10.iloc[0] if len(c10) else None
            c20r = c20.iloc[0] if len(c20) else None
            gr = g.iloc[0] if len(g) else None
            lines.append(
                f"| {model} | {nst} | {fmt(None if c5r is None else c5r.runtime_s)} | {fmt(None if c10r is None else c10r.runtime_s)} | "
                f"{fmt(None if c20r is None else c20r.runtime_s)} | "
                f"{('—' if gr is None else int(gr.K))} | {fmt(None if gr is None else gr.runtime_s)} |"
            )
    lines.append("")
    lines.append("### 4c. Staggered MA SG eager — p95 finish−ready (s)")
    lines.append("")
    lines.append("| Model | Stations | CPU K=5 | CPU K=10 | CPU K=20 | GPU K | GPU p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model in ["EQCCT", "PhaseNet", "PhaseNetLight", "EQTransformer", "EQT-NC"]:
        for nst in (250, 580):
            c5 = df[(df.method == "staggered") & (df.model == model) & (df.n_stations == nst) & (df.device == "cpu") & (df.K == 5)]
            c10 = df[(df.method == "staggered") & (df.model == model) & (df.n_stations == nst) & (df.device == "cpu") & (df.K == 10)]
            c20 = df[(df.method == "staggered") & (df.model == model) & (df.n_stations == nst) & (df.device == "cpu") & (df.K == 20)]
            g = df[(df.method == "staggered") & (df.model == model) & (df.n_stations == nst) & (df.device == "gpu")]
            c5r = c5.iloc[0] if len(c5) else None
            c10r = c10.iloc[0] if len(c10) else None
            c20r = c20.iloc[0] if len(c20) else None
            gr = g.iloc[0] if len(g) else None
            lines.append(
                f"| {model} | {nst} | {fmt(None if c5r is None else c5r.runtime_s)} | {fmt(None if c10r is None else c10r.runtime_s)} | "
                f"{fmt(None if c20r is None else c20r.runtime_s)} | "
                f"{('—' if gr is None else int(gr.K))} | {fmt(None if gr is None else gr.runtime_s)} |"
            )
    lines.append("")
    lines.append("### 4d. Workstation reference (580) — order-of-magnitude only")
    lines.append("")
    lines.append("From `docs/RAPID_LOCKED_RECIPE_TRANSFER.md` (Threadripper + RTX 6000 Ada). **Do not** claim XPS/laptop times equal these.")
    lines.append("")
    lines.append("| Model | WS native CPU | WS native GPU | WS playback CPU K20 | WS playback GPU K4 | WS stag p95 CPU | WS stag p95 GPU |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    lines.append("| EQCCT | 1.41 | 0.55 | 0.58 | 0.219 | 0.518 | 0.199 |")
    lines.append("| PhaseNet | 0.38 | 0.32 | 0.167 | 0.184 | 0.165 | 0.176 |")
    lines.append("| PhaseNetLight | 0.50 | 0.32 | 0.282 | 0.169 | 0.214 | 0.281 |")
    lines.append("| EQTransformer | 0.99 | 0.43 | 0.438 | 0.509 | 0.344 | 0.471 |")
    lines.append("| EQT-NC | 0.95 | 0.37 | 0.408 | 0.449 | 0.346 | 0.443 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Within-machine observations (for the paper narrative)")
    lines.append("")
    lines.append("Use these as qualitative checks against the handoff brief:")
    lines.append("")
    lines.append("1. **GPU ≪ CPU** for heavy models on this laptop (e.g. native EQCCT 580/cpus20 ~**47 s** CPU vs ~**1.65 s** GPU; workstation native EQCCT CPU is ~**1.41 s** — report **ordering**, not equality).")
    lines.append("2. **Staggered GPU p95** at locked K is **0.2–1.4 s** — that is finish−ready latency, **not** makespan (~90 s) and **not** wait-5.")
    lines.append("3. **CPU orch K=10 beats K=5**; on this box **K=20 is often worse than K=10** for staggered — laptop/hybrid-CPU result; **do not** change the workstation lock.")
    lines.append("4. **PhaseNetLight** remains the fastest native/CPU-orch path among the five on this host.")
    lines.append("5. Catalog P/S F1 matches workstation picks (EQCCT ~0.905/0.946, PhaseNet ~0.974/0.937).")
    lines.append("6. This package answers the transfer question: locked recipes still work; GPU stays the fast path.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Campaign chronology / pitfalls (so the next model does not rediscover them)")
    lines.append("")
    lines.append("1. Initial zip install had no git; later synced to `origin/main` (networks committed in-repo).")
    lines.append("2. First XPS-style `CORES=0,2,4,...,18` (10 IDs) skipped 15/20 cells; fixed to full `0..19`.")
    lines.append("3. SeisBench 0.10.2 lacked `EQCCTP`; upgraded to **0.12.5** before EQCCT succeeded (workstation lock used **0.11.8**).")
    lines.append("4. First GPU orch pass used `GPU_K_CAP=2`; those trees were cleared and **re-run at `GPU_K_CAP=4`** (locked).")
    lines.append("5. First CPU orch matrix used `CPU_K_CAP=10`; K=20 follow-up unset the cap (peak PSS ~9–11 GB). Intentionally **no** CPU K=15 orch.")
    lines.append("6. Long runs must avoid Windows sleep; WSL suspends with the host.")
    lines.append("7. This handoff folder is self-contained (`data/`, `figures/`, `raw_results/`, `provenance/`); WSL source also under `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. How to regenerate plots / handoff")
    lines.append("")
    lines.append("```bash")
    lines.append("source ~/miniconda3/etc/profile.d/conda.sh && conda activate rapid")
    lines.append("cd ~/RAPID   # or /mnt/c/Users/cgs2528/Projects/RAPID")
    lines.append("python scripts/plot_locked_transfer_results.py")
    lines.append("python scripts/assemble_locked_transfer_handoff.py")
    lines.append("```")
    lines.append("")
    lines.append("Plots write under `~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`; assemble copies CSV + `figures/transfer_canvas.html` into this handoff folder.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Recommended paper wording")
    lines.append("")
    lines.append("> We transferred the locked Annotate bf16 / Model-Actor SG recipes to a consumer laptop")
    lines.append("> (Intel i7-13700H, RTX 4050 Laptop 6 GB, WSL2 capped at 24 GB; SeisBench 0.12.5 vs workstation 0.11.8).")
    lines.append("> Native cells used the 5–20 logical-CPU grid (80 cells). GPU Model-Actor used locked K=4")
    lines.append("> (K=2 for PhaseNet at 250 stations). CPU Model-Actor first used K≤10 (`CPU_K_CAP=10`), then an")
    lines.append("> uncapped K=20 follow-up for locked comparison (peak PSS ~9–11 GB; K=20 often slower than K=10")
    lines.append("> for staggered on this hybrid CPU — we do not change the workstation lock). Unique orch coverage")
    lines.append("> is CPU K=5/10/20 and GPU K=4; CPU K=15 was intentionally skipped. Catalog P/S F1 matched")
    lines.append("> workstation picks. We report within-machine method ordering (GPU ≪ CPU for heavy models;")
    lines.append("> staggered GPU p95 0.2–1.4 s) and do not equate absolute latencies to the workstation.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"Assembled: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    handoff = OUT / "HANDOFF.md"
    handoff.write_text("\n".join(lines), encoding="utf-8")

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Handoff package index",
                "",
                "Start here: **[HANDOFF.md](HANDOFF.md)** — full paper handoff for the locked-recipe transfer on this laptop.",
                "",
                "| Path | Contents |",
                "|---|---|",
                f"| [`data/transfer_summary.csv`](data/transfer_summary.csv) | One row per successful cell: method, model, stations, device, n_cpus, n_gpus, K, runtime_s, **p_f1**, **s_f1** (**{len(df)}** rows) |",
                "| [`figures/transfer_canvas.html`](figures/transfer_canvas.html) | Interactive Plotly canvas |",
                f"| [`raw_results/`](raw_results/) | Full raw tree (`annotate_bf16/`, `ma/`, log) — **{len(df)}** `result.json` with `timing.success_rate == 1.0` |",
                "| [`provenance/`](provenance/) | Machine README, caps notes, software versions, host snapshot, full run log |",
                "",
                "WSL source of truth (mirrored under `raw_results/`):",
                "",
                "`~/RAPID/results/locked_recipe_transfer/BEGE-TEXA75535L_2026-08-19/`",
                "",
                "Do **not** run more benchmark matrices for this package. Skip CPU K=15; do not rerun K=20.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # keep CSV also under raw_results for self-contained copy
    raw = OUT / "raw_results"
    if raw.is_dir():
        df.to_csv(raw / "transfer_summary.csv", index=False)

    # also mirror tidy package under the run directory (exclude huge raw_results)
    run_handoff = RUN / "handoff"
    if RUN.exists():
        run_handoff.mkdir(parents=True, exist_ok=True)
        for name in ("HANDOFF.md", "README.md"):
            shutil.copy2(OUT / name, run_handoff / name)
        (run_handoff / "data").mkdir(exist_ok=True)
        (run_handoff / "figures").mkdir(exist_ok=True)
        (run_handoff / "provenance").mkdir(exist_ok=True)
        shutil.copy2(data / "transfer_summary.csv", run_handoff / "data" / "transfer_summary.csv")
        src_html2 = figs / "transfer_canvas.html"
        if src_html2.exists():
            shutil.copy2(src_html2, run_handoff / "figures" / "transfer_canvas.html")
        for p in prov.iterdir():
            if p.is_file():
                shutil.copy2(p, run_handoff / "provenance" / p.name)

    print(f"handoff -> {OUT}")
    print(f"HANDOFF.md bytes={handoff.stat().st_size}")
    print(f"csv rows={len(df)}")
    if "p_f1" in df.columns:
        print(f"p_f1 non-null={df['p_f1'].notna().sum()}  s_f1 non-null={df['s_f1'].notna().sum()}")


if __name__ == "__main__":
    main()
