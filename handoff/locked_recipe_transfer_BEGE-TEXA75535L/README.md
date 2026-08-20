# Handoff package index

Start here: **[HANDOFF.md](HANDOFF.md)** — full paper handoff for the locked-recipe transfer on this laptop.

| Path | Contents |
|---|---|
| `data/transfer_summary.csv` | One row per successful cell: method, model, stations, device, n_cpus, n_gpus, K, runtime_s (**140** rows) |
| `figures/transfer_canvas.html` | Interactive Plotly canvas |
| `raw_results/` | Full raw tree (`annotate_bf16/`, `ma/`, log, matrix script) — **140** `result.json` with `timing.success_rate == 1.0` |
| `provenance/` | Machine README, caps notes, software versions, host snapshot, full run log |

WSL source of truth (also mirrored here under `raw_results/`):

`\\wsl$\Ubuntu\home\cgs2528\RAPID\results\locked_recipe_transfer\BEGE-TEXA75535L_2026-08-19\`
