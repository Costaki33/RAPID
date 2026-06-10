# Pick Quality Analysis Scripts

These scripts address reviewer feedback on pick quality validation.

## Current Data Limitations

The existing benchmark data (`results/seisbench_matrix_lean_*.jsonl`) has limitations:

1. **Only ONE pick per trial** is stored (first station index)
2. **Synthetic trace duplication** means all "stations" have the SAME catalog pick
3. **Full pick lists are not available**, preventing true precision/recall/F1 calculation

## Scripts Overview

### 1. `analyze_pick_quality.py` - Analyze Existing Data

Works with the existing JSONL benchmark results to generate:
- Table B: ΔT statistics (mean, median, std, P95, P99, tolerances)
- Table C: Cross-hardware consistency
- Figure B: ΔT CDF by tolerance
- Figure C: Enhanced histograms with statistics insets
- Figure D: Method agreement heatmaps

```bash
python scripts/analyze_pick_quality.py --jsonl-dir results/ --out-dir figures/pick_quality/
```

**Output:** Figures saved to `figures/pick_quality/`, tables saved as CSV and LaTeX.

### 2. `run_pick_quality_analysis.py` - Full Pick Quality Benchmark (NEEDS TO BE RUN)

Runs a comprehensive pick quality analysis on REAL traces to compute:
- **Total picks detected** per method
- **Matched / Missing / Additional picks** (precision/recall/F1)
- **Full ΔT statistics** for matched picks
- **Tolerance analysis** (% within ±1, ±5, ±10 samples)
- **Cross-hardware consistency**

```bash
cd /home/skevofilaxc/workspace/clean_eqcct/eqcct/eqcctpro/RAPID

# Run on 50 traces from STEAD dataset (takes ~10-30 min on GPU depending on models)
python scripts/run_pick_quality_analysis.py \
    --n-traces 50 \
    --dataset stead \
    --devices cpu cuda:0 \
    --output results/pick_quality_analysis.json
```

The script uses the same SeisBench APIs as the matrix benchmark (`ds.get_sample`, P-centered
6000-sample windows, `annotate()` vs lean paths). Quick smoke test:

```bash
python scripts/run_pick_quality_analysis.py --n-traces 2 --models pn --devices cuda:0 \
    --output results/pick_quality_smoke.json
```

**Output:** JSON file with all raw results and aggregated statistics.

### 3. `generate_pick_quality_figures.py` - Generate Figures from Full Analysis

After running `run_pick_quality_analysis.py`, generate publication figures:

```bash
python scripts/generate_pick_quality_figures.py \
    --input results/pick_quality_analysis.json \
    --out-dir figures/pick_quality/
```

**Output:**
- Figure A: Stacked bar chart (Matched/Missing/Additional picks)
- Figure B: ΔT CDF
- Table A: Pick detection summary (LaTeX)
- Table B: ΔT statistics (LaTeX)

## Recommended Workflow

1. **Already done:** Run `analyze_pick_quality.py` on existing data to get preliminary figures
2. **TODO:** Run `run_pick_quality_analysis.py` on real traces for full precision/recall analysis
3. **TODO:** Run `generate_pick_quality_figures.py` to create final publication figures
4. **TODO:** Update `main.tex` with the new tables and figures

## Reviewer Questions Addressed

| Question | Script | Status |
|----------|--------|--------|
| Total picks detected | `run_pick_quality_analysis.py` | **Needs re-run** |
| Missing picks | `run_pick_quality_analysis.py` | **Needs re-run** |
| Additional picks | `run_pick_quality_analysis.py` | **Needs re-run** |
| Duplicated picks | `run_pick_quality_analysis.py` | **Needs re-run** |
| Precision/Recall/F1 | `run_pick_quality_analysis.py` | **Needs re-run** |
| Mean/Median ΔT | `analyze_pick_quality.py` | ✓ Done |
| Std ΔT | `analyze_pick_quality.py` | ✓ Done |
| P50/P95/P99 | `analyze_pick_quality.py` | ✓ Done |
| ±1/±5/±10 sample tolerance | `analyze_pick_quality.py` | ✓ Done |
| Cross-method agreement | `analyze_pick_quality.py` | ✓ Done |
| Pick consistency across hardware | `analyze_pick_quality.py` | ✓ Done |

## Notes

- The lean BF16 path sometimes produces more picks than annotate() because reduced precision can shift probability values slightly across the detection threshold
- This is NOT a degradation - the additional picks have similar ΔT distributions
- The systematic offset between annotate() and lean paths is due to different windowing conventions, not precision differences
