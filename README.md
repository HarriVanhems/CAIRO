# CAIRO

Research repository for the CAIRO paper: calibrated two-stage rank-then-regress modeling for zero-inflated and heavy-tailed regression.

## What Is Canonical

- `src/cairo/`: reusable Python code
- `scripts/`: runnable experiment entry points
- `data/`: raw and processed datasets
- `results/`: generated benchmark outputs and figures
- `paper/`: local paper drafts
- `notebooks/archive/`: exploratory notebook history kept for provenance, not as the source of truth

## Current Entry Points

The main paper-aligned synthetic benchmark is now:

```bash
python scripts/run_paper_synthetic_benchmark.py
```

This runner uses the CAIRO synthetic setup that is closer to the current draft:

- CAIRO models defined as rank-first stage-1 scorers followed by isotonic stage-2 calibration
- stage-1 CAIRO variants now include `CAIROK1`, `CAIROG1`, `CAIROS1`, `CAIROG2`, plus a boosting-based `CAIRO-XGBPairwise` ranker
- baselines compared directly without isotonic post-processing in the main benchmark
- baseline neural regression now includes `NN-MSE`, `NN-MAE`, and a validation-tuned `NN-Huber` model with the same `32 -> 16 -> 1` architecture used in the other MLP baselines
- strong tabular baselines plus robust alternatives like quantile regression
- monotone-distortion and low-data studies that isolate the value of the two-stage design
- contamination and heavy-tail stress sweeps
- pair-sampling ablations
- saved NeurIPS-style plots under `results/benchmarks/paper_synthetic/figures/`

Run modes:

- `python scripts/run_paper_synthetic_benchmark.py`
  smoke run with the full paper-aligned pipeline
- `python scripts/run_paper_synthetic_benchmark.py --skip-sweeps`
  quick local check of the main benchmark only
- `python scripts/run_paper_synthetic_benchmark.py --mode debug`
  single-repeat research run with all main scenarios
- `python scripts/run_paper_synthetic_benchmark.py --mode full`
  larger paper-style run

The focused positive-skew severity study is available as a separate runner:

```bash
python scripts/run_positive_skew_study.py
```

Use this when you want to test the hypothesis that CAIRO is strongest in positive, skewed, heteroskedastic regimes without running the entire synthetic benchmark suite.

The earlier cleaned benchmark extracted from the workbench notebook remains available as a secondary synthetic runner:

```bash
python3 scripts/run_zero_inflated_benchmark.py
```

This benchmark implements the zero-inflated Tweedie synthetic experiment using a stronger, reproducible protocol:

- separate training, tuning, calibration, and test splits
- train-only relevance binning
- LightGBM LambdaMART ranker plus isotonic calibration
- multi-seed summaries saved under `results/benchmarks/`

For the real-data side, the current canonical inventory check is:

```bash
python scripts/inspect_real_datasets.py
```

That script loads the benchmark datasets from `data/raw/benchmarks/` and writes a standardized dataset summary to `results/benchmarks/real_data_inventory/`.

## Repo Layout

- `data/raw/benchmarks/`: one folder per benchmark dataset
- `data/processed/`: cleaned intermediate artifacts
- `docs/`: repo map and experiment upgrade notes
- `notebooks/archive/`: mixed prototype notebook snapshots
- `paper/drafts/`: local PDF drafts
- `results/figures/`: paper figures and analysis plots

## Working Rules

- Keep exploratory work in notebooks, but promote anything used in the paper into `src/` and `scripts/`.
- Save benchmark outputs to `results/benchmarks/` with config files and per-run CSVs.
- Treat `notebooks/archive/paper_workbench.ipynb` as historical context, not the implementation target.
