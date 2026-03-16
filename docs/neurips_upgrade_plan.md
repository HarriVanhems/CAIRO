# NeurIPS-Quality Upgrade Plan

This plan is based on the current arXiv draft (`2602.14440`) and the archived notebook implementation.

## Immediate Improvements Already Applied

- A professional repo layout replaces the flat, mixed workspace.
- The exploratory notebook is archived instead of being treated as the canonical implementation.
- The zero-inflated Tweedie LambdaMART experiment has been extracted into a reproducible script with saved outputs.
- The synthetic benchmark protocol now separates tuning and calibration instead of reusing one split for both.

## Highest-Value Next Steps

1. Standardize the real-data benchmark.
   Write one loader per dataset, fix target definitions, and lock preprocessing so every baseline sees the same features and splits.

2. Promote CAIRO models into reusable modules.
   The notebook currently duplicates RankNet, weighted RankNet, and soft-Gini training logic. Those should live in `src/cairo/` with one training API and one evaluation API.

3. Strengthen statistical reporting.
   For submission-grade tables, report paired multi-seed results with confidence intervals and paired significance tests, not only mean performance.

4. Expand baselines.
   The current notebook already points toward stronger baselines such as LightGBM rankers and regressors. Add tuned gradient-boosting regressors and a calibrated ranker baseline across all experiments.

5. Add explicit ablations.
   The paper should isolate the effect of:
   - ranking objective
   - pair weighting
   - isotonic calibration
   - relevance discretization levels
   - calibration split size
   - contamination level and misspecification level

6. Improve reproducibility.
   Save configs, seeds, per-run outputs, and aggregate tables for every experiment under `results/benchmarks/`.

## Current Risks

- Real-data preprocessing is still implicit inside the notebook.
- Deep-model dependencies are not yet fully packaged.
- Some benchmark logic is duplicated across notebook sections, which increases the risk of subtle protocol drift.
