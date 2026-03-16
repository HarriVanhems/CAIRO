# NeurIPS Experiment Style Notes

The synthetic benchmark and plots are now organized around a few presentation patterns that show up repeatedly in recent NeurIPS work:

## Plot Conventions Adopted

1. Robustness sweeps instead of only fixed stress points.
   We added performance-vs-severity plots for contamination rate and Student-t tail heaviness. This matches the style of benchmark and robustness papers that emphasize trends across shift severity instead of one-off settings.

2. Calibration visualization, not only scalar calibration metrics.
   We added reliability-style plots based on quantile bins so calibration behavior is visible, not just summarized by a single gap metric.

3. Ablation curves with uncertainty.
   Pair-budget and sampler ablations are saved as line plots with confidence intervals rather than hidden inside notebook tables.

4. Multi-scenario summaries with error bars.
   The main benchmark now saves summary plots with mean and 95% confidence intervals across repeats.

## Reference Papers

- RankUp: Boosting Semi-Supervised Regression with an Auxiliary Ranking Classifier
  [NeurIPS 2024 proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c26a8494fe31695db965ae8b7244b7c1-Abstract-Conference.html)

- Benchmarking Distribution Shift in Tabular Data with TableShift
  [NeurIPS 2023 virtual poster page](https://neurips.cc/virtual/2023/session/74074)

- Better Uncertainty Calibration via Proper Scores for Classification and Beyond
  [NeurIPS 2022 proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3915a87ddac8e8c2f23dbabbcee6eec9-Abstract-Conference.html)
