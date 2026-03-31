from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import default_rng

from cairo.paths import RESULTS_DIR, ensure_dir
from cairo.reporting import write_csv, write_json
from cairo.synthetic.paper_benchmark import (
    FAIR_NEURAL_MODELS,
    RuntimeConfig,
    ScenarioSpec,
    SHARED_NEURAL_HIDDEN_UNITS,
    _add_panel_tag,
    _model_color,
    _plot_style,
    available_components,
    fit_isotonic,
    make_experiment_splits,
    predict_tf,
    ranknet_scores,
    run_one_repeat,
    summarize_with_ci,
    train_mlp_regressor_tf,
    train_ranknet_stage1,
)


DEFAULT_MODELS = FAIR_NEURAL_MODELS


@dataclass(frozen=True)
class MatchedErrorStudyConfig:
    runtime: RuntimeConfig
    severity_levels: tuple[float, ...]
    models: tuple[str, ...] = DEFAULT_MODELS
    output_dir: Path = RESULTS_DIR / "benchmarks" / "matched_error_study"
    visualization_severity: float = 0.75
    visualization_samples: int = 1200

    @classmethod
    def from_mode(
        cls,
        mode: str,
        output_dir: Path | None = None,
        repeats: int | None = None,
        severity_levels: tuple[float, ...] | None = None,
        visualization_severity: float = 0.75,
        visualization_samples: int = 1200,
    ) -> "MatchedErrorStudyConfig":
        out_dir = output_dir or RESULTS_DIR / "benchmarks" / "matched_error_study"
        runtime = RuntimeConfig.from_mode(mode, output_dir=out_dir, repeats=repeats)
        default_levels = {
            "smoke": (0.0, 0.5, 1.0),
            "debug": (0.0, 0.25, 0.5, 0.75, 1.0),
            "full": (0.0, 0.25, 0.5, 0.75, 1.0),
        }
        return cls(
            runtime=runtime,
            severity_levels=severity_levels or default_levels[mode],
            output_dir=out_dir,
            visualization_severity=visualization_severity,
            visualization_samples=visualization_samples,
        )


def _story_label(model_name: str) -> str:
    labels = {
        "CAIROK1": "CAIRO-K1",
        "CAIROG1": "CAIRO-G1",
        "CAIROS1": "CAIRO-S1",
        "CAIROG2": "CAIRO-G2",
        "NN-Huber": "NN-Huber",
        "NN-MSE": "NN-MSE",
        "NN-MAE": "NN-MAE",
    }
    return labels.get(model_name, model_name)


def _noise_family_title(noise_family: str) -> str:
    return {
        "Additive": "Additive heavy-tail noise",
        "Multiplicative": "Multiplicative heavy-tail noise",
    }[noise_family]


def _standardized_student_t(rr: np.random.Generator, df: float, size: int) -> np.ndarray:
    draws = rr.standard_t(df=df, size=size)
    variance = max(df / max(df - 2.0, 1e-6), 1e-6)
    return draws / math.sqrt(variance)


def _linear_core(seed: int, runtime: RuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(runtime.n_samples, 10))
    beta = rr.normal(size=10)
    z = x @ beta / math.sqrt(10.0)
    z = (z - z.mean()) / (z.std() + 1e-8)
    return x.astype(np.float32), z.astype(np.float32)


def generate_matched_additive_dataset(
    seed: int,
    runtime: RuntimeConfig,
    severity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, z = _linear_core(seed, runtime)
    rr = default_rng(seed + 10_001)
    df = max(2.5, 20.0 / (1.0 + 3.0 * severity))
    eps = _standardized_student_t(rr, df=df, size=runtime.n_samples)
    y = z + severity * eps
    return x, y.astype(np.float32), z.astype(np.float32)


def generate_matched_multiplicative_dataset(
    seed: int,
    runtime: RuntimeConfig,
    severity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, z = _linear_core(seed, runtime)
    rr = default_rng(seed + 20_001)
    mu = 2.5 + 0.7 * z
    sigma = severity
    mult = rr.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=runtime.n_samples)
    y = mu * mult
    return x, y.astype(np.float32), mu.astype(np.float32)


def _nonlinear_1d_base(x: np.ndarray) -> np.ndarray:
    return x + 0.55 * np.tanh(1.2 * x)


def _scenario_for_family(noise_family: str, severity: float) -> ScenarioSpec:
    if noise_family == "Additive":
        return ScenarioSpec(
            name=f"Matched additive (s={severity:.1f})",
            kind="matched_additive",
            generator=lambda seed, runtime, value=severity: generate_matched_additive_dataset(seed, runtime, value)[:2],
        )
    if noise_family == "Multiplicative":
        return ScenarioSpec(
            name=f"Matched multiplicative (s={severity:.1f})",
            kind="matched_multiplicative",
            generator=lambda seed, runtime, value=severity: generate_matched_multiplicative_dataset(seed, runtime, value)[:2],
        )
    raise ValueError(f"Unsupported noise family: {noise_family}")


def plot_matched_error_story(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    selected = [m for m in DEFAULT_MODELS if m in set(summary["Model"])]
    families = ["Additive", "Multiplicative"]
    metric_specs = (
        ("PairAcc", "Pair accuracy"),
        ("RMSE", "Test RMSE"),
        ("CalGap", "Calibration gap"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4), sharex=True)
    for row_idx, noise_family in enumerate(families):
        family_df = summary[summary["NoiseFamily"] == noise_family]
        for col_idx, (metric, title) in enumerate(metric_specs):
            axis = axes[row_idx][col_idx]
            for model_name in selected:
                subset = family_df[family_df["Model"] == model_name].sort_values("Severity")
                if subset.empty:
                    continue
                axis.errorbar(
                    subset["Severity"],
                    subset[f"{metric}_mean"],
                    yerr=np.nan_to_num(subset[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
                    marker="o",
                    linewidth=2.0 if model_name.startswith("CAIRO") else 1.8,
                    capsize=3,
                    color=_model_color(model_name),
                    label=_story_label(model_name),
                    alpha=1.0 if model_name.startswith("CAIRO") else 0.92,
                )
            axis.set_facecolor("#fcfcfc")
            axis.grid(True, which="major", alpha=0.16, linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("Severity")
            axis.set_ylabel(title)
            if col_idx == 0:
                axis.text(
                    -0.28,
                    0.5,
                    _noise_family_title(noise_family),
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

    for axis, tag in zip(axes.ravel(), ("A", "B", "C", "D", "E", "F")):
        _add_panel_tag(axis, tag)

    handles, labels = axes[-1][-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig((plot_dir / "matched_error_story").with_suffix(".png"), bbox_inches="tight")
    fig.savefig((plot_dir / "matched_error_story").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _generate_1d_dataset(
    seed: int,
    n_samples: int,
    severity: float,
    noise_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.uniform(-2.5, 2.5, size=n_samples)
    base = _nonlinear_1d_base(x)
    if noise_family == "Additive":
        mu = base
        df = max(2.5, 20.0 / (1.0 + 3.0 * severity))
        eps = _standardized_student_t(rr, df=df, size=n_samples)
        y = mu + severity * eps
    elif noise_family == "Multiplicative":
        mu = 2.0 + np.exp(0.45 * base)
        sigma = severity
        mult = rr.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=n_samples)
        y = mu * mult
    else:
        raise ValueError(f"Unsupported noise family: {noise_family}")
    return x.astype(np.float32), y.astype(np.float32), mu.astype(np.float32)


def _affine_rescale_for_display(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=float)
    scaled = values - values.mean()
    denom = values.std() + 1e-8
    scaled = scaled / denom
    scaled = scaled * target.std() + target.mean()
    return scaled


def build_1d_visualization_curves(config: MatchedErrorStudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    sample_rows: list[dict[str, float | str]] = []
    vis_runtime = replace(config.runtime, n_samples=max(config.visualization_samples, 800))
    severity = config.visualization_severity
    for family_idx, noise_family in enumerate(("Additive", "Multiplicative")):
        seed = config.runtime.seed + 90_000 + family_idx * 1_000
        x, y, mu = _generate_1d_dataset(seed, vis_runtime.n_samples, severity, noise_family)
        splits = make_experiment_splits(x.reshape(-1, 1), y, seed=seed)

        cairo = train_ranknet_stage1(
            splits["x_train"],
            splits["y_train"],
            splits["x_val"],
            splits["y_val"],
            vis_runtime,
            sampler="uniform",
            weight_mode="abs_diff",
            seed=seed + 20,
        )
        cairo_cal = fit_isotonic(
            ranknet_scores(cairo, splits["x_cal"], vis_runtime.batch_size),
            splits["y_cal"],
        )
        huber = train_mlp_regressor_tf(
            splits["x_train"],
            splits["y_train"],
            splits["x_val"],
            splits["y_val"],
            vis_runtime,
            seed=seed + 40,
            loss_name="huber",
        )

        grid = np.linspace(-2.5, 2.5, 250, dtype=np.float32)
        grid_base = _nonlinear_1d_base(grid)
        if noise_family == "Additive":
            true_mean = grid_base
        else:
            true_mean = 2.0 + np.exp(0.45 * grid_base)
        raw_scores = ranknet_scores(cairo, grid.reshape(-1, 1), vis_runtime.batch_size)
        stage1_display = _affine_rescale_for_display(raw_scores, true_mean)
        cairo_pred = cairo_cal.predict(raw_scores)
        huber_pred = predict_tf(huber, grid.reshape(-1, 1), vis_runtime.batch_size)

        for x_value, true_value, stage1_value, cairo_value, huber_value in zip(
            grid,
            true_mean,
            stage1_display,
            cairo_pred,
            huber_pred,
        ):
            rows.append(
                {
                    "NoiseFamily": noise_family,
                    "Severity": severity,
                    "x": float(x_value),
                    "TrueMean": float(true_value),
                    "Stage1Display": float(stage1_value),
                    "CAIROG1": float(cairo_value),
                    "NNHuber": float(huber_value),
                }
            )
        for x_value, y_value in zip(x, y):
            sample_rows.append(
                {
                    "NoiseFamily": noise_family,
                    "Severity": severity,
                    "x": float(x_value),
                    "y": float(y_value),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def plot_1d_visualization(curves: pd.DataFrame, samples: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if curves.empty or samples.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.1), sharex="col")
    families = ("Additive", "Multiplicative")
    for col_idx, noise_family in enumerate(families):
        curve_df = curves[curves["NoiseFamily"] == noise_family].sort_values("x")
        sample_df = samples[samples["NoiseFamily"] == noise_family]

        ax_pred = axes[0][col_idx]
        ax_pred.scatter(
            sample_df["x"],
            sample_df["y"],
            s=8,
            alpha=0.12,
            color="#8f8f8f",
            edgecolors="none",
        )
        ax_pred.plot(curve_df["x"], curve_df["TrueMean"], color="#222222", linestyle="--", linewidth=2.0, label="True mean")
        ax_pred.plot(curve_df["x"], curve_df["CAIROG1"], color=_model_color("CAIROG1"), linewidth=2.4, label="CAIRO-G1")
        ax_pred.plot(curve_df["x"], curve_df["NNHuber"], color=_model_color("NN-Huber"), linewidth=2.2, label="NN-Huber")
        panel_values = np.concatenate(
            [
                sample_df["y"].to_numpy(dtype=float),
                curve_df["TrueMean"].to_numpy(dtype=float),
                curve_df["CAIROG1"].to_numpy(dtype=float),
                curve_df["NNHuber"].to_numpy(dtype=float),
            ]
        )
        lower = float(np.quantile(panel_values, 0.01))
        upper = float(np.quantile(panel_values, 0.99))
        pad = 0.08 * max(upper - lower, 1e-6)
        ax_pred.set_ylim(lower - pad, upper + pad)
        ax_pred.set_title(_noise_family_title(noise_family))
        ax_pred.set_ylabel("Prediction")
        ax_pred.set_facecolor("#fcfcfc")
        ax_pred.grid(True, which="major", alpha=0.16, linewidth=0.8)

        ax_stage1 = axes[1][col_idx]
        ax_stage1.plot(curve_df["x"], curve_df["TrueMean"], color="#222222", linestyle="--", linewidth=2.0, label="True mean")
        ax_stage1.plot(
            curve_df["x"],
            curve_df["Stage1Display"],
            color=_model_color("CAIROG1"),
            linewidth=2.4,
            label="CAIRO-G1 Stage 1",
        )
        ax_stage1.set_xlabel("1D feature")
        ax_stage1.set_ylabel("Display scale")
        ax_stage1.set_facecolor("#fcfcfc")
        ax_stage1.grid(True, which="major", alpha=0.16, linewidth=0.8)
        ax_stage1.text(
            0.04,
            0.95,
            "Stage 1 score is affinely rescaled for display only.",
            transform=ax_stage1.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="#555555",
        )

    for axis, tag in zip(axes.ravel(), ("A", "B", "C", "D")):
        _add_panel_tag(axis, tag)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.6,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig((plot_dir / "matched_error_1d_visualization").with_suffix(".png"), bbox_inches="tight")
    fig.savefig((plot_dir / "matched_error_1d_visualization").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_matched_error_study(config: MatchedErrorStudyConfig) -> dict[str, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    print("[Study] Matched additive vs multiplicative sweep", flush=True)
    for severity in config.severity_levels:
        run_seed_offset = int(round(severity * 100))
        for noise_family in ("Additive", "Multiplicative"):
            scenario = _scenario_for_family(noise_family, severity)
            for repeat in range(config.runtime.n_repeats):
                run_seed = config.runtime.seed + 21_000 * repeat + run_seed_offset
                rows, _ = run_one_repeat(
                    scenario,
                    config.runtime,
                    repeat,
                    run_seed,
                    selected_models=set(config.models),
                )
                for row in rows:
                    row["NoiseFamily"] = noise_family
                    row["Severity"] = severity
                    result_rows.append(row)
    runs = pd.DataFrame(result_rows)
    summary = summarize_with_ci(runs, ["NoiseFamily", "Severity", "Model"])
    curves, samples = build_1d_visualization_curves(config)
    return {
        "runs": runs,
        "summary": summary,
        "visualization_curves": curves,
        "visualization_samples": samples,
    }


def save_matched_error_study(
    config: MatchedErrorStudyConfig,
    results: dict[str, pd.DataFrame],
    skip_plots: bool = False,
) -> Path:
    ensure_dir(config.output_dir)
    write_json(
        config.output_dir / "config.json",
        {
            "runtime": {**asdict(config.runtime), "output_dir": str(config.runtime.output_dir)},
            "severity_levels": list(config.severity_levels),
            "models": list(config.models),
            "output_dir": str(config.output_dir),
            "visualization_severity": config.visualization_severity,
            "visualization_samples": config.visualization_samples,
            "shared_backbone": list(SHARED_NEURAL_HIDDEN_UNITS),
        },
    )
    write_json(config.output_dir / "availability.json", available_components())
    write_csv(config.output_dir / "matched_error_runs.csv", results["runs"])
    write_csv(config.output_dir / "matched_error_summary.csv", results["summary"])
    write_csv(config.output_dir / "matched_error_1d_curves.csv", results["visualization_curves"])
    write_csv(config.output_dir / "matched_error_1d_samples.csv", results["visualization_samples"])
    if not skip_plots:
        plot_matched_error_story(results["summary"], config.output_dir)
        plot_1d_visualization(
            results["visualization_curves"],
            results["visualization_samples"],
            config.output_dir,
        )
    return config.output_dir
