from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import skew, spearmanr

from cairo.paths import RESULTS_DIR, ensure_dir
from cairo.reporting import write_csv, write_json
from cairo.synthetic.paper_benchmark import (
    FAIR_NEURAL_MODELS,
    RuntimeConfig,
    ScenarioSpec,
    available_components,
    plot_sweep,
    run_one_repeat,
    summarize_with_ci,
)


DEFAULT_MODELS = FAIR_NEURAL_MODELS


@dataclass(frozen=True)
class PositiveSkewStudyConfig:
    runtime: RuntimeConfig
    severity_levels: tuple[float, ...]
    models: tuple[str, ...] = DEFAULT_MODELS
    output_dir: Path = RESULTS_DIR / "benchmarks" / "positive_skew_study"

    @classmethod
    def from_mode(
        cls,
        mode: str,
        output_dir: Path | None = None,
        repeats: int | None = None,
        severity_levels: tuple[float, ...] | None = None,
    ) -> "PositiveSkewStudyConfig":
        out_dir = output_dir or RESULTS_DIR / "benchmarks" / "positive_skew_study"
        runtime = RuntimeConfig.from_mode(mode, output_dir=out_dir, repeats=repeats)
        default_levels = {
            "smoke": (0.0, 0.6, 1.2),
            "debug": (0.0, 0.4, 0.8, 1.2, 1.6),
            "full": (0.0, 0.4, 0.8, 1.2, 1.6, 2.0),
        }
        return cls(
            runtime=runtime,
            severity_levels=severity_levels or default_levels[mode],
            output_dir=out_dir,
        )


def generate_positive_skew_dataset(
    seed: int,
    runtime: RuntimeConfig,
    severity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(runtime.n_samples, 10))
    w = rr.normal(size=10)
    latent = (
        x @ w / np.sqrt(10)
        + 0.55 * np.sin(x[:, 0] + 0.5 * x[:, 1])
        + 0.25 * x[:, 2] * x[:, 3]
        - 0.20 * np.cos(x[:, 4])
    )
    latent = (latent - latent.mean()) / (latent.std() + 1e-8)

    # Severity increases tail-heaviness and amplifies multiplicative, mean-dependent noise.
    link_scale = 0.70 + 0.25 * severity
    mu = np.exp(link_scale * latent)
    sigma = 0.10 + 0.22 * severity
    multiplicative = rr.lognormal(
        mean=-0.5 * sigma**2,
        sigma=sigma,
        size=runtime.n_samples,
    )
    tail_boost = np.exp(0.12 * severity * np.maximum(latent, 0.0))
    y = mu * multiplicative * tail_boost
    return x.astype(np.float32), y.astype(np.float32), mu.astype(np.float32)


def _scenario_for_severity(severity: float) -> ScenarioSpec:
    return ScenarioSpec(
        name=f"Positive-skew heteroskedastic (s={severity:.1f})",
        kind="positive_skew_heteroskedastic",
        generator=lambda seed, runtime, value=severity: generate_positive_skew_dataset(seed, runtime, value)[:2],
    )


def _profile_row(severity: float, repeat: int, mu: np.ndarray, y: np.ndarray) -> dict[str, float]:
    median = float(np.median(y))
    tail_ratio = float(np.quantile(y, 0.95) / max(median, 1e-8))
    hetero_corr = spearmanr(mu, np.abs(y - mu)).correlation
    return {
        "Severity": severity,
        "Repeat": repeat,
        "TargetSkew": float(skew(y, bias=False)),
        "TailRatio95to50": tail_ratio,
        "HeteroCorr": float(hetero_corr) if hetero_corr is not None else np.nan,
    }


def _summarize_profile(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ("TargetSkew", "TailRatio95to50", "HeteroCorr")
    rows: list[dict[str, float]] = []
    for severity, group in frame.groupby("Severity", dropna=False):
        row: dict[str, float] = {"Severity": severity}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values)) if values.size else np.nan
            if values.size <= 1:
                row[f"{metric}_ci"] = np.nan
            else:
                row[f"{metric}_ci"] = float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Severity")


def _plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
        }
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


def _story_color(model_name: str) -> str:
    palette = {
        "CAIROK1": "#0E766E",
        "CAIROG1": "#C65D00",
        "CAIROS1": "#8F6B00",
        "CAIROG2": "#5B4CC4",
        "NN-Huber": "#B63A6B",
        "NN-MSE": "#3C4F76",
        "NN-MAE": "#7A7F87",
    }
    return palette.get(model_name, "#555555")


def _add_panel_tag(axis: Any, tag: str) -> None:
    axis.text(
        -0.15,
        1.08,
        tag,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")


def plot_profile(profile_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if profile_summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    specs = (
        ("TargetSkew", "Sample skewness"),
        ("TailRatio95to50", "P95 / median"),
        ("HeteroCorr", "corr($\\mu$, |$y-\\mu$|)"),
    )
    for axis, (metric, ylabel) in zip(axes, specs):
        axis.errorbar(
            profile_summary["Severity"],
            profile_summary[f"{metric}_mean"],
            yerr=np.nan_to_num(profile_summary[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
            marker="o",
            linewidth=1.8,
            capsize=3,
            color="#444444",
        )
        axis.set_title(ylabel)
        axis.set_xlabel("Severity")
        axis.set_ylabel(ylabel)
    _save_figure(fig, plot_dir / "positive_skew_profile")
    plt.close(fig)


def plot_positive_skew_story(summary: pd.DataFrame, profile_summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty or profile_summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    selected = [m for m in DEFAULT_MODELS if m in set(summary["Model"])]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.4))

    profile_specs = (
        ("TargetSkew", "Target skew", "Sample skewness"),
        ("TailRatio95to50", "Tail ratio", "P95 / median"),
        ("HeteroCorr", "Heteroskedasticity", "corr($\\mu$, |$y-\\mu$|)"),
    )
    for axis, (metric, title, ylabel) in zip(axes[0], profile_specs):
        axis.errorbar(
            profile_summary["Severity"],
            profile_summary[f"{metric}_mean"],
            yerr=np.nan_to_num(profile_summary[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
            marker="o",
            linewidth=2.0,
            capsize=3,
            color="#444444",
        )
        axis.set_title(title)
        axis.set_xlabel("Severity")
        axis.set_ylabel(ylabel)
        axis.set_facecolor("#fcfcfc")
        axis.grid(True, which="major", alpha=0.16, linewidth=0.8)

    metric_specs = (
        ("PairAcc", "Pair accuracy"),
        ("RMSE", "Test RMSE"),
        ("CalGap", "Calibration gap"),
    )
    for axis, (metric, title) in zip(axes[1], metric_specs):
        for model_name in selected:
            subset = summary[summary["Model"] == model_name].sort_values("Severity")
            axis.errorbar(
                subset["Severity"],
                subset[f"{metric}_mean"],
                yerr=np.nan_to_num(subset[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
                marker="o",
                linewidth=2.0 if model_name.startswith("CAIRO") else 1.8,
                capsize=3,
                color=_story_color(model_name),
                label=_story_label(model_name),
                alpha=1.0 if model_name.startswith("CAIRO") else 0.92,
            )
        axis.set_title(title)
        axis.set_xlabel("Severity")
        axis.set_ylabel(title)
        axis.set_facecolor("#fcfcfc")
        axis.grid(True, which="major", alpha=0.16, linewidth=0.8)

    for axis, tag in zip(axes.ravel(), ("A", "B", "C", "D", "E", "F")):
        _add_panel_tag(axis, tag)

    handles, labels = axes[1][-1].get_legend_handles_labels()
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
    fig.savefig((plot_dir / "positive_skew_story").with_suffix(".png"), bbox_inches="tight")
    fig.savefig((plot_dir / "positive_skew_story").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def run_positive_skew_study(config: PositiveSkewStudyConfig) -> dict[str, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, float]] = []
    print("[Study] Positive-skew heteroskedastic severity sweep", flush=True)
    for severity in config.severity_levels:
        scenario = _scenario_for_severity(severity)
        for repeat in range(config.runtime.n_repeats):
            run_seed = config.runtime.seed + 19000 * repeat + int(severity * 100)
            x, y, mu = generate_positive_skew_dataset(run_seed, config.runtime, severity)
            profile_rows.append(_profile_row(severity, repeat, mu, y))
            rows, _ = run_one_repeat(
                scenario,
                config.runtime,
                repeat,
                run_seed,
                selected_models=set(config.models),
            )
            for row in rows:
                row["Severity"] = severity
                result_rows.append(row)
    runs = pd.DataFrame(result_rows)
    summary = summarize_with_ci(runs, ["Severity", "Model"])
    profile = pd.DataFrame(profile_rows)
    profile_summary = _summarize_profile(profile)
    return {
        "runs": runs,
        "summary": summary,
        "profile": profile,
        "profile_summary": profile_summary,
    }


def save_positive_skew_study(
    config: PositiveSkewStudyConfig,
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
        },
    )
    write_json(config.output_dir / "availability.json", available_components())
    write_csv(config.output_dir / "positive_skew_runs.csv", results["runs"])
    write_csv(config.output_dir / "positive_skew_summary.csv", results["summary"])
    write_csv(config.output_dir / "positive_skew_profile.csv", results["profile"])
    write_csv(config.output_dir / "positive_skew_profile_summary.csv", results["profile_summary"])
    if not skip_plots:
        plot_sweep(
            results["summary"],
            config.output_dir,
            "positive_skew_sweep",
            "Severity",
        )
        plot_profile(results["profile_summary"], config.output_dir)
        plot_positive_skew_story(results["summary"], results["profile_summary"], config.output_dir)
    return config.output_dir
