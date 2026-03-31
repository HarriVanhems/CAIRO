from __future__ import annotations

import json
import math
import os
import random
import copy
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split

from cairo.paths import RESULTS_DIR, ensure_dir
from cairo.reporting import write_csv, write_json


def _configure_runtime_env() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(os.getenv("TMPDIR", "/tmp")) / "cairo-mpl-cache"),
    )


_configure_runtime_env()


MetricDict = dict[str, float]

METRICS = ["PairAcc", "Spearman", "Kendall", "RMSE", "MAE", "CalGap"]
PAIR_SAMPLERS = ("uniform", "hard_gap", "stratified")
MAIN_BENCHMARK_MODELS = (
    "CAIROK1",
    "CAIROG1",
    "CAIROS1",
    "CAIROG2",
    "CAIRO-XGBPairwise",
    "NN-Huber",
    "QuantileRegressor",
    "NN-MSE",
    "NN-MAE",
    "LightGBM-MSE",
    "XGBoost-MSE",
    "CatBoost-RMSE",
)
PRIMARY_PLOT_MODELS = (
    "CAIROK1",
    "CAIROG1",
    "CAIROS1",
    "CAIROG2",
    "CAIRO-XGBPairwise",
    "NN-Huber",
    "QuantileRegressor",
    "NN-MSE",
    "NN-MAE",
    "CatBoost-RMSE",
    "LightGBM-MSE",
)
FAIR_NEURAL_MODELS = (
    "CAIROK1",
    "CAIROG1",
    "CAIROS1",
    "CAIROG2",
    "NN-Huber",
    "NN-MSE",
    "NN-MAE",
)
STAGE2_PLOT_MODELS = (
    "CAIROK1-Stage1",
    "CAIROK1",
    "CAIROG1-Stage1",
    "CAIROG1",
    "CAIROS1-Stage1",
    "CAIROS1",
    "CAIROG2-Stage1",
    "CAIROG2",
    "CAIRO-XGBPairwise-Stage1",
    "CAIRO-XGBPairwise",
)
FAIR_NEURAL_STAGE2_MODELS = (
    "CAIROK1-Stage1",
    "CAIROK1",
    "CAIROG1-Stage1",
    "CAIROG1",
    "CAIROS1-Stage1",
    "CAIROS1",
    "CAIROG2-Stage1",
    "CAIROG2",
    "NN-Huber",
    "NN-MSE",
    "NN-MAE",
)
SHARED_NEURAL_HIDDEN_UNITS = (32, 16)
SHARED_NEURAL_LEAKY_SLOPE = 0.2
GINI_SOFT_RANK_TAU = 0.01
GINI_BATCH_SIZE_CAP = 64
GINI_EPOCH_MULTIPLIER = 2.0
GINI_PATIENCE_MULTIPLIER = 1.5
MONOTONE_STORY_FAMILIES = ("CAIROK1", "CAIROG1", "CAIROS1", "CAIROG2")
MONOTONE_STORY_FINAL_MODELS = FAIR_NEURAL_MODELS


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    seed: int
    n_repeats: int
    n_samples: int
    epochs_ranknet: int
    epochs_gini: int
    batch_size: int
    pairs_train: int
    pairs_val: int
    pair_accuracy_samples: int
    pair_budgets: tuple[int, ...]
    contamination_levels: tuple[float, ...]
    student_t_dfs: tuple[float, ...]
    monotone_strengths: tuple[float, ...]
    low_data_sizes: tuple[int, ...]
    huber_delta_multipliers: tuple[float, ...]
    output_dir: Path

    @classmethod
    def from_mode(
        cls,
        mode: str,
        output_dir: Path | None = None,
        repeats: int | None = None,
    ) -> "RuntimeConfig":
        if mode not in {"smoke", "debug", "full"}:
            raise ValueError(f"Unsupported mode: {mode}")

        base = {
            "smoke": dict(
                seed=2025,
                n_repeats=1,
                n_samples=1200,
                epochs_ranknet=10,
                epochs_gini=10,
                batch_size=256,
                pairs_train=8_000,
                pairs_val=2_000,
                pair_accuracy_samples=8_000,
                pair_budgets=(4_000, 8_000),
                contamination_levels=(0.0, 0.10),
                student_t_dfs=(2.5, 5.0),
                monotone_strengths=(0.0, 0.8, 1.6),
                low_data_sizes=(120, 300, 600, 1200),
                huber_delta_multipliers=(0.5, 1.0, 2.0),
            ),
            "debug": dict(
                seed=2025,
                n_repeats=1,
                n_samples=2500,
                epochs_ranknet=30,
                epochs_gini=30,
                batch_size=384,
                pairs_train=40_000,
                pairs_val=10_000,
                pair_accuracy_samples=40_000,
                pair_budgets=(20_000, 40_000),
                contamination_levels=(0.0, 0.05, 0.10, 0.20),
                student_t_dfs=(2.0, 2.5, 3.0, 5.0, 10.0),
                monotone_strengths=(0.0, 0.4, 0.8, 1.2, 1.6),
                low_data_sizes=(150, 300, 600, 1200, 2500),
                huber_delta_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
            ),
            "full": dict(
                seed=2025,
                n_repeats=5,
                n_samples=6000,
                epochs_ranknet=80,
                epochs_gini=80,
                batch_size=1024,
                pairs_train=250_000,
                pairs_val=60_000,
                pair_accuracy_samples=200_000,
                pair_budgets=(70_000, 250_000),
                contamination_levels=(0.0, 0.05, 0.10, 0.20, 0.30),
                student_t_dfs=(2.0, 2.5, 3.0, 5.0, 10.0),
                monotone_strengths=(0.0, 0.4, 0.8, 1.2, 1.6, 2.0),
                low_data_sizes=(200, 500, 1000, 2500, 5000),
                huber_delta_multipliers=(0.25, 0.5, 1.0, 2.0, 4.0),
            ),
        }[mode]
        if repeats is not None:
            base["n_repeats"] = repeats
        return cls(
            mode=mode,
            output_dir=output_dir or RESULTS_DIR / "benchmarks" / "paper_synthetic",
            **base,
        )


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    kind: str
    generator: Callable[[int, RuntimeConfig], tuple[np.ndarray, np.ndarray]]
    primary: bool = True


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def pairwise_accuracy(y: np.ndarray, s: np.ndarray, m: int, seed: int) -> float:
    rr = default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    n = len(y)
    i = rr.integers(0, n, m)
    j = rr.integers(0, n, m)
    mask = i != j
    i, j = i[mask], j[mask]
    dy = y[i] - y[j]
    ds = s[i] - s[j]
    valid = (dy != 0) & (ds != 0)
    if valid.sum() == 0:
        return np.nan
    dy = dy[valid]
    ds = ds[valid]
    return float(np.mean((dy > 0) == (ds > 0)))


def _safe_spearman(y: np.ndarray, s: np.ndarray) -> float:
    try:
        rho, _ = spearmanr(y, s)
        return float(rho)
    except Exception:
        return np.nan


def _safe_kendall(y: np.ndarray, s: np.ndarray) -> float:
    try:
        tau, _ = kendalltau(y, s)
        return float(tau)
    except Exception:
        return np.nan


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.mean(np.abs(y - yhat)))


def calibration_gap_quantile_bins(y: np.ndarray, yhat: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if np.allclose(np.std(yhat), 0):
        return np.nan
    quantiles = np.quantile(yhat, np.linspace(0, 1, n_bins + 1))
    quantiles[-1] = np.nextafter(quantiles[-1], np.inf)
    gaps = []
    for idx in range(n_bins):
        mask = (yhat >= quantiles[idx]) & (yhat < quantiles[idx + 1])
        if mask.sum() < 5:
            continue
        gaps.append(abs(y[mask].mean() - yhat[mask].mean()))
    return float(np.mean(gaps)) if gaps else np.nan


def reliability_curve_quantile_bins(
    y: np.ndarray,
    yhat: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    quantiles = np.quantile(yhat, np.linspace(0, 1, n_bins + 1))
    quantiles[-1] = np.nextafter(quantiles[-1], np.inf)
    rows = []
    for idx in range(n_bins):
        mask = (yhat >= quantiles[idx]) & (yhat < quantiles[idx + 1])
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin": idx,
                "pred_mean": float(yhat[mask].mean()),
                "obs_mean": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def pack_metrics(y: np.ndarray, yhat: np.ndarray, cfg: RuntimeConfig, seed: int) -> MetricDict:
    return {
        "PairAcc": pairwise_accuracy(y, yhat, m=cfg.pair_accuracy_samples, seed=seed + 7),
        "Spearman": _safe_spearman(y, yhat),
        "Kendall": _safe_kendall(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAE": mae(y, yhat),
        "CalGap": calibration_gap_quantile_bins(y, yhat, n_bins=10),
    }


def gen_normal_regression(seed: int, cfg: RuntimeConfig, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w = rr.normal(size=10)
    z = x @ w / np.sqrt(10)
    y = z + rr.normal(scale=sigma, size=cfg.n_samples)
    return x.astype(np.float32), y.astype(np.float32)


def gen_nonlinear_index_heavytail(seed: int, cfg: RuntimeConfig, hidden: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w1 = rr.normal(size=(10, hidden)) / np.sqrt(10)
    b1 = rr.normal(size=(hidden,)) * 0.1
    w2 = rr.normal(size=(hidden,)) / np.sqrt(hidden)
    hidden_state = np.maximum(0.0, x @ w1 + b1)
    z = np.tanh(hidden_state @ w2)
    mu = np.exp(z)
    eps = rr.lognormal(mean=0.0, sigma=0.8, size=cfg.n_samples)
    y = mu + np.sqrt(mu) * eps
    return x.astype(np.float32), y.astype(np.float32)


def gen_gamma_regression(seed: int, cfg: RuntimeConfig, shape: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w = rr.normal(size=10)
    mu = np.exp(x @ w / np.sqrt(10))
    y = rr.gamma(shape=shape, scale=mu / shape)
    return x.astype(np.float32), y.astype(np.float32)


def gen_heteroskedastic_lognormal(seed: int, cfg: RuntimeConfig) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w = rr.normal(size=10)
    mu = np.exp(x @ w / np.sqrt(10))
    noise = rr.lognormal(mean=0.0, sigma=0.7, size=cfg.n_samples) * (mu**0.5)
    y = mu + noise
    return x.astype(np.float32), y.astype(np.float32)


def gen_student_t_heavytail(
    seed: int,
    cfg: RuntimeConfig,
    df: float = 2.5,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w = rr.normal(size=10)
    z = x @ w / np.sqrt(10)
    eps = rr.standard_t(df=df, size=cfg.n_samples) * sigma
    y = z + eps
    return x.astype(np.float32), y.astype(np.float32)


def gen_label_contamination(
    seed: int,
    cfg: RuntimeConfig,
    eps: float = 0.1,
    out_scale: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w = rr.normal(size=10)
    z = x @ w / np.sqrt(10)
    y = z + rr.normal(scale=1.0, size=cfg.n_samples)
    mask = rr.uniform(size=cfg.n_samples) < eps
    y[mask] = y[mask] + rr.normal(scale=out_scale, size=mask.sum())
    return x.astype(np.float32), y.astype(np.float32)


def gen_monotone_distortion(
    seed: int,
    cfg: RuntimeConfig,
    strength: float = 1.0,
    noise_scale: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    rr = default_rng(seed)
    x = rr.normal(size=(cfg.n_samples, 10))
    w1 = rr.normal(size=(10, 24)) / np.sqrt(10)
    b1 = rr.normal(size=(24,)) * 0.15
    w2 = rr.normal(size=(24,)) / np.sqrt(24)
    latent = np.maximum(0.0, x @ w1 + b1) @ w2
    latent = (latent - latent.mean()) / (latent.std() + 1e-8)
    monotone_map = latent + strength * (latent**3) / 3.0 + 0.35 * strength * np.tanh(2.0 * latent)
    noise = rr.normal(scale=noise_scale * (1.0 + 0.3 * np.abs(latent)), size=cfg.n_samples)
    y = monotone_map + noise
    return x.astype(np.float32), y.astype(np.float32)


def build_main_scenarios() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("Normal (σ=1.0)", "main", lambda seed, cfg: gen_normal_regression(seed, cfg)),
        ScenarioSpec("Gamma (shape=2.0)", "main", lambda seed, cfg: gen_gamma_regression(seed, cfg)),
        ScenarioSpec(
            "Heteroskedastic lognormal",
            "main",
            lambda seed, cfg: gen_heteroskedastic_lognormal(seed, cfg),
        ),
        ScenarioSpec(
            "Student-t heavy tail (df=2.5)",
            "main",
            lambda seed, cfg: gen_student_t_heavytail(seed, cfg, df=2.5),
        ),
        ScenarioSpec(
            "Label contamination (10%)",
            "main",
            lambda seed, cfg: gen_label_contamination(seed, cfg, eps=0.10),
        ),
        ScenarioSpec(
            "Nonlinear-index + heavy tail",
            "main",
            lambda seed, cfg: gen_nonlinear_index_heavytail(seed, cfg),
        ),
    ]


def make_experiment_splits(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    calibration_frac: float = 0.20,
    validation_frac: float = 0.20,
) -> dict[str, np.ndarray]:
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.3, random_state=seed
    )
    x_model, x_cal, y_model, y_cal = train_test_split(
        x_train_full,
        y_train_full,
        test_size=calibration_frac,
        random_state=seed + 1,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_model,
        y_model,
        test_size=validation_frac,
        random_state=seed + 2,
    )
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_model": x_model,
        "y_model": y_model,
        "x_cal": x_cal,
        "y_cal": y_cal,
        "x_test": x_test,
        "y_test": y_test,
    }


def _import_tf_stack() -> tuple[Any, Any, Any, Any]:
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers
    from tensorflow.nn import leaky_relu

    tf.get_logger().setLevel("ERROR")
    return tf, layers, Model, Input, leaky_relu


def _build_shared_mlp_backbone_tf(layers: Any, inputs: Any) -> Any:
    x = inputs
    for units in SHARED_NEURAL_HIDDEN_UNITS:
        x = layers.Dense(units)(x)
        x = layers.LeakyReLU(negative_slope=SHARED_NEURAL_LEAKY_SLOPE)(x)
    return x


def _import_torch_stack() -> tuple[Any, Any, Any, Any]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from fast_soft_sort.pytorch_ops import soft_rank

    return torch, nn, optim, soft_rank


def _import_lightgbm_module() -> Any:
    import lightgbm as lgb

    return lgb


def _import_xgboost_ranker() -> Any:
    from xgboost import XGBRanker

    return XGBRanker


def _load_optional_baseline_apis() -> dict[str, Any]:
    apis: dict[str, Any] = {}
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        apis["lightgbm"] = None
    else:
        apis["lightgbm"] = LGBMRegressor

    try:
        from xgboost import XGBRegressor
    except Exception:
        apis["xgboost"] = None
    else:
        apis["xgboost"] = XGBRegressor

    try:
        from catboost import CatBoostRegressor
    except Exception:
        apis["catboost"] = None
    else:
        apis["catboost"] = CatBoostRegressor
    return apis


def available_components() -> dict[str, bool]:
    availability = {
        "tensorflow": True,
        "torch": True,
        "fast_soft_sort": True,
        "lightgbm": True,
        "xgboost": True,
        "catboost": True,
    }
    try:
        _import_tf_stack()
    except Exception:
        availability["tensorflow"] = False
    try:
        _import_torch_stack()
    except Exception:
        availability["torch"] = False
        availability["fast_soft_sort"] = False
    try:
        _import_lightgbm_module()
    except Exception:
        availability["lightgbm"] = False
    try:
        _import_xgboost_ranker()
    except Exception:
        availability["xgboost"] = False
    optional = _load_optional_baseline_apis()
    for key in ("lightgbm", "catboost"):
        availability[key] = optional[key] is not None
    availability["xgboost"] = availability["xgboost"] and optional["xgboost"] is not None
    return availability


def _dense_rank_targets(y: np.ndarray) -> np.ndarray:
    # Preserve ties so the tree ranker receives an ordinal supervision signal
    # aligned with RankNet's pairwise ordering objective.
    return rankdata(y, method="dense").astype(np.float32) - 1.0


def _sample_pairs_uniform(y: np.ndarray, num_pairs: int, rr: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    i = rr.integers(0, n, num_pairs)
    j = rr.integers(0, n, num_pairs)
    mask = i != j
    return i[mask], j[mask]


def _sample_pairs_hard_gap(
    y: np.ndarray,
    num_pairs: int,
    rr: np.random.Generator,
    oversample_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    i, j = _sample_pairs_uniform(y, int(num_pairs * oversample_factor), rr)
    gaps = np.abs(y[i] - y[j])
    if len(gaps) <= num_pairs:
        return i, j
    top = np.argpartition(gaps, -num_pairs)[-num_pairs:]
    return i[top], j[top]


def _sample_pairs_stratified_quantile(
    y: np.ndarray,
    num_pairs: int,
    rr: np.random.Generator,
    n_bins: int = 10,
    min_bin_gap: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    quantiles = np.quantile(y, np.linspace(0, 1, n_bins + 1))
    quantiles[-1] = np.nextafter(quantiles[-1], np.inf)
    bins = [np.where((y >= quantiles[k]) & (y < quantiles[k + 1]))[0] for k in range(n_bins)]
    i_list: list[int] = []
    j_list: list[int] = []
    tries = 0
    while len(i_list) < num_pairs and tries < num_pairs * 20:
        a = rr.integers(0, n_bins)
        b = rr.integers(0, n_bins)
        if abs(a - b) < min_bin_gap or len(bins[a]) == 0 or len(bins[b]) == 0:
            tries += 1
            continue
        i_choice = int(rr.choice(bins[a]))
        j_choice = int(rr.choice(bins[b]))
        if i_choice != j_choice:
            i_list.append(i_choice)
            j_list.append(j_choice)
        tries += 1
    if len(i_list) < 2:
        return _sample_pairs_uniform(y, num_pairs, rr)
    return np.asarray(i_list, dtype=int), np.asarray(j_list, dtype=int)


def _empirical_cdf(y: np.ndarray) -> np.ndarray:
    return rankdata(y, method="average").astype(np.float32) / float(len(y))


def make_pairs(
    x: np.ndarray,
    y: np.ndarray,
    num_pairs: int,
    seed: int,
    sampler: str = "uniform",
    weight_mode: str = "uniform",
    debug: bool = False,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray | None]:
    rr = default_rng(seed)
    if sampler == "uniform":
        i, j = _sample_pairs_uniform(y, num_pairs, rr)
    elif sampler == "hard_gap":
        i, j = _sample_pairs_hard_gap(y, num_pairs, rr, oversample_factor=5 if debug else 8)
    elif sampler == "stratified":
        i, j = _sample_pairs_stratified_quantile(y, num_pairs, rr)
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    xi, xj = x[i], x[j]
    yi, yj = y[i], y[j]
    targets = (yi > yj).astype(np.float32).reshape(-1, 1)
    if weight_mode == "uniform":
        return (xi, xj), targets, None
    if weight_mode == "abs_diff":
        weights = np.abs(yi - yj).astype(np.float32)
    elif weight_mode == "cdf_diff":
        cdf = _empirical_cdf(y)
        weights = np.abs(cdf[i] - cdf[j]).astype(np.float32)
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    weights = weights / (weights.mean() + 1e-8)
    return (xi, xj), targets, weights


def _logit_binary_accuracy_metric(tf: Any) -> Any:
    class LogitBinaryAccuracy(tf.keras.metrics.Metric):
        def __init__(self, name: str = "acc", **kwargs: Any) -> None:
            super().__init__(name=name, **kwargs)
            self._acc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

        def update_state(self, y_true: Any, y_pred: Any, sample_weight: Any = None) -> None:
            self._acc.update_state(y_true, tf.nn.sigmoid(y_pred), sample_weight=sample_weight)

        def result(self) -> Any:
            return self._acc.result()

        def reset_state(self) -> None:
            self._acc.reset_state()

    return LogitBinaryAccuracy


def _tf_dataset_pairs(
    tf: Any,
    pair_inputs: tuple[np.ndarray, np.ndarray],
    targets: np.ndarray,
    sample_weight: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Any:
    if sample_weight is None:
        ds = tf.data.Dataset.from_tensor_slices((pair_inputs, targets))
    else:
        ds = tf.data.Dataset.from_tensor_slices((pair_inputs, targets, sample_weight))
    if shuffle:
        ds = ds.shuffle(min(len(targets), 50_000), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _keras_training_schedule(cfg: RuntimeConfig, model_family: str) -> tuple[int, int, int]:
    base = cfg.epochs_ranknet
    family_scale = {
        "ranknet_pairwise": 1.0,
        # Pointwise neural baselines see far fewer optimizer steps per epoch
        # than the pairwise CAIRO models, so give them materially larger ceilings.
        "nn_mse": 5.0,
        "nn_mae": 6.0,
        "nn_huber": 6.5,
        "nn_logcosh": 6.5,
    }
    max_epochs = max(6, int(round(base * family_scale.get(model_family, 1.0))))
    if model_family == "ranknet_pairwise":
        patience = max(3, min(10, max_epochs // 10))
    elif model_family in {"nn_huber", "nn_logcosh"}:
        patience = max(8, min(32, max_epochs // 10))
    else:
        patience = max(7, min(28, max_epochs // 12))
    lr_patience = max(4, patience // 2)
    return max_epochs, patience, lr_patience


def _gini_training_schedule(cfg: RuntimeConfig) -> tuple[int, int]:
    max_epochs = max(8, int(round(cfg.epochs_gini * 1.2)))
    patience = max(5, min(14, max_epochs // 8))
    return max_epochs, patience


def _set_selected_epoch_metadata(model: Any, history: Any, monitor: str = "val_loss", mode: str = "min") -> None:
    values = history.history.get(monitor, [])
    if values:
        if mode == "min":
            best_epoch = int(np.argmin(values) + 1)
        else:
            best_epoch = int(np.argmax(values) + 1)
    else:
        losses = history.history.get("loss", [])
        best_epoch = int(len(losses)) if losses else 0
    setattr(model, "selected_epochs", best_epoch)
    setattr(model, "fit_epochs", int(len(history.history.get("loss", []))))


def train_ranknet_stage1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    sampler: str,
    weight_mode: str,
    seed: int,
) -> Any:
    tf, layers, Model, Input, _ = _import_tf_stack()
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    max_epochs, patience, lr_patience = _keras_training_schedule(cfg, "ranknet_pairwise")

    def build_branch(d_in: int) -> Any:
        inp = Input(shape=(d_in,))
        x = _build_shared_mlp_backbone_tf(layers, inp)
        out = layers.Dense(1, activation="linear")(x)
        return Model(inp, out, name="branch")

    d_in = x_train.shape[1]
    scorer = build_branch(d_in)
    xi = Input(shape=(d_in,), name="xi")
    xj = Input(shape=(d_in,), name="xj")
    logits = scorer(xi) - scorer(xj)
    pair_model = Model([xi, xj], logits, name="ranknet_logits")

    (xi_train, xj_train), targets, weights = make_pairs(
        x_train,
        y_train,
        cfg.pairs_train,
        seed=seed,
        sampler=sampler,
        weight_mode=weight_mode,
        debug=cfg.mode == "debug",
    )
    (xi_val, xj_val), targets_val, weights_val = make_pairs(
        x_val,
        y_val,
        cfg.pairs_val,
        seed=seed + 1,
        sampler="uniform",
        weight_mode=weight_mode,
        debug=cfg.mode == "debug",
    )

    ds_train = _tf_dataset_pairs(
        tf, (xi_train, xj_train), targets, weights, cfg.batch_size, True, seed
    )
    ds_val = _tf_dataset_pairs(
        tf, (xi_val, xj_val), targets_val, weights_val, cfg.batch_size, False, seed
    )
    pair_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    )
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-5,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
    history = pair_model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=max_epochs,
        verbose=0,
        callbacks=callbacks,
    )
    _set_selected_epoch_metadata(scorer, history, monitor="val_loss", mode="min")
    return scorer


def ranknet_scores(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(x, batch_size=batch_size, verbose=0).squeeze()


def train_xgb_pairwise_stage1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
) -> Any:
    XGBRanker = _import_xgboost_ranker()
    n_estimators_by_mode = {"smoke": 250, "debug": 600, "full": 1200}
    y_rank_train = _dense_rank_targets(y_train)

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=n_estimators_by_mode[cfg.mode],
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        tree_method="hist",
        n_jobs=1,
        verbosity=0,
    )
    model.fit(
        x_train,
        y_rank_train,
        group=[len(y_rank_train)],
        verbose=False,
    )
    return model


def xgb_pairwise_scores(model: Any, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x), dtype=np.float32).squeeze()


def train_gininet_stage1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
    tau: float = GINI_SOFT_RANK_TAU,
) -> Any:
    torch, nn, optim, soft_rank = _import_torch_stack()
    torch.manual_seed(seed)
    max_epochs, patience = _gini_training_schedule(cfg)
    max_epochs = max(12, int(round(max_epochs * GINI_EPOCH_MULTIPLIER)))
    patience = max(8, int(round(patience * GINI_PATIENCE_MULTIPLIER)))
    inner_batch_size = min(cfg.batch_size, GINI_BATCH_SIZE_CAP)

    class GiniNet(nn.Module):
        def __init__(self, d_in: int) -> None:
            super().__init__()
            blocks: list[Any] = []
            in_features = d_in
            for units in SHARED_NEURAL_HIDDEN_UNITS:
                blocks.extend(
                    [
                        nn.Linear(in_features, units),
                        nn.LeakyReLU(negative_slope=SHARED_NEURAL_LEAKY_SLOPE),
                    ]
                )
                in_features = units
            blocks.append(nn.Linear(in_features, 1))
            self.net = nn.Sequential(*blocks)

        def forward(self, batch: Any) -> Any:
            return self.net(batch).squeeze(-1)

    def gini_cov_loss(targets: Any, scores: Any) -> Any:
        ranks = soft_rank(scores.unsqueeze(0), regularization_strength=tau).squeeze(0)
        centered_y = targets - targets.mean()
        centered_y = centered_y / (targets.std(unbiased=False) + 1e-6)
        centered_r = ranks - ranks.mean()
        return -(centered_y * centered_r).mean()

    x_tensor = torch.tensor(x_train, dtype=torch.float32, device="cpu")
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device="cpu")
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32, device="cpu")
    n, d_in = x_train.shape
    model = GiniNet(d_in).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, patience // 2),
        min_lr=1e-5,
    )
    rr = default_rng(seed)
    best_metric = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    best_epoch = 0
    model.train()
    for epoch in range(max_epochs):
        idx = rr.permutation(n)
        for start in range(0, n, inner_batch_size):
            batch = idx[start : start + inner_batch_size]
            if len(batch) < 2:
                continue
            xb = x_tensor[batch]
            yb = y_tensor[batch]
            optimizer.zero_grad()
            scores = model(xb)
            loss = gini_cov_loss(yb, scores)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            val_scores = model(x_val_tensor).cpu().numpy().squeeze()
            val_metric = _safe_spearman(y_val, val_scores)
            scheduler.step(val_metric if np.isfinite(val_metric) else -1.0)
            if np.isfinite(val_metric) and val_metric > best_metric:
                best_metric = val_metric
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
                best_epoch = epoch + 1
            else:
                stale_epochs += 1
            model.train()
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    setattr(model, "selected_epochs", best_epoch)
    return model


def gininet_scores(model: Any, x: np.ndarray) -> np.ndarray:
    torch, _, _, _ = _import_torch_stack()
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device="cpu")
        return model(tensor).cpu().numpy().squeeze()


def fit_isotonic(scores_cal: np.ndarray, y_cal: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(scores_cal), np.asarray(y_cal))
    return iso


def _build_mlp_regressor_tf(
    layers: Any,
    Model: Any,
    Input: Any,
    d_in: int,
    loss_obj: Any,
    model_name: str,
) -> Any:
    inp = Input(shape=(d_in,))
    x = _build_shared_mlp_backbone_tf(layers, inp)
    out = layers.Dense(1, activation="linear")(x)
    model = Model(inp, out, name=model_name)
    model.compile(optimizer="adam", loss=loss_obj)
    return model


def _fit_mlp_regressor_tf(
    tf: Any,
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
    model_family: str,
) -> Any:
    max_epochs, patience, lr_patience = _keras_training_schedule(cfg, model_family)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(min(len(y_train), 50_000), seed=seed, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=lr_patience,
            min_lr=1e-5,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    ]
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=max_epochs,
        verbose=0,
        callbacks=callbacks,
    )
    _set_selected_epoch_metadata(model, history, monitor="val_loss", mode="min")
    return model


def _robust_target_scale(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    center = float(np.median(y))
    mad = float(np.median(np.abs(y - center)))
    iqr = float(np.subtract(*np.quantile(y, [0.75, 0.25])) / 1.349)
    std = float(np.std(y))
    return max(1.4826 * mad, iqr, 0.1 * std, 1e-3)


def _tune_huber_delta_tf(
    tf: Any,
    layers: Any,
    Model: Any,
    Input: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
) -> tuple[Any, float]:
    d_in = x_train.shape[1]
    scale = _robust_target_scale(y_train)
    candidate_deltas = sorted(
        {
            max(1e-4, scale * multiplier)
            for multiplier in cfg.huber_delta_multipliers
        }
    )
    best_model = None
    best_delta = candidate_deltas[0]
    best_key = (float("inf"), float("inf"))
    for idx, delta in enumerate(candidate_deltas):
        local_seed = seed + idx
        tf.keras.backend.clear_session()
        tf.random.set_seed(local_seed)
        model = _build_mlp_regressor_tf(
            layers,
            Model,
            Input,
            d_in,
            tf.keras.losses.Huber(delta=float(delta)),
            model_name=f"mlp_huber_delta_{idx}",
        )
        _fit_mlp_regressor_tf(
            tf,
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            local_seed,
            model_family="nn_huber",
        )
        yhat_val = model.predict(x_val, batch_size=cfg.batch_size, verbose=0).squeeze()
        key = (rmse(y_val, yhat_val), mae(y_val, yhat_val))
        if key < best_key:
            best_key = key
            best_delta = float(delta)
            best_model = model
    if best_model is None:
        raise RuntimeError("Huber delta tuning failed to produce a model.")
    setattr(best_model, "selected_huber_delta", best_delta)
    return best_model, best_delta


def train_mlp_regressor_tf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
    loss_name: str,
) -> Any:
    tf, layers, Model, Input, _ = _import_tf_stack()
    d_in = x_train.shape[1]
    if loss_name == "huber":
        model, _ = _tune_huber_delta_tf(
            tf,
            layers,
            Model,
            Input,
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            seed,
        )
        return model

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    losses = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "mae": tf.keras.losses.MeanAbsoluteError(),
        "logcosh": tf.keras.losses.LogCosh(),
    }
    model_family = {
        "mse": "nn_mse",
        "mae": "nn_mae",
        "logcosh": "nn_logcosh",
    }[loss_name]
    model = _build_mlp_regressor_tf(
        layers,
        Model,
        Input,
        d_in,
        losses[loss_name],
        model_name=f"mlp_{loss_name}",
    )
    _fit_mlp_regressor_tf(tf, model, x_train, y_train, x_val, y_val, cfg, seed, model_family=model_family)
    return model


def predict_tf(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(x, batch_size=batch_size, verbose=0).squeeze()


def _record_predictions(
    rows: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    scenario_name: str,
    repeat: int,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
) -> None:
    rows.append(
        {
            "Scenario": scenario_name,
            "Repeat": repeat,
            "Model": model_name,
            **pack_metrics(y_true, y_pred, cfg, seed=seed),
        }
    )
    if model_name in PRIMARY_PLOT_MODELS:
        reliability = reliability_curve_quantile_bins(y_true, y_pred, n_bins=10)
        if not reliability.empty:
            reliability["Scenario"] = scenario_name
            reliability["Repeat"] = repeat
            reliability["Model"] = model_name
            reliability_rows.extend(reliability.to_dict("records"))


def _log_training_metadata(model: Any, *, delta: float | None = None) -> None:
    selected_epochs = getattr(model, "selected_epochs", None)
    extras: list[str] = []
    if selected_epochs:
        extras.append(f"best epoch={selected_epochs}")
    if delta is not None:
        extras.append(f"delta={delta:.4g}")
    if extras:
        print(f"    {' | '.join(extras)}", flush=True)


def run_one_repeat(
    scenario: ScenarioSpec,
    cfg: RuntimeConfig,
    repeat: int,
    run_seed: int,
    selected_models: set[str] | None = None,
    include_uncalibrated_cairo: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    set_all_seeds(run_seed)
    x, y = scenario.generator(run_seed, cfg)
    splits = make_experiment_splits(x, y, seed=run_seed)
    x_train = splits["x_train"]
    y_train = splits["y_train"]
    x_val = splits["x_val"]
    y_val = splits["y_val"]
    x_model = splits["x_model"]
    y_model = splits["y_model"]
    x_cal = splits["x_cal"]
    y_cal = splits["y_cal"]
    x_test = splits["x_test"]
    y_test = splits["y_test"]

    availability = available_components()
    optional = _load_optional_baseline_apis()
    rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    want = lambda model_name: selected_models is None or model_name in selected_models

    print(f"[Scenario] {scenario.name} | repeat={repeat + 1} | seed={run_seed}", flush=True)

    if want("CAIROK1") or (include_uncalibrated_cairo and want("CAIROK1-Stage1")):
        print("  - Training CAIROK1", flush=True)
        scorer = train_ranknet_stage1(
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            sampler="uniform",
            weight_mode="uniform",
            seed=run_seed + 10,
        )
        raw_scores = ranknet_scores(scorer, x_test, cfg.batch_size)
        _log_training_metadata(scorer)
        if include_uncalibrated_cairo and want("CAIROK1-Stage1"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIROK1-Stage1",
                y_test,
                raw_scores,
                cfg,
                run_seed + 10,
            )
        iso = fit_isotonic(ranknet_scores(scorer, x_cal, cfg.batch_size), y_cal)
        yhat = iso.predict(raw_scores)
        if want("CAIROK1"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "CAIROK1", y_test, yhat, cfg, run_seed + 10)

    if want("CAIROG1") or (include_uncalibrated_cairo and want("CAIROG1-Stage1")):
        print("  - Training CAIROG1", flush=True)
        scorer = train_ranknet_stage1(
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            sampler="uniform",
            weight_mode="abs_diff",
            seed=run_seed + 20,
        )
        raw_scores = ranknet_scores(scorer, x_test, cfg.batch_size)
        _log_training_metadata(scorer)
        if include_uncalibrated_cairo and want("CAIROG1-Stage1"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIROG1-Stage1",
                y_test,
                raw_scores,
                cfg,
                run_seed + 20,
            )
        iso = fit_isotonic(ranknet_scores(scorer, x_cal, cfg.batch_size), y_cal)
        yhat = iso.predict(raw_scores)
        if want("CAIROG1"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "CAIROG1", y_test, yhat, cfg, run_seed + 20)

    if want("CAIROS1") or (include_uncalibrated_cairo and want("CAIROS1-Stage1")):
        print("  - Training CAIROS1", flush=True)
        scorer = train_ranknet_stage1(
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            sampler="uniform",
            weight_mode="cdf_diff",
            seed=run_seed + 25,
        )
        raw_scores = ranknet_scores(scorer, x_test, cfg.batch_size)
        _log_training_metadata(scorer)
        if include_uncalibrated_cairo and want("CAIROS1-Stage1"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIROS1-Stage1",
                y_test,
                raw_scores,
                cfg,
                run_seed + 25,
            )
        iso = fit_isotonic(ranknet_scores(scorer, x_cal, cfg.batch_size), y_cal)
        yhat = iso.predict(raw_scores)
        if want("CAIROS1"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIROS1",
                y_test,
                yhat,
                cfg,
                run_seed + 25,
            )

    if (want("CAIROG2") or (include_uncalibrated_cairo and want("CAIROG2-Stage1"))) and availability["torch"] and availability["fast_soft_sort"]:
        print("  - Training CAIROG2", flush=True)
        gininet = train_gininet_stage1(x_train, y_train, x_val, y_val, cfg, seed=run_seed + 30)
        raw_scores = gininet_scores(gininet, x_test)
        _log_training_metadata(gininet)
        if include_uncalibrated_cairo and want("CAIROG2-Stage1"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIROG2-Stage1",
                y_test,
                raw_scores,
                cfg,
                run_seed + 30,
            )
        iso = fit_isotonic(gininet_scores(gininet, x_cal), y_cal)
        yhat = iso.predict(raw_scores)
        if want("CAIROG2"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIROG2",
                y_test,
                yhat,
                cfg,
                run_seed + 30,
            )

    if (want("CAIRO-XGBPairwise") or (include_uncalibrated_cairo and want("CAIRO-XGBPairwise-Stage1"))) and availability["xgboost"]:
        print("  - Training CAIRO-XGBPairwise", flush=True)
        ranker = train_xgb_pairwise_stage1(x_train, y_train, x_val, y_val, cfg, seed=run_seed + 35)
        raw_scores = xgb_pairwise_scores(ranker, x_test)
        if include_uncalibrated_cairo and want("CAIRO-XGBPairwise-Stage1"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIRO-XGBPairwise-Stage1",
                y_test,
                raw_scores,
                cfg,
                run_seed + 35,
            )
        iso = fit_isotonic(xgb_pairwise_scores(ranker, x_cal), y_cal)
        yhat = iso.predict(raw_scores)
        if want("CAIRO-XGBPairwise"):
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CAIRO-XGBPairwise",
                y_test,
                yhat,
                cfg,
                run_seed + 35,
            )

    baseline_specs = {
        "NN-MSE": "mse",
        "NN-MAE": "mae",
        "NN-Huber": "huber",
        "NN-LogCosh": "logcosh",
    }
    for model_name, loss_name in baseline_specs.items():
        if not want(model_name):
            continue
        print(f"  - Training {model_name}", flush=True)
        model = train_mlp_regressor_tf(
            x_train,
            y_train,
            x_val,
            y_val,
            cfg,
            seed=run_seed + 100,
            loss_name=loss_name,
        )
        _log_training_metadata(model, delta=getattr(model, "selected_huber_delta", None))
        yhat = predict_tf(model, x_test, cfg.batch_size)
        _record_predictions(rows, reliability_rows, scenario.name, repeat, model_name, y_test, yhat, cfg, run_seed + 100)

    if want("RandomForest"):
        print("  - Training RandomForest", flush=True)
        rf = RandomForestRegressor(
            n_estimators=120 if cfg.mode == "debug" else 300,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=run_seed,
        )
        rf.fit(x_model, y_model)
        yhat = rf.predict(x_test)
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "RandomForest", y_test, yhat, cfg, run_seed + 200)

    if want("QuantileRegressor"):
        print("  - Training QuantileRegressor", flush=True)
        quantile = QuantileRegressor(quantile=0.5, alpha=1e-4, solver="highs")
        quantile.fit(x_model.astype(np.float64), y_model.astype(np.float64))
        yhat = quantile.predict(x_test.astype(np.float64))
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "QuantileRegressor", y_test, yhat, cfg, run_seed + 215)

    if optional["lightgbm"] is not None and want("LightGBM-MSE"):
        print("  - Training LightGBM-MSE", flush=True)
        model = optional["lightgbm"](
            n_estimators=120 if cfg.mode == "smoke" else (200 if cfg.mode == "debug" else 500),
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=run_seed,
            objective="regression",
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(x_model, y_model)
        yhat = model.predict(x_test)
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "LightGBM-MSE", y_test, yhat, cfg, run_seed + 220)

    if optional["xgboost"] is not None and want("XGBoost-MSE"):
        print("  - Training XGBoost-MSE", flush=True)
        model = optional["xgboost"](
            n_estimators=150 if cfg.mode == "smoke" else (250 if cfg.mode == "debug" else 800),
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=run_seed,
            objective="reg:squarederror",
            n_jobs=1,
            verbosity=0,
        )
        model.fit(x_model, y_model)
        yhat = model.predict(x_test)
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "XGBoost-MSE", y_test, yhat, cfg, run_seed + 230)

    if optional["catboost"] is not None and want("CatBoost-RMSE"):
        print("  - Training CatBoost-RMSE", flush=True)
        model = optional["catboost"](
            iterations=200 if cfg.mode == "smoke" else (400 if cfg.mode == "debug" else 2000),
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=run_seed,
            verbose=False,
            thread_count=1,
        )
        model.fit(x_model, y_model)
        yhat = model.predict(x_test)
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "CatBoost-RMSE", y_test, yhat, cfg, run_seed + 240)
    return rows, reliability_rows


def run_pair_sampling_ablation(
    scenario: ScenarioSpec,
    cfg: RuntimeConfig,
    repeat: int,
    run_seed: int,
) -> list[dict[str, Any]]:
    set_all_seeds(run_seed)
    x, y = scenario.generator(run_seed, cfg)
    splits = make_experiment_splits(x, y, seed=run_seed)
    x_train = splits["x_train"]
    y_train = splits["y_train"]
    x_val = splits["x_val"]
    y_val = splits["y_val"]
    x_cal = splits["x_cal"]
    y_cal = splits["y_cal"]
    x_test = splits["x_test"]
    y_test = splits["y_test"]
    rows = []
    print(f"[Ablation] {scenario.name}", flush=True)
    for sampler in PAIR_SAMPLERS:
        for budget in cfg.pair_budgets:
            print(f"  - sampler={sampler}, pair_budget={budget}", flush=True)
            scorer = train_ranknet_stage1(
                x_train,
                y_train,
                x_val,
                y_val,
                replace(cfg, pairs_train=budget, pairs_val=min(cfg.pairs_val, max(5_000, budget // 5))),
                sampler=sampler,
                weight_mode="uniform",
                seed=run_seed + 500,
            )
            iso = fit_isotonic(ranknet_scores(scorer, x_cal, cfg.batch_size), y_cal)
            yhat = iso.predict(ranknet_scores(scorer, x_test, cfg.batch_size))
            rows.append(
                {
                    "Scenario": scenario.name,
                    "Repeat": repeat,
                    "Sampler": sampler,
                    "PairBudget": budget,
                    **pack_metrics(y_test, yhat, cfg, seed=run_seed + 500),
                }
            )
    return rows


def summarize_with_ci(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = {column: value for column, value in zip(group_cols, key_tuple)}
        for metric in METRICS:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_ci"] = np.nan
                continue
            row[f"{metric}_mean"] = float(np.mean(values))
            if values.size <= 1:
                row[f"{metric}_ci"] = np.nan
            else:
                row[f"{metric}_ci"] = float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_overall_ranks(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ("PairAcc", "Spearman", "Kendall"):
        pivot = frame.pivot_table(index="Scenario", columns="Model", values=metric, aggfunc="mean")
        ranks = pivot.rank(axis=1, ascending=False, method="average")
        mean_ranks = ranks.mean(axis=0).sort_values()
        for model, value in mean_ranks.items():
            rows.append({"Metric": metric, "Model": model, "AverageRank": float(value)})
    for metric in ("RMSE", "MAE", "CalGap"):
        pivot = frame.pivot_table(index="Scenario", columns="Model", values=metric, aggfunc="mean")
        ranks = pivot.rank(axis=1, ascending=True, method="average")
        mean_ranks = ranks.mean(axis=0).sort_values()
        for model, value in mean_ranks.items():
            rows.append({"Metric": metric, "Model": model, "AverageRank": float(value)})
    return pd.DataFrame(rows)


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


def _model_color(model_name: str) -> str:
    palette = {
        "CAIROK1": "#1b9e77",
        "CAIROG1": "#d95f02",
        "CAIROS1": "#a6761d",
        "CAIROG2": "#7570b3",
        "CAIRO-XGBPairwise": "#e6ab02",
        "CAIROK1-Stage1": "#76c7a2",
        "CAIROG1-Stage1": "#f4a261",
        "CAIROS1-Stage1": "#d8b365",
        "CAIROG2-Stage1": "#a4a3d8",
        "CAIRO-XGBPairwise-Stage1": "#f3d46b",
        "NN-Huber": "#e7298a",
        "QuantileRegressor": "#b15928",
        "NN-MSE": "#fb9a99",
        "NN-MAE": "#ff7f00",
        "RandomForest": "#66a61e",
        "LightGBM-MSE": "#1f78b4",
        "CatBoost-RMSE": "#4daf4a",
        "XGBoost-MSE": "#377eb8",
    }
    return palette.get(model_name, "#555555")


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")


def _add_panel_tag(axis: Any, tag: str) -> None:
    axis.text(
        -0.16,
        1.08,
        tag,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
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


def _monotone_story_palette() -> dict[str, str]:
    return {
        "CAIROK1": "#0E766E",
        "CAIROG1": "#C65D00",
        "CAIROS1": "#8F6B00",
        "CAIROG2": "#5B4CC4",
        "NN-Huber": "#B63A6B",
        "NN-MSE": "#3C4F76",
        "NN-MAE": "#7A7F87",
    }


def _plot_metric_with_ci(
    axis: Any,
    subset: pd.DataFrame,
    x_col: str,
    metric: str,
    color: str,
    label: str,
    linewidth: float = 2.2,
    alpha: float = 1.0,
    linestyle: str = "-",
    zorder: int = 2,
) -> None:
    if subset.empty:
        return
    axis.plot(
        subset[x_col],
        subset[f"{metric}_mean"],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
        marker="o",
        markersize=4.5,
        label=label,
        zorder=zorder,
    )
    ci_col = f"{metric}_ci"
    if ci_col in subset.columns:
        means = subset[f"{metric}_mean"].to_numpy(dtype=float)
        cis = np.nan_to_num(subset[ci_col].to_numpy(dtype=float), nan=0.0)
        axis.fill_between(
            subset[x_col].to_numpy(dtype=float),
            means - cis,
            means + cis,
            color=color,
            alpha=0.10 * alpha,
            linewidth=0,
            zorder=max(0, zorder - 1),
        )


def _paired_metric_gain(
    runs: pd.DataFrame,
    family: str,
    metric: str,
    x_col: str,
) -> pd.DataFrame:
    stage1 = runs[runs["Model"] == f"{family}-Stage1"][[x_col, "Repeat", metric]].rename(
        columns={metric: f"{metric}_stage1"}
    )
    stage2 = runs[runs["Model"] == family][[x_col, "Repeat", metric]].rename(
        columns={metric: f"{metric}_stage2"}
    )
    merged = stage1.merge(stage2, on=[x_col, "Repeat"], how="inner")
    merged[f"{metric}_gain"] = merged[f"{metric}_stage1"] - merged[f"{metric}_stage2"]
    rows: list[dict[str, float]] = []
    for value, group in merged.groupby(x_col, dropna=False):
        gains = group[f"{metric}_gain"].to_numpy(dtype=float)
        rows.append(
            {
                x_col: float(value),
                f"{metric}_gain_mean": float(np.mean(gains)),
                f"{metric}_gain_ci": (
                    float(1.96 * np.std(gains, ddof=1) / np.sqrt(gains.size))
                    if gains.size > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(x_col)


def plot_monotone_story(runs: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if runs.empty or summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    palette = _monotone_story_palette()
    strengths = np.sort(summary["DistortionStrength"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.3))
    ax_stage1, ax_gain, ax_rmse, ax_calgap = axes.ravel()

    for axis in axes.ravel():
        axis.set_facecolor("#fcfcfc")
        axis.grid(True, which="major", alpha=0.16, linewidth=0.8)
        axis.margins(x=0.04)
        axis.set_xlim(float(np.min(strengths)) - 0.03, float(np.max(strengths)) + 0.03)

    for family in MONOTONE_STORY_FAMILIES:
        subset = summary[summary["Model"] == f"{family}-Stage1"].sort_values("DistortionStrength")
        _plot_metric_with_ci(
            ax_stage1,
            subset,
            "DistortionStrength",
            "PairAcc",
            color=palette[family],
            label=_story_label(family),
            linewidth=2.4,
            zorder=3,
        )

        gain_df = _paired_metric_gain(runs, family, metric="RMSE", x_col="DistortionStrength")
        if not gain_df.empty:
            ax_gain.plot(
                gain_df["DistortionStrength"],
                gain_df["RMSE_gain_mean"],
                color=palette[family],
                linewidth=2.4,
                marker="o",
                markersize=4.5,
                label=_story_label(family),
                zorder=3,
            )
            gain_ci = np.nan_to_num(gain_df["RMSE_gain_ci"].to_numpy(dtype=float), nan=0.0)
            gain_mean = gain_df["RMSE_gain_mean"].to_numpy(dtype=float)
            xvals = gain_df["DistortionStrength"].to_numpy(dtype=float)
            ax_gain.fill_between(
                xvals,
                gain_mean - gain_ci,
                gain_mean + gain_ci,
                color=palette[family],
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

    for model_name in MONOTONE_STORY_FINAL_MODELS:
        subset = summary[summary["Model"] == model_name].sort_values("DistortionStrength")
        linewidth = 2.5 if model_name.startswith("CAIRO") else 2.0
        alpha = 1.0 if model_name.startswith("CAIRO") else 0.92
        zorder = 4 if model_name.startswith("CAIRO") else 2
        _plot_metric_with_ci(
            ax_rmse,
            subset,
            "DistortionStrength",
            "RMSE",
            color=palette[model_name],
            label=_story_label(model_name),
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        _plot_metric_with_ci(
            ax_calgap,
            subset,
            "DistortionStrength",
            "CalGap",
            color=palette[model_name],
            label=_story_label(model_name),
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )

    ax_stage1.set_title("Stage 1 Keeps the Order")
    ax_stage1.set_xlabel("Distortion strength $\\alpha$")
    ax_stage1.set_ylabel("Pair accuracy")

    ax_gain.set_title("Stage 2 Repairs Scale")
    ax_gain.set_xlabel("Distortion strength $\\alpha$")
    ax_gain.set_ylabel("RMSE reduction from isotonic calibration")
    ax_gain.axhline(0.0, color="#6f6f6f", linewidth=1.0, linestyle=(0, (4, 3)), alpha=0.75)
    ax_gain.text(
        0.03,
        0.96,
        "Positive values mean calibration helps.",
        transform=ax_gain.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#555555",
    )

    ax_rmse.set_title("Final RMSE After Calibration")
    ax_rmse.set_xlabel("Distortion strength $\\alpha$")
    ax_rmse.set_ylabel("Test RMSE")

    ax_calgap.set_title("Final Calibration Gap")
    ax_calgap.set_xlabel("Distortion strength $\\alpha$")
    ax_calgap.set_ylabel("Calibration gap")

    for axis, tag in zip(axes.ravel(), ("A", "B", "C", "D")):
        _add_panel_tag(axis, tag)

    handles, labels = ax_calgap.get_legend_handles_labels()
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
    fig.savefig((plot_dir / "monotone_story").with_suffix(".png"), bbox_inches="tight")
    fig.savefig((plot_dir / "monotone_story").with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_main_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    selected = [m for m in PRIMARY_PLOT_MODELS if m in set(summary["Model"])]
    scenario_order = list(dict.fromkeys(summary["Scenario"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    metric_specs = [("PairAcc", True), ("RMSE", False), ("CalGap", False)]
    x = np.arange(len(scenario_order))
    for axis, (metric, _) in zip(axes, metric_specs):
        for model_name in selected:
            subset = summary[summary["Model"] == model_name].set_index("Scenario").reindex(scenario_order)
            means = subset[f"{metric}_mean"].to_numpy(dtype=float)
            cis = np.nan_to_num(subset[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0)
            axis.errorbar(
                x,
                means,
                yerr=cis,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=model_name,
                color=_model_color(model_name),
            )
        axis.set_title(metric)
        axis.set_xticks(x)
        axis.set_xticklabels(scenario_order, rotation=25, ha="right")
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    _save_figure(fig, plot_dir / "main_summary")
    plt.close(fig)


def plot_pair_ablation(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    scenarios = list(dict.fromkeys(summary["Scenario"]))
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(10, 3.8 * len(scenarios)), squeeze=False)
    for row_idx, scenario_name in enumerate(scenarios):
        subset = summary[summary["Scenario"] == scenario_name]
        for col_idx, metric in enumerate(("PairAcc", "CalGap")):
            axis = axes[row_idx][col_idx]
            for sampler in PAIR_SAMPLERS:
                sampler_df = subset[subset["Sampler"] == sampler].sort_values("PairBudget")
                axis.errorbar(
                    sampler_df["PairBudget"],
                    sampler_df[f"{metric}_mean"],
                    yerr=np.nan_to_num(sampler_df[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    label=sampler,
                )
            axis.set_title(f"{scenario_name} | {metric}")
            axis.set_xlabel("Pair budget")
            axis.set_ylabel(metric)
    axes[0][-1].legend(frameon=False, loc="best")
    _save_figure(fig, plot_dir / "pair_sampling_ablation")
    plt.close(fig)


def plot_sweep(
    summary: pd.DataFrame,
    output_dir: Path,
    sweep_name: str,
    x_col: str,
    models: tuple[str, ...] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    selected = [m for m in (models or PRIMARY_PLOT_MODELS) if m in set(summary["Model"])]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for axis, metric in zip(axes, ("PairAcc", "RMSE", "CalGap")):
        for model_name in selected:
            subset = summary[summary["Model"] == model_name].sort_values(x_col)
            axis.errorbar(
                subset[x_col],
                subset[f"{metric}_mean"],
                yerr=np.nan_to_num(subset[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=_model_color(model_name),
                label=model_name,
            )
        axis.set_title(metric)
        axis.set_xlabel(x_col)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    _save_figure(fig, plot_dir / sweep_name)
    plt.close(fig)


def plot_low_data_sweep(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    scenarios = list(dict.fromkeys(summary["StudyScenario"]))
    selected = [m for m in PRIMARY_PLOT_MODELS if m in set(summary["Model"])]
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(11, 3.8 * len(scenarios)), squeeze=False)
    for row_idx, scenario_name in enumerate(scenarios):
        subset = summary[summary["StudyScenario"] == scenario_name]
        for col_idx, metric in enumerate(("PairAcc", "RMSE")):
            axis = axes[row_idx][col_idx]
            for model_name in selected:
                model_df = subset[subset["Model"] == model_name].sort_values("SampleBudget")
                axis.errorbar(
                    model_df["SampleBudget"],
                    model_df[f"{metric}_mean"],
                    yerr=np.nan_to_num(model_df[f"{metric}_ci"].to_numpy(dtype=float), nan=0.0),
                    marker="o",
                    linewidth=1.8,
                    capsize=3,
                    color=_model_color(model_name),
                    label=model_name,
                )
            axis.set_title(f"{scenario_name} | {metric}")
            axis.set_xlabel("Total labeled samples")
            axis.set_ylabel(metric)
    axes[0][-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    _save_figure(fig, plot_dir / "low_data_sweep")
    plt.close(fig)


def plot_reliability_curves(reliability: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if reliability.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    grouped = (
        reliability.groupby(["Scenario", "Model", "bin"], as_index=False)[["pred_mean", "obs_mean"]]
        .mean()
        .sort_values(["Scenario", "Model", "bin"])
    )
    scenarios = [
        scenario
        for scenario in ("Heteroskedastic lognormal", "Label contamination (10%)")
        if scenario in set(grouped["Scenario"])
    ]
    if not scenarios:
        scenarios = list(dict.fromkeys(grouped["Scenario"]))[:2]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.5 * len(scenarios), 4.2), squeeze=False)
    for axis, scenario_name in zip(axes[0], scenarios):
        subset = grouped[grouped["Scenario"] == scenario_name]
        for model_name in PRIMARY_PLOT_MODELS:
            model_df = subset[subset["Model"] == model_name]
            if model_df.empty:
                continue
            axis.plot(
                model_df["pred_mean"],
                model_df["obs_mean"],
                marker="o",
                linewidth=1.8,
                color=_model_color(model_name),
                label=model_name,
            )
        xmin = float(subset["pred_mean"].min())
        xmax = float(subset["pred_mean"].max())
        axis.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black", linewidth=1)
        axis.set_title(f"Reliability: {scenario_name}")
        axis.set_xlabel("Predicted bin mean")
        axis.set_ylabel("Observed bin mean")
    axes[0][-1].legend(frameon=False, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    _save_figure(fig, plot_dir / "reliability_curves")
    plt.close(fig)


def run_main_benchmark(
    cfg: RuntimeConfig,
    scenario_filter: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    scenarios = build_main_scenarios()
    if scenario_filter:
        scenarios = [scenario for scenario in scenarios if scenario.name in scenario_filter]
    print(f"[Main] Running {len(scenarios)} scenario(s) in {cfg.mode} mode", flush=True)
    for scenario in scenarios:
        for repeat in range(cfg.n_repeats):
            run_seed = cfg.seed + 1000 * repeat
            rows, reliability = run_one_repeat(
                scenario,
                cfg,
                repeat,
                run_seed,
                selected_models=set(MAIN_BENCHMARK_MODELS),
            )
            all_rows.extend(rows)
            reliability_rows.extend(reliability)
    runs = pd.DataFrame(all_rows)
    reliability_df = pd.DataFrame(reliability_rows)
    summary = summarize_with_ci(runs, ["Scenario", "Model"])
    return runs, summary, reliability_df


def run_pair_sampling_study(cfg: RuntimeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    study_scenarios = [
        scenario
        for scenario in build_main_scenarios()
        if scenario.name in {"Heteroskedastic lognormal", "Student-t heavy tail (df=2.5)"}
    ]
    rows: list[dict[str, Any]] = []
    for scenario in study_scenarios:
        for repeat in range(min(cfg.n_repeats, 2)):
            run_seed = cfg.seed + 5000 * repeat
            rows.extend(run_pair_sampling_ablation(scenario, cfg, repeat, run_seed))
    ablation = pd.DataFrame(rows)
    summary = summarize_with_ci(ablation, ["Scenario", "Sampler", "PairBudget"])
    return ablation, summary


def run_monotone_distortion_study(cfg: RuntimeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = set(FAIR_NEURAL_STAGE2_MODELS)
    rows: list[dict[str, Any]] = []
    print("[Study] Monotone distortion sweep", flush=True)
    for strength in cfg.monotone_strengths:
        scenario = ScenarioSpec(
            name=f"Monotone distortion (α={strength:.1f})",
            kind="monotone_distortion",
            generator=lambda seed, local_cfg, value=strength: gen_monotone_distortion(seed, local_cfg, strength=value),
        )
        for repeat in range(cfg.n_repeats):
            run_seed = cfg.seed + 11000 * repeat + int(strength * 100)
            scenario_rows, _ = run_one_repeat(
                scenario,
                cfg,
                repeat,
                run_seed,
                selected_models=selected,
                include_uncalibrated_cairo=True,
            )
            for row in scenario_rows:
                row["DistortionStrength"] = strength
                rows.append(row)
    runs = pd.DataFrame(rows)
    summary = summarize_with_ci(runs, ["DistortionStrength", "Model"])
    return runs, summary


def run_low_data_study(cfg: RuntimeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = {
        "CAIROK1",
        "CAIROG1",
        "CAIROS1",
        "CAIROG2",
        "CAIRO-XGBPairwise",
        "NN-Huber",
        "QuantileRegressor",
        "NN-MSE",
        "NN-MAE",
        "CatBoost-RMSE",
        "LightGBM-MSE",
    }
    study_specs = [
        ("Normal (σ=1.0)", lambda seed, local_cfg: gen_normal_regression(seed, local_cfg)),
        (
            "Monotone distortion (α=1.2)",
            lambda seed, local_cfg: gen_monotone_distortion(seed, local_cfg, strength=1.2),
        ),
    ]
    rows: list[dict[str, Any]] = []
    print("[Study] Low-data sweep", flush=True)
    for n_samples in cfg.low_data_sizes:
        local_cfg = replace(cfg, n_samples=n_samples)
        for study_name, generator in study_specs:
            scenario = ScenarioSpec(name=study_name, kind="low_data", generator=generator)
            for repeat in range(cfg.n_repeats):
                run_seed = cfg.seed + 13000 * repeat + n_samples
                scenario_rows, _ = run_one_repeat(
                    scenario,
                    local_cfg,
                    repeat,
                    run_seed,
                    selected_models=selected,
                )
                for row in scenario_rows:
                    row["SampleBudget"] = n_samples
                    row["StudyScenario"] = study_name
                    rows.append(row)
    runs = pd.DataFrame(rows)
    summary = summarize_with_ci(runs, ["StudyScenario", "SampleBudget", "Model"])
    return runs, summary


def run_stress_sweeps(cfg: RuntimeConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = {
        "CAIROK1",
        "CAIROG1",
        "CAIROS1",
        "CAIROG2",
        "CAIRO-XGBPairwise",
        "NN-Huber",
        "QuantileRegressor",
        "NN-MSE",
        "NN-MAE",
        "CatBoost-RMSE",
        "LightGBM-MSE",
    }

    contamination_rows: list[dict[str, Any]] = []
    print("[Sweep] Contamination sweep", flush=True)
    for eps in cfg.contamination_levels:
        scenario = ScenarioSpec(
            name=f"Contamination sweep ({eps:.2f})",
            kind="contamination_sweep",
            generator=lambda seed, local_cfg, value=eps: gen_label_contamination(seed, local_cfg, eps=value),
        )
        for repeat in range(cfg.n_repeats):
            run_seed = cfg.seed + 7000 * repeat + int(eps * 100)
            rows, _ = run_one_repeat(scenario, cfg, repeat, run_seed, selected_models=selected)
            for row in rows:
                row["ContaminationRate"] = eps
                contamination_rows.append(row)

    student_rows: list[dict[str, Any]] = []
    print("[Sweep] Student-t sweep", flush=True)
    for df_value in cfg.student_t_dfs:
        scenario = ScenarioSpec(
            name=f"Student-t sweep (df={df_value:.1f})",
            kind="student_t_sweep",
            generator=lambda seed, local_cfg, value=df_value: gen_student_t_heavytail(seed, local_cfg, df=value),
        )
        for repeat in range(cfg.n_repeats):
            run_seed = cfg.seed + 9000 * repeat + int(df_value * 10)
            rows, _ = run_one_repeat(scenario, cfg, repeat, run_seed, selected_models=selected)
            for row in rows:
                row["StudentT_df"] = df_value
                student_rows.append(row)

    contamination = pd.DataFrame(contamination_rows)
    student = pd.DataFrame(student_rows)
    contamination_summary = summarize_with_ci(contamination, ["ContaminationRate", "Model"])
    student_summary = summarize_with_ci(student, ["StudentT_df", "Model"])
    return contamination, contamination_summary, student, student_summary


def save_outputs(
    cfg: RuntimeConfig,
    runs: pd.DataFrame,
    summary: pd.DataFrame,
    reliability: pd.DataFrame,
    ablation: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    monotone: pd.DataFrame,
    monotone_summary: pd.DataFrame,
    low_data: pd.DataFrame,
    low_data_summary: pd.DataFrame,
    contamination: pd.DataFrame,
    contamination_summary: pd.DataFrame,
    student: pd.DataFrame,
    student_summary: pd.DataFrame,
) -> None:
    ensure_dir(cfg.output_dir)
    write_json(cfg.output_dir / "config.json", {**asdict(cfg), "output_dir": str(cfg.output_dir)})
    write_json(cfg.output_dir / "availability.json", available_components())
    write_csv(cfg.output_dir / "main_runs.csv", runs)
    write_csv(cfg.output_dir / "main_summary.csv", summary)
    write_csv(cfg.output_dir / "reliability_bins.csv", reliability)
    write_csv(cfg.output_dir / "pair_ablation_runs.csv", ablation)
    write_csv(cfg.output_dir / "pair_ablation_summary.csv", ablation_summary)
    write_csv(cfg.output_dir / "monotone_distortion_runs.csv", monotone)
    write_csv(cfg.output_dir / "monotone_distortion_summary.csv", monotone_summary)
    write_csv(cfg.output_dir / "low_data_runs.csv", low_data)
    write_csv(cfg.output_dir / "low_data_summary.csv", low_data_summary)
    write_csv(cfg.output_dir / "contamination_sweep_runs.csv", contamination)
    write_csv(cfg.output_dir / "contamination_sweep_summary.csv", contamination_summary)
    write_csv(cfg.output_dir / "student_t_sweep_runs.csv", student)
    write_csv(cfg.output_dir / "student_t_sweep_summary.csv", student_summary)
    write_csv(cfg.output_dir / "overall_ranks.csv", compute_overall_ranks(runs))


def create_plots(
    cfg: RuntimeConfig,
    summary: pd.DataFrame,
    reliability: pd.DataFrame,
    ablation_summary: pd.DataFrame,
    monotone_runs: pd.DataFrame,
    monotone_summary: pd.DataFrame,
    low_data_summary: pd.DataFrame,
    contamination_summary: pd.DataFrame,
    student_summary: pd.DataFrame,
) -> None:
    plot_main_summary(summary, cfg.output_dir)
    plot_pair_ablation(ablation_summary, cfg.output_dir)
    plot_sweep(
        monotone_summary,
        cfg.output_dir,
        "monotone_distortion_sweep",
        "DistortionStrength",
    )
    plot_sweep(
        monotone_summary,
        cfg.output_dir,
        "monotone_stage2_comparison",
        "DistortionStrength",
        models=FAIR_NEURAL_STAGE2_MODELS,
    )
    plot_monotone_story(monotone_runs, monotone_summary, cfg.output_dir)
    plot_low_data_sweep(low_data_summary, cfg.output_dir)
    plot_sweep(contamination_summary, cfg.output_dir, "contamination_sweep", "ContaminationRate")
    plot_sweep(student_summary, cfg.output_dir, "student_t_sweep", "StudentT_df")
    plot_reliability_curves(reliability, cfg.output_dir)


def run_full_synthetic_benchmark(
    cfg: RuntimeConfig,
    scenario_filter: set[str] | None = None,
    skip_sweeps: bool = False,
    skip_plots: bool = False,
) -> dict[str, pd.DataFrame]:
    runs, summary, reliability = run_main_benchmark(cfg, scenario_filter=scenario_filter)
    if skip_sweeps:
        ablation = pd.DataFrame()
        ablation_summary = pd.DataFrame()
        monotone = pd.DataFrame()
        monotone_summary = pd.DataFrame()
        low_data = pd.DataFrame()
        low_data_summary = pd.DataFrame()
        contamination = pd.DataFrame()
        contamination_summary = pd.DataFrame()
        student = pd.DataFrame()
        student_summary = pd.DataFrame()
    else:
        ablation, ablation_summary = run_pair_sampling_study(cfg)
        monotone, monotone_summary = run_monotone_distortion_study(cfg)
        low_data, low_data_summary = run_low_data_study(cfg)
        contamination, contamination_summary, student, student_summary = run_stress_sweeps(cfg)

    save_outputs(
        cfg,
        runs,
        summary,
        reliability,
        ablation,
        ablation_summary,
        monotone,
        monotone_summary,
        low_data,
        low_data_summary,
        contamination,
        contamination_summary,
        student,
        student_summary,
    )
    if not skip_plots:
        create_plots(
            cfg,
            summary,
            reliability,
            ablation_summary,
            monotone,
            monotone_summary,
            low_data_summary,
            contamination_summary,
            student_summary,
        )
    return {
        "runs": runs,
        "summary": summary,
        "reliability": reliability,
        "ablation": ablation,
        "ablation_summary": ablation_summary,
        "monotone": monotone,
        "monotone_summary": monotone_summary,
        "low_data": low_data,
        "low_data_summary": low_data_summary,
        "contamination": contamination,
        "contamination_summary": contamination_summary,
        "student": student,
        "student_summary": student_summary,
    }
