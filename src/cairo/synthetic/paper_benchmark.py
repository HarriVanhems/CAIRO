from __future__ import annotations

import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
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
PRIMARY_PLOT_MODELS = (
    "CAIRO-RankNet",
    "CAIRO-RankNet-GiniW",
    "CAIRO-GiniNet-SoftRank",
    "NN-Huber+ISO",
    "RandomForest+ISO",
    "LightGBM-MSE+ISO",
)


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
    output_dir: Path

    @classmethod
    def from_mode(
        cls,
        mode: str,
        output_dir: Path | None = None,
        repeats: int | None = None,
    ) -> "RuntimeConfig":
        if mode not in {"debug", "full"}:
            raise ValueError(f"Unsupported mode: {mode}")

        base = {
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


def _import_tf_stack() -> tuple[Any, Any, Any, Any]:
    import tensorflow as tf
    from tensorflow.keras import Input, Model, layers
    from tensorflow.nn import leaky_relu

    tf.get_logger().setLevel("ERROR")
    return tf, layers, Model, Input, leaky_relu


def _import_torch_stack() -> tuple[Any, Any, Any, Any]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from fast_soft_sort.pytorch_ops import soft_rank

    return torch, nn, optim, soft_rank


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
    optional = _load_optional_baseline_apis()
    for key in ("lightgbm", "xgboost", "catboost"):
        availability[key] = optional[key] is not None
    return availability


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


def make_pairs(
    x: np.ndarray,
    y: np.ndarray,
    num_pairs: int,
    seed: int,
    sampler: str = "uniform",
    weighted: bool = False,
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
    targets = (yi > yj).astype(np.float32)
    if not weighted:
        return (xi, xj), targets, None
    weights = np.abs(yi - yj).astype(np.float32)
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


def train_ranknet_stage1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: RuntimeConfig,
    sampler: str,
    weighted: bool,
    seed: int,
) -> Any:
    tf, layers, Model, Input, leaky_relu = _import_tf_stack()
    tf.random.set_seed(seed)

    def build_branch(d_in: int) -> Any:
        inp = Input(shape=(d_in,))
        x = layers.Dense(32, activation=leaky_relu)(inp)
        x = layers.Dense(16, activation=leaky_relu)(x)
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
        weighted=weighted,
        debug=cfg.mode == "debug",
    )
    (xi_val, xj_val), targets_val, weights_val = make_pairs(
        x_train,
        y_train,
        cfg.pairs_val,
        seed=seed + 1,
        sampler="uniform",
        weighted=weighted,
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
        metrics=[_logit_binary_accuracy_metric(tf)(name="acc")],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    pair_model.fit(ds_train, validation_data=ds_val, epochs=cfg.epochs_ranknet, verbose=0, callbacks=callbacks)
    return scorer


def ranknet_scores(model: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    return model.predict(x, batch_size=batch_size, verbose=0).squeeze()


def train_gininet_stage1(
    x_train: np.ndarray,
    y_train: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
    tau: float = 1.0,
) -> Any:
    torch, nn, optim, soft_rank = _import_torch_stack()
    torch.manual_seed(seed)

    class GiniNet(nn.Module):
        def __init__(self, d_in: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, batch: Any) -> Any:
            return self.net(batch).squeeze(-1)

    def gini_cov_loss(targets: Any, scores: Any) -> Any:
        ranks = soft_rank(scores.unsqueeze(0), regularization_strength=tau).squeeze(0)
        centered_y = targets - targets.mean()
        centered_r = ranks - ranks.mean()
        return -(centered_y * centered_r).mean()

    x_tensor = torch.tensor(x_train, dtype=torch.float32, device="cpu")
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device="cpu")
    n, d_in = x_train.shape
    model = GiniNet(d_in).to("cpu")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    rr = default_rng(seed)
    model.train()
    for _ in range(cfg.epochs_gini):
        idx = rr.permutation(n)
        for start in range(0, n, cfg.batch_size):
            batch = idx[start : start + cfg.batch_size]
            if len(batch) < 2:
                continue
            xb = x_tensor[batch]
            yb = y_tensor[batch]
            optimizer.zero_grad()
            scores = model(xb)
            loss = gini_cov_loss(yb, scores)
            loss.backward()
            optimizer.step()
    model.eval()
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


def train_mlp_regressor_tf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RuntimeConfig,
    seed: int,
    loss_name: str,
) -> Any:
    tf, layers, Model, Input, leaky_relu = _import_tf_stack()
    tf.random.set_seed(seed)
    d_in = x_train.shape[1]
    losses = {
        "mse": tf.keras.losses.MeanSquaredError(),
        "mae": tf.keras.losses.MeanAbsoluteError(),
        "huber": tf.keras.losses.Huber(delta=1.0),
        "logcosh": tf.keras.losses.LogCosh(),
    }
    inp = Input(shape=(d_in,))
    x = layers.Dense(32, activation=leaky_relu)(inp)
    x = layers.Dense(16, activation=leaky_relu)(x)
    out = layers.Dense(1, activation="linear")(x)
    model = Model(inp, out, name=f"mlp_{loss_name}")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses[loss_name])

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    ds_train = ds_train.shuffle(min(len(y_train), 50_000), seed=seed, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(cfg.batch_size).prefetch(tf.data.AUTOTUNE)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    model.fit(ds_train, validation_data=ds_val, epochs=cfg.epochs_ranknet, verbose=0, callbacks=callbacks)
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


def run_one_repeat(
    scenario: ScenarioSpec,
    cfg: RuntimeConfig,
    repeat: int,
    run_seed: int,
    selected_models: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    set_all_seeds(run_seed)
    x, y = scenario.generator(run_seed, cfg)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.3, random_state=run_seed
    )
    x_train_rank, x_cal, y_train_rank, y_cal = train_test_split(
        x_train_full, y_train_full, test_size=0.3, random_state=run_seed + 1
    )
    x_train_reg, x_val_reg, y_train_reg, y_val_reg = train_test_split(
        x_train_full, y_train_full, test_size=0.2, random_state=run_seed + 2
    )

    availability = available_components()
    optional = _load_optional_baseline_apis()
    rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    want = lambda model_name: selected_models is None or model_name in selected_models

    if want("CAIRO-RankNet"):
        scorer = train_ranknet_stage1(x_train_rank, y_train_rank, cfg, sampler="uniform", weighted=False, seed=run_seed + 10)
        iso = fit_isotonic(ranknet_scores(scorer, x_cal, cfg.batch_size), y_cal)
        yhat = iso.predict(ranknet_scores(scorer, x_test, cfg.batch_size))
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "CAIRO-RankNet", y_test, yhat, cfg, run_seed + 10)

    if want("CAIRO-RankNet-GiniW"):
        scorer = train_ranknet_stage1(x_train_rank, y_train_rank, cfg, sampler="uniform", weighted=True, seed=run_seed + 20)
        iso = fit_isotonic(ranknet_scores(scorer, x_cal, cfg.batch_size), y_cal)
        yhat = iso.predict(ranknet_scores(scorer, x_test, cfg.batch_size))
        _record_predictions(rows, reliability_rows, scenario.name, repeat, "CAIRO-RankNet-GiniW", y_test, yhat, cfg, run_seed + 20)

    if want("CAIRO-GiniNet-SoftRank") and availability["torch"] and availability["fast_soft_sort"]:
        gininet = train_gininet_stage1(x_train_rank, y_train_rank, cfg, seed=run_seed + 30)
        iso = fit_isotonic(gininet_scores(gininet, x_cal), y_cal)
        yhat = iso.predict(gininet_scores(gininet, x_test))
        _record_predictions(
            rows,
            reliability_rows,
            scenario.name,
            repeat,
            "CAIRO-GiniNet-SoftRank",
            y_test,
            yhat,
            cfg,
            run_seed + 30,
        )

    baseline_specs = {
        "NN-MSE": "mse",
        "NN-MAE": "mae",
        "NN-Huber": "huber",
        "NN-LogCosh": "logcosh",
    }
    for model_name, loss_name in baseline_specs.items():
        needs_any = want(model_name) or want(f"{model_name}+ISO")
        if not needs_any:
            continue
        model = train_mlp_regressor_tf(
            x_train_reg,
            y_train_reg,
            x_val_reg,
            y_val_reg,
            cfg,
            seed=run_seed + 100,
            loss_name=loss_name,
        )
        yhat = predict_tf(model, x_test, cfg.batch_size)
        if want(model_name):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, model_name, y_test, yhat, cfg, run_seed + 100)
        if want(f"{model_name}+ISO"):
            iso = fit_isotonic(predict_tf(model, x_cal, cfg.batch_size), y_cal)
            yhat_iso = iso.predict(yhat)
            _record_predictions(rows, reliability_rows, scenario.name, repeat, f"{model_name}+ISO", y_test, yhat_iso, cfg, run_seed + 101)

    if want("RandomForest") or want("RandomForest+ISO"):
        rf = RandomForestRegressor(
            n_estimators=120 if cfg.mode == "debug" else 300,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=run_seed,
        )
        rf.fit(x_train_full, y_train_full)
        yhat = rf.predict(x_test)
        if want("RandomForest"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "RandomForest", y_test, yhat, cfg, run_seed + 200)
        if want("RandomForest+ISO"):
            iso = fit_isotonic(rf.predict(x_cal), y_cal)
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "RandomForest+ISO",
                y_test,
                iso.predict(yhat),
                cfg,
                run_seed + 201,
            )

    if want("HuberRegressor") or want("HuberRegressor+ISO"):
        huber = HuberRegressor()
        huber.fit(x_train_full.astype(np.float64), y_train_full.astype(np.float64))
        yhat = huber.predict(x_test.astype(np.float64))
        if want("HuberRegressor"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "HuberRegressor", y_test, yhat, cfg, run_seed + 210)
        if want("HuberRegressor+ISO"):
            iso = fit_isotonic(huber.predict(x_cal.astype(np.float64)), y_cal)
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "HuberRegressor+ISO",
                y_test,
                iso.predict(yhat),
                cfg,
                run_seed + 211,
            )

    if optional["lightgbm"] is not None and (want("LightGBM-MSE") or want("LightGBM-MSE+ISO")):
        model = optional["lightgbm"](
            n_estimators=200 if cfg.mode == "debug" else 500,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=run_seed,
            objective="regression",
            n_jobs=1,
            verbosity=-1,
        )
        model.fit(x_train_full, y_train_full)
        yhat = model.predict(x_test)
        if want("LightGBM-MSE"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "LightGBM-MSE", y_test, yhat, cfg, run_seed + 220)
        if want("LightGBM-MSE+ISO"):
            iso = fit_isotonic(model.predict(x_cal), y_cal)
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "LightGBM-MSE+ISO",
                y_test,
                iso.predict(yhat),
                cfg,
                run_seed + 221,
            )

    if optional["xgboost"] is not None and (want("XGBoost-MSE") or want("XGBoost-MSE+ISO")):
        model = optional["xgboost"](
            n_estimators=250 if cfg.mode == "debug" else 800,
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
        model.fit(x_train_full, y_train_full)
        yhat = model.predict(x_test)
        if want("XGBoost-MSE"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "XGBoost-MSE", y_test, yhat, cfg, run_seed + 230)
        if want("XGBoost-MSE+ISO"):
            iso = fit_isotonic(model.predict(x_cal), y_cal)
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "XGBoost-MSE+ISO",
                y_test,
                iso.predict(yhat),
                cfg,
                run_seed + 231,
            )

    if optional["catboost"] is not None and (want("CatBoost-RMSE") or want("CatBoost-RMSE+ISO")):
        model = optional["catboost"](
            iterations=400 if cfg.mode == "debug" else 2000,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=run_seed,
            verbose=False,
            thread_count=1,
        )
        model.fit(x_train_full, y_train_full)
        yhat = model.predict(x_test)
        if want("CatBoost-RMSE"):
            _record_predictions(rows, reliability_rows, scenario.name, repeat, "CatBoost-RMSE", y_test, yhat, cfg, run_seed + 240)
        if want("CatBoost-RMSE+ISO"):
            iso = fit_isotonic(model.predict(x_cal), y_cal)
            _record_predictions(
                rows,
                reliability_rows,
                scenario.name,
                repeat,
                "CatBoost-RMSE+ISO",
                y_test,
                iso.predict(yhat),
                cfg,
                run_seed + 241,
            )
    return rows, reliability_rows


def run_pair_sampling_ablation(
    scenario: ScenarioSpec,
    cfg: RuntimeConfig,
    repeat: int,
    run_seed: int,
) -> list[dict[str, Any]]:
    set_all_seeds(run_seed)
    x, y = scenario.generator(run_seed, cfg)
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x, y, test_size=0.3, random_state=run_seed
    )
    x_train_rank, x_cal, y_train_rank, y_cal = train_test_split(
        x_train_full, y_train_full, test_size=0.3, random_state=run_seed + 1
    )
    rows = []
    for sampler in PAIR_SAMPLERS:
        for budget in cfg.pair_budgets:
            scorer = train_ranknet_stage1(
                x_train_rank,
                y_train_rank,
                replace(cfg, pairs_train=budget, pairs_val=min(cfg.pairs_val, max(5_000, budget // 5))),
                sampler=sampler,
                weighted=False,
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
        "CAIRO-RankNet": "#1b9e77",
        "CAIRO-RankNet-GiniW": "#d95f02",
        "CAIRO-GiniNet-SoftRank": "#7570b3",
        "NN-Huber+ISO": "#e7298a",
        "RandomForest+ISO": "#66a61e",
        "LightGBM-MSE+ISO": "#1f78b4",
    }
    return palette.get(model_name, "#555555")


def _save_figure(fig: Any, path_stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")


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


def plot_sweep(summary: pd.DataFrame, output_dir: Path, sweep_name: str, x_col: str) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    _plot_style()
    plot_dir = ensure_dir(output_dir / "figures")
    selected = [m for m in PRIMARY_PLOT_MODELS if m in set(summary["Model"])]
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
    for scenario in scenarios:
        for repeat in range(cfg.n_repeats):
            run_seed = cfg.seed + 1000 * repeat
            rows, reliability = run_one_repeat(scenario, cfg, repeat, run_seed)
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


def run_stress_sweeps(cfg: RuntimeConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = {
        "CAIRO-RankNet",
        "CAIRO-RankNet-GiniW",
        "CAIRO-GiniNet-SoftRank",
        "NN-Huber+ISO",
        "RandomForest+ISO",
        "LightGBM-MSE+ISO",
    }

    contamination_rows: list[dict[str, Any]] = []
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
    contamination_summary: pd.DataFrame,
    student_summary: pd.DataFrame,
) -> None:
    plot_main_summary(summary, cfg.output_dir)
    plot_pair_ablation(ablation_summary, cfg.output_dir)
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
        contamination = pd.DataFrame()
        contamination_summary = pd.DataFrame()
        student = pd.DataFrame()
        student_summary = pd.DataFrame()
    else:
        ablation, ablation_summary = run_pair_sampling_study(cfg)
        contamination, contamination_summary, student, student_summary = run_stress_sweeps(cfg)

    save_outputs(
        cfg,
        runs,
        summary,
        reliability,
        ablation,
        ablation_summary,
        contamination,
        contamination_summary,
        student,
        student_summary,
    )
    if not skip_plots:
        create_plots(cfg, summary, reliability, ablation_summary, contamination_summary, student_summary)
    return {
        "runs": runs,
        "summary": summary,
        "reliability": reliability,
        "ablation": ablation,
        "ablation_summary": ablation_summary,
        "contamination": contamination,
        "contamination_summary": contamination_summary,
        "student": student,
        "student_summary": student_summary,
    }
