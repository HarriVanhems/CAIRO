from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from cairo.paths import RESULTS_DIR, ensure_dir
from cairo.reporting import write_csv, write_json

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - dependency guard
    lgb = None
    LIGHTGBM_IMPORT_ERROR = exc
else:
    LIGHTGBM_IMPORT_ERROR = None


@dataclass(frozen=True)
class ZeroInflatedBenchmarkConfig:
    q_values: tuple[float, ...] = (1.0, 0.85, 0.75, 0.5, 0.25, 0.1)
    replications: int = 10
    n_train: int = 2_000
    n_tune: int = 500
    n_calibration: int = 500
    n_test: int = 5_000
    phi: float = 1.0
    tweedie_power: float = 1.5
    random_seed_offset: int = 1_000
    grid_levels: tuple[int, ...] = (5, 7)
    grid_num_leaves: tuple[int, ...] = (31, 63)
    grid_learning_rate: tuple[float, ...] = (0.05, 0.1)
    grid_trees: tuple[int, ...] = (400, 800)
    eval_at: int = 10
    n_jobs: int = 1


def two_hills_two_valleys(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.exp(-5 * ((1 - x1) ** 2 + x2**2)) + np.exp(-5 * (x1**2 + (1 - x2) ** 2))


def tweedie_compound_poisson(
    mu: np.ndarray,
    phi: float = 1.0,
    power: float = 1.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    lam = (mu ** (2 - power)) / (phi * (2 - power))
    alpha = (2 - power) / (power - 1)
    theta = phi * (power - 1) * (mu ** (power - 1))
    counts = rng.poisson(lam)
    out = np.zeros_like(mu)
    active = counts > 0
    if np.any(active):
        idx = np.where(active)[0]
        out[idx] = rng.gamma(shape=alpha * counts[idx], scale=theta[idx])
    return out


def generate_split(
    n: int,
    q: float,
    phi: float,
    power: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    features = np.column_stack([x1, x2])
    signal = two_hills_two_valleys(x1, x2)
    mu = np.exp(signal)
    y = tweedie_compound_poisson(mu, phi=phi, power=power, rng=rng)
    if q < 1.0:
        y = y * (rng.random(n) < q)
    premium = q * mu
    return features, y, premium


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-y_score)
    rel = y_true[order]
    gains = 2.0**rel - 1.0
    discount = 1.0 / np.log2(np.arange(2, 2 + min(k, len(rel))))
    dcg = float(np.sum(gains[: len(discount)] * discount))
    ideal = np.sort(y_true)[::-1]
    idcg = float(np.sum((2.0**ideal[: len(discount)] - 1.0) * discount))
    return dcg / idcg if idcg > 0 else np.nan


def normalized_gini(actual: np.ndarray, pred: np.ndarray) -> float:
    def gini_core(a: np.ndarray, p: np.ndarray) -> float:
        order = np.argsort(-p)
        ranked = a[order]
        total = ranked.sum()
        if total == 0:
            return 0.0
        curve = np.cumsum(ranked) / total
        baseline = np.arange(1, len(ranked) + 1) / len(ranked)
        return float(np.sum(curve - baseline) / len(ranked))

    denom = gini_core(actual, actual)
    if denom == 0:
        return np.nan
    return gini_core(actual, pred) / denom


def fit_relevance_binner(y_train: np.ndarray, n_levels: int) -> np.ndarray:
    edges = np.quantile(y_train, np.linspace(0, 1, n_levels + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([y_train.min(), y_train.max() + 1e-9])
    return edges


def apply_binner(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    internal = edges[1:-1]
    rel = np.digitize(y, internal, right=True).astype(int)
    return np.clip(rel, 0, len(edges) - 2)


def _silent_callbacks() -> list:
    if lgb is None:
        return []
    try:
        return [lgb.log_evaluation(period=0)]
    except Exception:  # pragma: no cover - version compatibility
        return []


def fit_lambdamart(
    x_train: np.ndarray,
    y_rel_train: np.ndarray,
    x_tune: np.ndarray,
    y_rel_tune: np.ndarray,
    seed: int,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    eval_at: int,
    n_jobs: int,
):
    if lgb is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "LightGBM is required for the zero-inflated benchmark."
        ) from LIGHTGBM_IMPORT_ERROR

    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        min_data_in_leaf=20,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_rel_train,
        group=[len(y_rel_train)],
        eval_set=[(x_tune, y_rel_tune)],
        eval_group=[[len(y_rel_tune)]],
        eval_at=[eval_at],
        eval_metric="ndcg",
        callbacks=_silent_callbacks(),
    )
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, eval_at: int) -> dict[str, float]:
    rho = spearmanr(y_true, y_pred).correlation
    return {
        "mad": float(np.mean(np.abs(y_true - y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "spearman": float(rho) if rho is not None else np.nan,
        "ndcg_at_k": float(ndcg_at_k(y_true, y_pred, k=eval_at)),
        "normalized_gini": float(normalized_gini(y_true, y_pred)),
    }


def run_once(config: ZeroInflatedBenchmarkConfig, q: float, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    x_train, y_train, _ = generate_split(config.n_train, q, config.phi, config.tweedie_power, rng)
    x_tune, y_tune, premium_tune = generate_split(
        config.n_tune, q, config.phi, config.tweedie_power, rng
    )
    x_cal, _, premium_cal = generate_split(
        config.n_calibration, q, config.phi, config.tweedie_power, rng
    )
    x_test, _, premium_test = generate_split(
        config.n_test, q, config.phi, config.tweedie_power, rng
    )

    best: tuple[float, int, int, float, int, object] | None = None
    for n_levels in config.grid_levels:
        edges = fit_relevance_binner(y_train, n_levels)
        y_rel_train = apply_binner(y_train, edges)
        y_rel_tune = apply_binner(y_tune, edges)
        for num_leaves in config.grid_num_leaves:
            for learning_rate in config.grid_learning_rate:
                for n_estimators in config.grid_trees:
                    model = fit_lambdamart(
                        x_train,
                        y_rel_train,
                        x_tune,
                        y_rel_tune,
                        seed=seed,
                        n_estimators=n_estimators,
                        num_leaves=num_leaves,
                        learning_rate=learning_rate,
                        eval_at=config.eval_at,
                        n_jobs=config.n_jobs,
                    )
                    tune_scores = model.predict(x_tune)
                    cal_scores = model.predict(x_cal)
                    iso = IsotonicRegression(out_of_bounds="clip").fit(cal_scores, premium_cal)
                    tuned_predictions = iso.transform(tune_scores)
                    tune_mad = float(np.mean(np.abs(premium_tune - tuned_predictions)))
                    candidate = (
                        tune_mad,
                        n_levels,
                        num_leaves,
                        learning_rate,
                        n_estimators,
                        model,
                    )
                    if best is None or candidate[0] < best[0]:
                        best = candidate

    assert best is not None
    _, n_levels, num_leaves, learning_rate, n_estimators, model = best
    cal_scores = model.predict(x_cal)
    test_scores = model.predict(x_test)
    iso = IsotonicRegression(out_of_bounds="clip").fit(cal_scores, premium_cal)
    y_pred = iso.transform(test_scores)
    metrics = evaluate_predictions(premium_test, y_pred, eval_at=config.eval_at)
    return {
        "q": q,
        "seed": seed,
        "n_levels": n_levels,
        "num_leaves": num_leaves,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        **metrics,
    }


def run_experiment(config: ZeroInflatedBenchmarkConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for q in config.q_values:
        for replication in range(config.replications):
            seed = config.random_seed_offset + replication
            rows.append(run_once(config=config, q=q, seed=seed))
    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby("q")
        .agg(
            mad_mean=("mad", "mean"),
            mad_std=("mad", "std"),
            mae_mean=("mae", "mean"),
            rmse_mean=("rmse", "mean"),
            spearman_mean=("spearman", "mean"),
            ndcg_mean=("ndcg_at_k", "mean"),
            gini_mean=("normalized_gini", "mean"),
        )
        .reset_index()
    )
    return per_run, summary


def save_experiment(
    per_run: pd.DataFrame,
    summary: pd.DataFrame,
    config: ZeroInflatedBenchmarkConfig,
    output_dir: Path | None = None,
) -> Path:
    base_dir = output_dir or RESULTS_DIR / "benchmarks" / "zero_inflated_tweedie"
    ensure_dir(base_dir)
    write_csv(base_dir / "per_run.csv", per_run)
    write_csv(base_dir / "summary.csv", summary)
    write_json(base_dir / "config.json", asdict(config))
    return base_dir


def default_output_dir() -> Path:
    return RESULTS_DIR / "benchmarks" / "zero_inflated_tweedie"
