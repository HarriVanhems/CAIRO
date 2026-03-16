from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cairo.paths import DATA_DIR


BENCHMARKS_DIR = DATA_DIR / "raw" / "benchmarks"


@dataclass
class DatasetBundle:
    name: str
    target_name: str
    features: pd.DataFrame
    target: pd.Series
    notes: str = ""


def _dataset_dir(name: str) -> Path:
    return BENCHMARKS_DIR / name


def load_abalone() -> DatasetBundle:
    columns = [
        "sex",
        "length",
        "diameter",
        "height",
        "whole_weight",
        "shucked_weight",
        "viscera_weight",
        "shell_weight",
        "rings",
    ]
    frame = pd.read_csv(_dataset_dir("abalone") / "abalone.data", header=None, names=columns)
    target = frame.pop("rings")
    features = pd.get_dummies(frame, columns=["sex"], drop_first=False)
    return DatasetBundle("abalone", "rings", features, target, notes="Sex one-hot encoded.")


def load_auto_mpg() -> DatasetBundle:
    rows = []
    with (_dataset_dir("auto_mpg") / "auto-mpg.data").open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=8)
            if len(parts) != 9:
                continue
            rows.append(parts)
    columns = [
        "mpg",
        "cylinders",
        "displacement",
        "horsepower",
        "weight",
        "acceleration",
        "model_year",
        "origin",
        "car_name",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for column in columns[:-1]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["origin"] = frame["origin"].astype("Int64").astype(str)
    target = frame.pop("mpg")
    features = pd.get_dummies(frame, columns=["origin"], drop_first=False)
    features = features.drop(columns=["car_name"])
    return DatasetBundle(
        "auto_mpg",
        "mpg",
        features,
        target,
        notes="Car name dropped; origin one-hot encoded; horsepower missing values kept as NaN.",
    )


def load_communities_and_crime() -> DatasetBundle:
    frame = pd.read_csv(
        _dataset_dir("communities_and_crime") / "communities.data",
        header=None,
        na_values="?",
    )
    target = frame.iloc[:, -1].rename("violent_crimes_per_pop")
    features = frame.iloc[:, 5:-1].copy()
    features.columns = [f"x_{idx}" for idx in range(features.shape[1])]
    return DatasetBundle(
        "communities_and_crime",
        "violent_crimes_per_pop",
        features,
        target,
        notes="Dropped the first five identifier columns; missing values preserved as NaN.",
    )


def load_computer_hardware() -> DatasetBundle:
    columns = [
        "vendor_name",
        "model_name",
        "myct",
        "mmin",
        "mmax",
        "cach",
        "chmin",
        "chmax",
        "prp",
        "erp",
    ]
    frame = pd.read_csv(
        _dataset_dir("computer_hardware") / "machine.data",
        header=None,
        names=columns,
    )
    target = frame.pop("prp")
    features = frame.drop(columns=["vendor_name", "model_name", "erp"])
    return DatasetBundle(
        "computer_hardware",
        "prp",
        features,
        target,
        notes="Dropped vendor/model identifiers and ERP leakage column.",
    )


def load_concrete_slump_test() -> DatasetBundle:
    frame = pd.read_csv(_dataset_dir("concrete_slump_test") / "slump_test.data")
    target = frame.pop("SLUMP(cm)")
    features = frame.drop(columns=["No"])
    return DatasetBundle(
        "concrete_slump_test",
        "SLUMP(cm)",
        features,
        target,
        notes="Dropped row identifier column.",
    )


def load_parkinsons_telemonitoring() -> DatasetBundle:
    frame = pd.read_csv(_dataset_dir("parkinsons_telemonitoring") / "parkinsons_updrs.data")
    target = frame.pop("total_UPDRS")
    features = frame.drop(columns=["subject#"])
    return DatasetBundle(
        "parkinsons_telemonitoring",
        "total_UPDRS",
        features,
        target,
        notes="Dropped subject identifier column.",
    )


def load_real_datasets() -> list[DatasetBundle]:
    return [
        load_abalone(),
        load_auto_mpg(),
        load_communities_and_crime(),
        load_computer_hardware(),
        load_concrete_slump_test(),
        load_parkinsons_telemonitoring(),
    ]


def dataset_summary_table(datasets: list[DatasetBundle]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        rows.append(
            {
                "dataset": dataset.name,
                "n_rows": int(len(dataset.target)),
                "n_features": int(dataset.features.shape[1]),
                "target": dataset.target_name,
                "missing_feature_values": int(dataset.features.isna().sum().sum()),
                "notes": dataset.notes,
            }
        )
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
