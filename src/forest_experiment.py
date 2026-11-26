"""Development-only experiments for the notebook forest, without imputation."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import pandas as pd

from .configuration import TopologyConfig
from .metrics import RankingResult, evaluate_ranking
from .numpy_forest import RandomForest
from .window_association import AssociatedDataset, build_development_dataset


@dataclass(frozen=True)
class ExperimentResult:
    forest: RandomForest
    X_train: pd.DataFrame
    train_audit: pd.DataFrame
    validation: RankingResult
    fit_seconds: float
    evaluation_seconds: float


def load_development(project_root: Path, *, data_root: Path | None = None) -> AssociatedDataset:
    """Run the existing detector/association pipeline with development telemetry only."""
    return build_development_dataset(
        TopologyConfig(project_root / "config/topology.json"),
        project_root / "config/incident_split.json",
        data_root=data_root,
    )


def select_features(X_train: pd.DataFrame) -> pd.Index:
    """Keep only columns that are finite and nonconstant throughout train."""
    finite = np.isfinite(X_train.to_numpy(dtype=float, na_value=np.nan)).all(axis=0)
    columns = X_train.columns[finite & X_train.nunique().gt(1).to_numpy()]
    if columns.empty:
        raise ValueError("No finite, nonconstant training features remain.")
    return columns


def supervised_tables(dataset: AssociatedDataset) -> tuple[pd.DataFrame, pd.Series]:
    """Expose the associated rows with incident/service identity in the index only."""
    index = pd.MultiIndex.from_frame(dataset.features.identifiers[["incident_id", "service"]])
    return dataset.features.X.set_axis(index), dataset.y.set_axis(index)


def run_experiment(
    dataset: AssociatedDataset, *, train_incidents: int, validation_incidents: int = 12,
    split_seed: int = 42, n_trees: int = 10, forest_seed: int = 42,
) -> ExperimentResult:
    """Take nested train prefixes and a fixed validation prefix of shuffled incident IDs."""
    X, y = supervised_tables(dataset)
    ids = dataset.incidents["incident_id"].sort_values().to_numpy()
    if (type(train_incidents) is not int or type(validation_incidents) is not int
            or min(train_incidents, validation_incidents) < 1
            or train_incidents + validation_incidents > len(ids)):
        raise ValueError("Train and validation sizes must be positive and fit within development.")
    ids = np.random.default_rng(split_seed).permutation(ids)
    validation_ids = ids[:validation_incidents]
    train_ids = ids[validation_incidents:validation_incidents + train_incidents]
    row_ids = X.index.get_level_values("incident_id")
    train_mask = row_ids.isin(train_ids)
    validation_mask = row_ids.isin(validation_ids)
    train_audit = dataset.incidents.loc[dataset.incidents.incident_id.isin(train_ids)].copy()
    validation_audit = dataset.incidents.loc[dataset.incidents.incident_id.isin(validation_ids)]
    X_train = X.loc[train_mask]
    columns = select_features(X_train)
    X_train = X_train.loc[:, columns]
    start = perf_counter()
    forest = RandomForest(n_trees=n_trees, seed=forest_seed).fit(X_train, y.loc[train_mask].to_frame())
    fit_seconds = perf_counter() - start
    start = perf_counter()
    result = evaluate_ranking(
        X.loc[validation_mask, columns], y.loc[validation_mask], validation_audit, forest.predict,
    )
    return ExperimentResult(forest, X_train, train_audit, result, fit_seconds, perf_counter() - start)


def demo_roundtrip(forest: RandomForest, rows: pd.DataFrame) -> pd.DataFrame:
    """Show per-tree and aggregate scores after a temporary JSON export."""
    with TemporaryDirectory(prefix="numpy-forest-") as directory:
        path = Path(directory) / "forest.json"
        forest.export(path)
        loaded = RandomForest.from_export(path)
    scores = pd.DataFrame(
        [loaded.predict_tree_scores(row) for _, row in rows.iterrows()],
        index=rows.index,
        columns=[f"tree_{i}" for i in range(1, loaded.n_trees + 1)],
    )
    scores["forest_score"] = [loaded.predict(row) for _, row in rows.iterrows()]
    return scores
