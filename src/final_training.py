"""Train and export the fixed RCA forest using only development incidents."""

import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .configuration import TopologyConfig
from .forest_experiment import select_features, supervised_tables
from .numpy_forest import RandomForest
from .window_association import build_development_dataset


def train_final_forest(
    project_root: Path, output: Path, *, data_root: Path | None = None,
) -> dict:
    """Export a new bundle with the forest, input configuration and train audit."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"El destino ya existe: {output}")

    config_path = project_root / "config/topology.json"
    split_path = project_root / "config/incident_split.json"
    configuration = TopologyConfig(config_path)
    calibration = configuration.detector_calibration
    topology = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    root = configuration.data_root if data_root is None else data_root.resolve()
    dataset = build_development_dataset(configuration, split_path, data_root=root)
    X, y = supervised_tables(dataset)
    if set(dataset.incidents.incident_id) != set(split["development"]):
        raise ValueError("La auditoría no coincide con los incidentes de desarrollo.")
    X = X.loc[:, select_features(X)]
    forest = RandomForest(n_trees=10, seed=42).fit(X, y.to_frame())

    topology["data_config"]["root"] = str(root)
    topology.pop("detector_calibration_file", None)
    if calibration is not None:
        topology["detector_calibration_file"] = "detector_calibration.json"
    manifest = {
        "schema_version": 1,
        "dataset": split["dataset"],
        "training_partition": "development",
        "development_incidents": len(dataset.incidents),
        "associated_incidents": X.index.get_level_values("incident_id").nunique(),
        "training_rows": len(X),
        "features": X.columns.tolist(),
        "feature_selection": "finite_and_nonconstant_in_associated_development",
        "n_trees": forest.n_trees,
        "seed": forest.seed,
        "max_depth": 8,
        "min_samples_leaf": 2,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".forest-") as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        forest.export(bundle / "forest.json")
        snapshots = {
            "topology.json": topology,
            "incident_split.json": split,
            "manifest.json": manifest,
        }
        if calibration is not None:
            snapshots["detector_calibration.json"] = calibration
        for filename, document in snapshots.items():
            (bundle / filename).write_text(
                json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        dataset.incidents.to_csv(bundle / "development_audit.csv", index=False)
        loaded = RandomForest.from_export(bundle / "forest.json")
        for _, row in X.head(3).iterrows():
            if loaded.predict(row) != forest.predict(row):
                raise RuntimeError("El bosque recargado cambió una puntuación de referencia.")
        if output.exists():
            raise FileExistsError(f"El destino ya existe: {output}")
        bundle.rename(output)
    return manifest
