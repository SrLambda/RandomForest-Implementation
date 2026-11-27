"""Evaluate the reserved incidents with the exported development forest."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from .configuration import TopologyConfig
from .forest_experiment import supervised_tables
from .metrics import evaluate_ranking
from .numpy_forest import RandomForest
from .window_association import build_holdout_dataset


def evaluate_final_forest(
    bundle: Path, output: Path, *, data_root: Path | None = None,
) -> dict[str, object]:
    """Score the holdout and save a report without fitting or selecting features."""
    if output.exists():
        raise FileExistsError(f"El informe ya existe: {output}")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    model_path = bundle / "forest.json"
    features = json.loads(model_path.read_text(encoding="utf-8"))["features"]
    if manifest.get("training_partition") != "development" or manifest["features"] != features:
        raise ValueError("El bosque y el manifiesto de desarrollo no coinciden.")
    forest = RandomForest.from_export(model_path)
    if forest.n_trees != manifest["n_trees"]:
        raise ValueError("El número de árboles no coincide con el manifiesto.")
    dataset = build_holdout_dataset(
        TopologyConfig(bundle / "topology.json"), bundle / "incident_split.json", data_root=data_root,
    )
    X, y = supervised_tables(dataset)
    X = X.reindex(columns=features) if X.empty else X.loc[:, features]
    ranking = evaluate_ranking(X, y, dataset.incidents, forest.predict)
    audit = dataset.incidents.merge(
        ranking.incidents.drop(columns=["false_alerts_before_injection"]),
        on="incident_id", validate="one_to_one",
    )
    summary = {key: None if pd.isna(value) else value for key, value in ranking.summary.items()}
    ranks = audit.root_rank.dropna()
    report = {
        "partition": "test",
        "artifact_sha256": {
            name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
            for name in ("forest.json", "manifest.json", "topology.json", "incident_split.json",
                         "detector_calibration.json") if (bundle / name).is_file()
        },
        "conditional_ranking": summary,
        "outcomes": audit.scoring_status.value_counts().to_dict(),
        "association_status": audit.association_status.value_counts().to_dict(),
        "incidents_with_false_alerts": int(audit.false_alerts_before_injection.gt(0).sum()),
        "all_reserved_incidents": {
            "top_1": float(ranks.le(1).sum() / len(audit)),
            "top_3": float(ranks.le(3).sum() / len(audit)),
            "mrr": float((1 / ranks).sum() / len(audit)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".final-eval-") as temporary:
        results = Path(temporary) / "results"
        results.mkdir()
        audit.to_csv(results / "incident_audit.csv", index=False)
        dataset.events.to_csv(results / "event_audit.csv", index=False)
        dataset.windows.to_csv(results / "window_audit.csv", index=False)
        ranking.confusion.to_csv(results / "confusion.csv")
        (results / "summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
        )
        if output.exists():
            raise FileExistsError(f"El informe ya existe: {output}")
        results.rename(output)
    return report
