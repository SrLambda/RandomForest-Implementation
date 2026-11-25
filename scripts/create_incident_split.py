# Crear o inspeccionar la reserva de test sin cargar métricas

import argparse
from collections import Counter
from pathlib import Path

from src.configuration import TopologyConfig
from src.evaluation import IncidentSplit
from src.ingestion import RCAEvalLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Partición persistida de RE2-SS: 72 desarrollo / 18 test")
    parser.add_argument("--config", default="config/topology.json")
    parser.add_argument("--output", default="config/incident_split.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        loader = RCAEvalLoader(configuration=TopologyConfig(args.config))
        metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
        path = Path(args.output)
        if path.exists():
            split = IncidentSplit.load(path, metadata)
            print(f"Partición existente validada y conservada: {path}")
        else:
            split = IncidentSplit.create(metadata, args.seed)
            split.save(path)
            print(f"Partición creada: {path}")
        print(f"Semilla guardada: {split.seed}")
        print(f"Desarrollo: {len(split.development)}; test: {len(split.test)}")
        by_id = {item.incident_id: item for item in metadata}
        for name, ids in (("Desarrollo", split.development), ("Test", split.test)):
            print(f"{name}, servicios: {dict(sorted(Counter(by_id[key].root_cause_service for key in ids).items()))}")
            print(f"{name}, fallos: {dict(sorted(Counter(by_id[key].fault_type for key in ids).items()))}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"No se pudo preparar la partición: {exc}\n")


if __name__ == "__main__":
    main()
