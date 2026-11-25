# Inspección operativa de un caso: python -m scripts.run_window_detector.

import argparse
from collections import Counter
from time import perf_counter

from src.configuration import TopologyConfig
from src.evaluation import IncidentSplit
from src.ingestion import RCAEvalLoader
from src.window_detector import WindowDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducir ventanas sin entrenar ni ejecutar RCA")
    parser.add_argument("--config", default="config/topology.json")
    parser.add_argument("--incident", required=True, help="Identificador local del caso")
    parser.add_argument("--split", default="config/incident_split.json")
    parser.add_argument("--partition", choices=("development", "test"), default="development",
                        help="Usar test únicamente para evaluación final, nunca para calibrar")
    args = parser.parse_args()
    start = perf_counter()
    try:
        config = TopologyConfig(args.config)
        loader = RCAEvalLoader(configuration=config)
        metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
        split = IncidentSplit.load(args.split, metadata)
        allowed = split.development if args.partition == "development" else split.test
        if args.incident not in allowed:
            raise ValueError(f"{args.incident} no pertenece a la partición {args.partition}")
        metrics = loader.load_metrics(args.incident)
        states: Counter[str] = Counter()
        unavailable = 0
        for update in WindowDetector(config).run(metrics):
            states[update.state.value] += 1
            unavailable += bool(update.unavailable_metrics)
            for event in update.events:
                print(
                    f"{event.timestamp} {event.kind} alert={event.alert_id} "
                    f"recovered={event.recovered}", flush=True,
                )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"No se pudo procesar el caso: {exc}\n")
    print(f"Observaciones por estado: {dict(states)}")
    print(f"Observaciones con métricas no evaluables: {unavailable}")
    print(f"Duración de carga y procesamiento: {perf_counter() - start:.3f} s")
    print("DIAGNOSIS_DUE solicita el diagnóstico; todavía no hay modelo RCA conectado.")


if __name__ == "__main__":
    main()
