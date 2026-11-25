# Evaluación descriptiva y calibración solo en los 72 casos de desarrollo

import argparse
import json

from src.configuration import TopologyConfig
from src.detector_evaluation import evaluate_development


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluar detector exclusivamente en desarrollo')
    parser.add_argument('--config', default='config/topology.json')
    parser.add_argument('--split', default='config/incident_split.json')
    parser.add_argument('--output', default='reports/detector/development')
    args = parser.parse_args()
    try:
        summary = evaluate_development(TopologyConfig(args.config), args.split, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudo evaluar: {exc}\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'false_alert_trigger_metrics'}, indent=2))
    print(f'Reportes guardados en {args.output}')


if __name__ == '__main__':
    main()
