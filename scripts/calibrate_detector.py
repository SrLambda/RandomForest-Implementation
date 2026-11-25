# Generar tolerancias y reducción de falsos positivos

import argparse
import json
from pathlib import Path

from src.configuration import TopologyConfig
from src.detector_calibration import fit_deviation_floors


def main() -> None:
    parser = argparse.ArgumentParser(description='Calibrar tolerancias solo con desarrollo')
    parser.add_argument('--config', default='config/topology.json')
    parser.add_argument('--split', default='config/incident_split.json')
    parser.add_argument('--output', default='config/detector_calibration.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quantile', type=float, default=.999)
    parser.add_argument('--margin', type=float, default=2.)
    args = parser.parse_args()
    try:
        result = fit_deviation_floors(TopologyConfig(args.config), args.split, args.seed, args.quantile, args.margin)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x',encoding='utf-8') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudo calibrar: {exc}\n')
    print(f"Ajuste: {len(result['fit_incidents'])}; validación interna: {len(result['validation_incidents'])}")
    print(f"Métricas calibradas: {len(result['minimum_deviation'])}; sin calibrar: {result['uncalibrated_metrics']}")
    print(f'Calibración guardada en {args.output}; no se sobrescriben archivos existentes.')


if __name__ == '__main__':
    main()
