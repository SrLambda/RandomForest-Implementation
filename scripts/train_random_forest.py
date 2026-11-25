"""Entrenar un Random Forest por fold y puntuar todos sus candidatos de validación."""

import argparse

from src.model import ForestParameters
from src.model_training import train_development_forests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/processed/prepared_development_folds')
    parser.add_argument('--output', default='artifacts/random_forest/development')
    parser.add_argument('--config', default='config/random_forest.json')
    args = parser.parse_args()
    try:
        report = train_development_forests(args.source, args.output, ForestParameters.load(args.config))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudo entrenar el Random Forest: {exc}\n')
    print(f"Puntuados {report['scored_validation_incidents']}/{report['attempted_validation_incidents']} "
          f"incidentes; {report['validation_rows']} candidatos en {report['elapsed_sec']:.2f} s")
    print(f'Archivos en {args.output}. Pendientes: ranking y métricas por incidente.')


if __name__ == '__main__':
    main()
