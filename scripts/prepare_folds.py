# Preparar datasets de desarrollo con calibración exclusiva de train por fold

import argparse

from src.configuration import TopologyConfig
from src.cross_validation import prepare_development_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/topology.json')
    parser.add_argument('--split', default='config/incident_split.json')
    parser.add_argument('--output', default='data/processed/development_folds')
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()
    try:
        report = prepare_development_folds(
            TopologyConfig(args.config), args.split, args.output, args.n_splits,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudieron preparar los folds: {exc}\n')
    print(f"Preparados {report['n_splits']} folds en {report['elapsed_sec']:.2f} s: {args.output}")
    print('Pendientes: imputación, undersampling de train, entrenamiento y ranking.')


if __name__ == '__main__':
    main()
