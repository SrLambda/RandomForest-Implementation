# Imputar por fold y submuestrear negativos exclusivamente en train

import argparse

from src.data_preparation import prepare_training_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/processed/development_folds')
    parser.add_argument('--output', default='data/processed/prepared_development_folds')
    parser.add_argument('--negative-ratio', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    try:
        result = prepare_training_folds(args.source, args.output, negative_ratio=args.negative_ratio, seed=args.seed)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudieron preparar los datos: {exc}\n')
    print(f"Preparados {len(result['folds'])} folds en {result['elapsed_sec']:.2f} s: {args.output}")


if __name__ == '__main__':
    main()
