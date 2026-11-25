# Asociar ventanas e incidentes y exportar el dataset de desarrollo

import argparse
from pathlib import Path

from src.configuration import TopologyConfig
from src.window_association import build_development_dataset, save_associated_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='Asociación offline RCA: solo desarrollo, con auditoría')
    parser.add_argument('--config', default='config/topology.json')
    parser.add_argument('--split', default='config/incident_split.json')
    parser.add_argument('--output', default='data/processed/associated_development')
    args = parser.parse_args()
    try:
        if Path(args.output).exists():
            raise FileExistsError('La salida ya existe; elegir otro directorio con --output')
        configuration = TopologyConfig(args.config)
        dataset = build_development_dataset(configuration, args.split)
        manifest = save_associated_dataset(dataset, args.output, configuration, args.split)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudo construir el dataset: {exc}\n')
    print(f"Incidentes asociados: {manifest['associated_incidents']}/{manifest['attempted_incidents']}")
    print(f"Estados: {manifest['incident_status_counts']}")
    print(f"Muestras: {manifest['samples']}; features: {manifest['features']}")
    print(f"Causas raíz: {manifest['positive_samples']}; resto: {manifest['negative_samples']}")
    print(f"Falsas alertas anteriores conservadas en auditoría: {manifest['false_alerts_before_injection']}")
    print(f"Ventanas: {manifest['window_status_counts']}")
    print(f"Tiempo: {manifest['elapsed_sec']:.3f} s; archivos en {args.output}")
    print('Dataset exploratorio: no entrenar una validación independiente reutilizando la calibración global.')


if __name__ == '__main__':
    main()
