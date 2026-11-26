# Exportar features del primer diagnóstico de un caso de desarrollo

import argparse
import json
from pathlib import Path
from time import perf_counter

from src.configuration import TopologyConfig
from src.evaluation import IncidentSplit
from src.feature_builder import FEATURE_SCHEMA_VERSION, FeatureBuilder, assign_root_cause_labels
from src.ingestion import RCAEvalLoader
from src.window_detector import WindowDetector


def main() -> None:
    parser = argparse.ArgumentParser(description='Construir features sin entrenar ni imputar')
    parser.add_argument('--incident', required=True)
    parser.add_argument('--config', default='config/topology.json')
    parser.add_argument('--split', default='config/incident_split.json')
    parser.add_argument('--output', help='Directorio nuevo; por defecto data/processed/features/<incident>')
    parser.add_argument('--labels', action='store_true', help='Añadir y offline; rechaza ventanas preinyección')
    args = parser.parse_args()
    start = perf_counter()
    try:
        config = TopologyConfig(args.config)
        loader = RCAEvalLoader(configuration=config)
        metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
        split = IncidentSplit.load(args.split, metadata)
        config.validate_calibration_split(split.development, split.test, split.metadata_sha256)
        if args.incident not in split.development:
            raise ValueError('Este comando solo procesa incidentes de desarrollo')
        diagnosis = next((event for update in WindowDetector(config).run(loader.load_metrics(args.incident))
                          for event in update.events if event.kind == 'DIAGNOSIS_DUE'), None)
        if diagnosis is None:
            raise ValueError('No hay diagnóstico completo; el caso no genera muestras RCA y debe reportarse')
        batch = FeatureBuilder(config).build(diagnosis)
        labels = assign_root_cause_labels(batch, loader.load_metadata(args.incident)) if args.labels else None
        output = Path(args.output) if args.output else Path('data/processed/features') / args.incident
        output.mkdir(parents=True, exist_ok=False)
        batch.X.to_parquet(output / 'X.parquet', index=False)
        batch.identifiers.to_parquet(output / 'identifiers.parquet', index=False)
        if labels is not None:
            labels.to_frame().to_parquet(output / 'y.parquet', index=False)
        manifest = {
            'feature_schema_version': FEATURE_SCHEMA_VERSION, 'incident_id': args.incident,
            'alert_id': diagnosis.alert_id, 'diagnosis_timestamp': diagnosis.timestamp,
            'selection': 'first_complete_diagnosis_without_using_labels',
            'rows': len(batch.X), 'feature_names': batch.X.columns.tolist(),
            'labeled': labels is not None,
            'notes': ['El notebook selecciona columnas finitas y no constantes en train.',
                      'No es el dataset completo de entrenamiento; no oculta incidentes sin diagnóstico.'],
        }
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'No se pudieron construir features: {exc}\n')
    print(f'Incidente: {args.incident}; alerta: {diagnosis.alert_id}')
    print(f'Servicios: {len(batch.X)}; features: {len(batch.X.columns)}; NaN: {int(batch.X.isna().sum().sum())}')
    if labels is not None:
        print(f'Causa raíz: {int(labels.sum())}; resto: {int(labels.eq(0).sum())}')
    print(f'Archivos guardados en {output}; duración: {perf_counter() - start:.3f} s')


if __name__ == '__main__':
    main()
