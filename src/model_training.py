"""Entrenamiento por fold y scores fuera de muestra, sin consultar test final."""

import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn

from .data_preparation import TrainMedianImputer, load_prepared_samples
from .model import ForestParameters, RootCauseForest


def train_development_forests(
    source: str | Path, output: str | Path, parameters: ForestParameters,
) -> dict[str, object]:
    """Entrenar un bosque por fold y puntuar su validación completa.

    Reutiliza el train ya imputado/submuestreado sin reajustar el imputador ni
    balancear validación. Los scores se unen a IDs/y solo DESPUÉS de predecir.
    Guarda trazabilidad e incidentes sin muestra; Top-k/MRR pertenecen a metrics.
    """
    started = perf_counter()
    source, destination = Path(source).resolve(), Path(output).resolve()
    summary = json.loads((source / 'summary.json').read_text())
    folds_doc = json.loads((source / 'folds.json').read_text())
    split = json.loads((source / 'incident_split.json').read_text())
    if (summary.get('stage') != 'train_data_preparation' or summary.get('status') != 'complete'
            or folds_doc.get('metadata_sha256') != split.get('metadata_sha256')):
        raise ValueError('Se requieren folds preparados completos y una partición compatible')
    folds = folds_doc['folds']
    validation_ids = [key for fold in folds for key in fold['validation']]
    if (len(validation_ids) != len(set(validation_ids))
            or set(validation_ids) != set(split['development'])
            or set(validation_ids) & set(split['test'])
            or len({fold['fold'] for fold in folds}) != len(folds)):
        raise ValueError('Folds inválidos: validación debe cubrir desarrollo una sola vez')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'random_forest.json').write_text(json.dumps(parameters.to_dict(), indent=2) + '\n')
    for name in ('folds.json', 'incident_split.json'):
        (destination / name).write_text((source / name).read_text())
    results, all_scores, all_audits = [], [], []
    for fold in folds:
        train_ids, val_ids = set(fold['train']), set(fold['validation'])
        if train_ids & val_ids or train_ids | val_ids != set(split['development']):
            raise ValueError('Train y validación deben ser disjuntos y cubrir desarrollo')
        folder = source / f"fold_{fold['fold']}"
        manifest = json.loads((folder / 'manifest.json').read_text())
        if (manifest['fold'] != fold['fold'] or set(manifest['train_incidents']) != train_ids
                or set(manifest['validation_incidents']) != val_ids):
            raise ValueError('El manifiesto preparado difiere de los folds')
        train = load_prepared_samples(folder / 'train')
        validation = load_prepared_samples(folder / 'validation')
        if not set(train.groups) <= train_ids or not set(validation.groups) <= val_ids:
            raise ValueError('Las muestras contienen incidentes ajenos al rol')
        imputer_state = json.loads((folder / 'imputer.json').read_text())
        imputer = TrainMedianImputer.from_state(imputer_state)
        # Las muestras están imputadas: transform debe ser identidad, no fit.
        for samples in (train, validation):
            if not imputer.transform(samples.X).equals(samples.X):
                raise ValueError('Los datos de entrada no están completamente imputados')
        audit_source = Path(manifest['audit_source'])
        raw_ids = pd.read_parquet(audit_source / 'validation/identifiers.parquet')
        raw_y = pd.read_parquet(audit_source / 'validation/y.parquet')['y']
        positions = pd.read_parquet(folder / 'validation/source_rows.parquet')['source_row']
        if (not raw_ids.equals(validation.identifiers) or not raw_y.equals(validation.y)
                or not np.array_equal(positions.to_numpy(), np.arange(len(raw_ids)))):
            raise ValueError('La validación fue submuestreada o alterada')
        audit = pd.read_csv(audit_source / 'validation/incident_audit.csv')
        if (len(audit) != len(val_ids) or set(audit.incident_id) != val_ids
                or set(audit.loc[audit.status.eq('associated'), 'incident_id']) != set(validation.groups)):
            raise ValueError('La auditoría de validación no coincide con las muestras')
        if len(train.X) != manifest['train_rows_after'] or len(validation.X) != manifest['validation_rows']:
            raise ValueError('Cambió el número de filas preparadas')
        target = destination / folder.name
        target.mkdir()
        forest = RootCauseForest(parameters).fit(train.X, train.y)
        prediction_start = perf_counter()
        scores = forest.predict_proba(validation.X)
        prediction_sec = perf_counter() - prediction_start
        predictions = validation.identifiers.copy()
        predictions['root_cause_score'] = scores
        predictions['y_true'] = validation.y
        predictions['fold'] = fold['fold']
        predictions.to_parquet(target / 'validation_scores.parquet', index=False)
        forest.save(target / 'model.pkl')
        forest.feature_importances().rename_axis('feature').to_csv(target / 'feature_importances.csv')
        (target / 'imputer.json').write_text(json.dumps(imputer_state, indent=2) + '\n')
        (target / 'prepared_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        audit['fold'] = fold['fold']
        audit.to_csv(target / 'incident_audit.csv', index=False)
        # Huellas de los archivos efectivamente consumidos; no son features.
        fingerprints = {
            str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for role in ('train', 'validation')
            for path in sorted((folder / role).glob('*.parquet'))
        }
        result = {
            'fold': fold['fold'], 'source': str(folder), 'parameters': parameters.to_dict(),
            'train_rows': len(train.X), 'train_positive_rows': int(train.y.sum()),
            'validation_rows': len(validation.X), 'validation_incidents': len(val_ids),
            'associated_validation_incidents': validation.groups.nunique(),
            'false_alerts_before_injection': int(audit.false_alerts_before_injection.sum()),
            'features': train.X.columns.tolist(), 'fit_elapsed_sec': forest.fit_elapsed_sec,
            'prediction_elapsed_sec': prediction_sec, 'input_sha256': fingerprints,
        }
        (target / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        all_scores.append(predictions)
        all_audits.append(audit)
        results.append(result)
        print(f"Fold {fold['fold']}: train={len(train.X)}, scores={len(predictions)}, "
              f"incidentes={validation.groups.nunique()}/{len(val_ids)}", flush=True)
    predictions = pd.concat(all_scores, ignore_index=True)
    if predictions.duplicated(['incident_id', 'service']).any():
        raise ValueError('Hay servicios repetidos entre folds de validación')
    predictions.to_parquet(destination / 'validation_scores.parquet', index=False)
    pd.concat(all_audits, ignore_index=True).to_csv(destination / 'incident_audit.csv', index=False)
    report = {
        'schema_version': 1, 'stage': 'random_forest_training', 'status': 'complete',
        'source': str(source), 'parameters': parameters.to_dict(), 'folds': results,
        'attempted_validation_incidents': len(validation_ids),
        'scored_validation_incidents': predictions.incident_id.nunique(),
        'validation_rows': len(predictions), 'elapsed_sec': perf_counter() - started,
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__},
        'notes': [
            'Scores de clase 1: no son probabilidades calibradas ni suman 1 por incidente.',
            'Train preparado previamente; validación completa sin submuestreo.',
            'Ranking y métricas pendientes; conservar omisiones y falsas alertas en auditoría.',
            'Modelos por fold de desarrollo, no un modelo final para producción.',
            'CV exploratoria: reglas del detector seleccionadas anteriormente con desarrollo.',
        ],
    }
    (destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
