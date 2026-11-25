# Preparación por fold, previa a imputación, balanceo y entrenamiento
#Los grupos se fijan con los 72 IDs

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.model_selection import GroupKFold

from .configuration import TopologyConfig
from .detector_calibration import fit_deviation_floors
from .evaluation import IncidentSplit
from .ingestion import RCAEvalLoader
from .window_association import build_development_dataset, save_associated_dataset


def prepare_development_folds(
    configuration: TopologyConfig, split_path: str | Path,
    output: str | Path, n_splits: int = 5,
) -> dict[str, object]:
    # Reajustar tolerancias y regenerar datasets sin consultar métricas de test.

    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or not 2 <= n_splits <= 72:
        raise ValueError('n_splits debe ser un entero entre 2 y 72')
    loader = RCAEvalLoader(configuration=configuration)
    metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
    split = IncidentSplit.load(split_path, metadata)
    ids = np.array(sorted(split.development))
    # Una fila provisional por incidente ANTES de detectar o excluir ventanas.
    folds = [
        {'fold': number, 'train': ids[train].tolist(), 'validation': ids[val].tolist()}
        for number, (train, val) in enumerate(
            GroupKFold(n_splits=n_splits).split(ids, groups=ids), 1,
        )
    ]
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'folds.json').write_text(json.dumps({
        'schema_version': 1, 'method': 'GroupKFold', 'n_splits': n_splits,
        'metadata_sha256': split.metadata_sha256, 'folds': folds,
    }, indent=2) + '\n')
    (destination / 'incident_split.json').write_text(Path(split_path).read_text())
    document = json.loads(configuration.path.read_text())
    document['data_config']['root'] = str(configuration.data_root)
    document.pop('detector_calibration_file', None)
    # Esta configuración inicial no carga tolerancias aprendidas globalmente.
    (destination / 'topology_uncalibrated.json').write_text(json.dumps(document, indent=2) + '\n')
    base = TopologyConfig(destination / 'topology_uncalibrated.json')
    results: list[dict[str, object]] = []
    started = perf_counter()
    for fold in folds:
        number = fold['fold']
        train, validation = tuple(fold['train']), tuple(fold['validation'])
        directory = destination / f'fold_{number}'
        directory.mkdir()
        print(f'Fold {number}/{n_splits}: train={len(train)}, validación={len(validation)}', flush=True)
        calibration = fit_deviation_floors(base, split_path, fit_incidents=train)
        (directory / 'detector_calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
        calibrated = dict(document, detector_calibration_file='detector_calibration.json')
        (directory / 'topology.json').write_text(json.dumps(calibrated, indent=2) + '\n')
        config = TopologyConfig(directory / 'topology.json')
        config.validate_fold_calibration(train, validation, split.development, split.metadata_sha256)
        result: dict[str, object] = {'fold': number}
        train_columns: list[str] | None = None
        for role in ('train', 'validation'):
            dataset = build_development_dataset(
                config, split_path, fold_train=train, fold_validation=validation, role=role,
            )
            columns = dataset.features.X.columns.tolist()
            if role == 'train':
                train_columns = columns
            elif columns != train_columns:
                raise ValueError('Train y validación tienen esquemas de features distintos')
            manifest = save_associated_dataset(dataset, directory / role, config, split_path)
            # Complementar el manifiesto común con la procedencia específica.
            manifest.update(kind='fold_supervised_dataset', role=role, fold=number,
                            calibration_fit_incidents=list(train),
                            notes=[
                                'Tolerancias ajustadas exclusivamente con train del fold.',
                                'Sin imputación ni submuestreo; conservar candidatos en validación.',
                                'Ranking futuro condicionado a ventanas asociables; reportar cobertura aparte.',
                                'Reglas del detector seleccionadas previamente en desarrollo; CV exploratoria.',
                            ])
            (directory / role / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            result[role] = {key: manifest[key] for key in (
                'attempted_incidents', 'associated_incidents', 'samples',
                'false_alerts_before_injection', 'incident_status_counts',
            )}
        results.append(result)
    report: dict[str, object] = {
        'schema_version': 1, 'stage': 'fold_data_preparation', 'folds': results,
        'n_splits': n_splits, 'elapsed_sec': perf_counter() - started,
        'status': 'complete', 'test_metrics_loaded': False,
        'limitation': 'CV exploratoria: reglas del detector seleccionadas anteriormente en desarrollo.',
    }
    (destination / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
