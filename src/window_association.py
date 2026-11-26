# Asociación offline de diagnósticos e incidentes con auditoría completa

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .configuration import TopologyConfig
from .evaluation import IncidentSplit
from .feature_builder import FEATURE_SCHEMA_VERSION, FeatureBatch, FeatureBuilder, assign_root_cause_labels
from .ingestion import IncidentMetadata, RCAEvalLoader
from .window_detector import DetectorEvent, WindowDetector

_WINDOW_COLUMNS = (
    'incident_id', 'alert_id', 'diagnosis_timestamp', 'suspicion_start_timestamp',
    'window_start_timestamp', 'window_end_timestamp', 'injection_timestamp',
    'status', 'selected', 'recovered',
)
_EVENT_COLUMNS = (
    'incident_id', 'alert_id', 'event', 'timestamp', 'start_timestamp',
    'injection_timestamp', 'offset_sec', 'recovered',
)


@dataclass(frozen=True, slots=True)
class WindowAssociation:
    """Primera ventana asociable y trazabilidad de todos los eventos del caso."""

    selected: DetectorEvent | None
    status: str
    windows: pd.DataFrame
    events: pd.DataFrame


@dataclass(frozen=True, slots=True)
class AssociatedDataset:
    """Muestras alineadas y auditorías; sin imputar ni balancear clases."""

    features: FeatureBatch
    y: pd.Series
    incidents: pd.DataFrame
    windows: pd.DataFrame
    events: pd.DataFrame
    elapsed_sec: float


def associate_windows(
    metadata: IncidentMetadata, events: Iterable[DetectorEvent],
) -> WindowAssociation:
    # Elegir la primera ventana completa que comienza después de inyección.

    selected: DetectorEvent | None = None
    windows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    completed_alerts: set[str] = set()
    last_timestamp: int | None = None
    for event in events:
        if event.incident_id != metadata.incident_id:
            raise ValueError('La secuencia contiene eventos de otro incidente')
        if last_timestamp is not None and event.timestamp < last_timestamp:
            raise ValueError('Los eventos deben estar ordenados cronológicamente')
        last_timestamp = event.timestamp
        event_rows.append({
            'incident_id': metadata.incident_id, 'alert_id': event.alert_id,
            'event': event.kind, 'timestamp': event.timestamp,
            'start_timestamp': event.start_timestamp,
            'injection_timestamp': metadata.injection_timestamp,
            'offset_sec': event.timestamp - metadata.injection_timestamp,
            'recovered': event.recovered,
        })
        if event.kind not in ('DIAGNOSIS_DUE', 'DIAGNOSIS_INCOMPLETE'):
            continue
        if not event.alert_id or event.start_timestamp is None:
            raise ValueError('El diagnóstico necesita alert_id e inicio de sospecha')
        window_start: int | None = None
        window_end: int | None = None
        choose = False
        if event.kind == 'DIAGNOSIS_INCOMPLETE':
            status = 'incomplete'
        else:
            if event.alert_id in completed_alerts:
                raise ValueError(f'Diagnóstico completo duplicado: {event.alert_id}')
            completed_alerts.add(event.alert_id)
            if event.observations is None or event.baseline is None or event.observations.empty or event.baseline.empty:
                raise ValueError('Diagnóstico completo sin baseline u observaciones')
            times = event.observations['time']
            window_start, window_end = int(times.min()), int(times.max())
            if window_end != event.timestamp or event.baseline.time.max() >= window_start:
                raise ValueError('Ventana futura, solapada o incompatible con diagnóstico')
            if event.start_timestamp > window_start:
                raise ValueError('La ventana comienza antes de la sospecha declarada')
            if event.timestamp < metadata.injection_timestamp:
                status = 'before_injection'
            elif min(window_start, event.start_timestamp) < metadata.injection_timestamp:
                status = 'crosses_injection'
            elif selected is None:
                status = 'selected'
                selected = event
                choose = True
            else:
                status = 'additional_post_injection'
        windows.append({
            'incident_id': metadata.incident_id, 'alert_id': event.alert_id,
            'diagnosis_timestamp': event.timestamp,
            'suspicion_start_timestamp': event.start_timestamp,
            'window_start_timestamp': window_start, 'window_end_timestamp': window_end,
            'injection_timestamp': metadata.injection_timestamp,
            'status': status, 'selected': choose, 'recovered': event.recovered,
        })
    kinds = {row['event'] for row in event_rows}
    if selected is not None:
        status = 'associated'
    elif 'WARMUP_INCOMPLETE' in kinds:
        status = 'warmup_incomplete'
    elif 'ALERT_CONFIRMED' not in kinds:
        status = 'no_confirmed_alert'
    elif 'DIAGNOSIS_INCOMPLETE' in kinds:
        status = 'no_complete_associable_diagnosis'
    else:
        status = 'no_associable_diagnosis'
    return WindowAssociation(
        selected, status,
        pd.DataFrame(windows, columns=list(_WINDOW_COLUMNS)),
        pd.DataFrame(event_rows, columns=list(_EVENT_COLUMNS)),
    )


def build_development_dataset(
    configuration: TopologyConfig,
    split_path: str | Path = 'config/incident_split.json',
    *, fold_train: tuple[str, ...] | None = None,
    fold_validation: tuple[str, ...] | None = None,
    role: str = 'development',
    data_root: str | Path | None = None,
) -> AssociatedDataset:
    """Construir una muestra por (incidente, servicio) solo en desarrollo.

    Por defecto usa la calibración exploratoria vigente. Para un fold, exige
    ambos conjuntos y comprueba que la calibración corresponde exactamente a
    train antes de cargar métricas del rol solicitado. No carga métricas de test.
    """
    start = perf_counter()
    loader = RCAEvalLoader(root=data_root, configuration=configuration)
    metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
    split = IncidentSplit.load(split_path, metadata)
    if role == 'development' and fold_train is None and fold_validation is None:
        configuration.validate_calibration_split(split.development, split.test, split.metadata_sha256)
        selected_ids = split.development
    elif role in ('train', 'validation') and fold_train is not None and fold_validation is not None:
        configuration.validate_fold_calibration(
            fold_train, fold_validation, split.development, split.metadata_sha256,
        )
        selected_ids = fold_train if role == 'train' else fold_validation
    else:
        raise ValueError('Especificar ambos conjuntos del fold y un rol válido')
    builder = FeatureBuilder(configuration)
    X_parts: list[pd.DataFrame] = []
    id_parts: list[pd.DataFrame] = []
    y_parts: list[pd.Series] = []
    incident_rows: list[dict[str, object]] = []
    window_parts: list[pd.DataFrame] = []
    event_parts: list[pd.DataFrame] = []
    schema: list[str] | None = None
    for number, incident_id in enumerate(selected_ids, 1):
        info = loader.load_metadata(incident_id)
        observations = loader.load_metrics(incident_id)
        events = (event for update in WindowDetector(configuration).run(observations) for event in update.events)
        association = associate_windows(info, events)
        window_parts.append(association.windows)
        event_parts.append(association.events)
        status = association.status
        sample_count = 0
        event = association.selected
        if event is not None:
            batch = builder.build(event)
            if not batch.identifiers.service.eq(info.root_cause_service).any():
                # No se cambia de ventana usando la presencia de la etiqueta.
                status = 'root_cause_not_observed'
            else:
                labels = assign_root_cause_labels(batch, info)
                if schema is None:
                    schema = batch.X.columns.tolist()
                if batch.X.columns.tolist() != schema:
                    raise ValueError(f'El esquema de features cambió en {incident_id}')
                X_parts.append(batch.X)
                id_parts.append(batch.identifiers)
                y_parts.append(labels)
                sample_count = len(batch.X)
        audit = association.events
        confirmed = audit.loc[audit.event.eq('ALERT_CONFIRMED')]
        incident_rows.append({
            'incident_id': incident_id, 'status': status,
            'root_cause_service': info.root_cause_service, 'fault_type': info.fault_type,
            'injection_timestamp': info.injection_timestamp,
            'selected_alert_id': event.alert_id if event is not None else None,
            'selected_diagnosis_timestamp': event.timestamp if event is not None else None,
            'false_alerts_before_injection': int(confirmed.timestamp.lt(info.injection_timestamp).sum()),
            'confirmed_alerts': len(confirmed),
            'diagnoses_before_injection': int(association.windows.status.eq('before_injection').sum()),
            'diagnoses_crossing_injection': int(association.windows.status.eq('crosses_injection').sum()),
            'additional_post_injection_diagnoses': int(association.windows.status.eq('additional_post_injection').sum()),
            'incomplete_diagnoses': int(association.windows.status.eq('incomplete').sum()),
            'supervised_samples': sample_count,
        })
        if number % 12 == 0:
            print(f'Asociación de desarrollo: {number}/{len(selected_ids)}', flush=True)
    if not X_parts:
        raise ValueError('No se generaron muestras supervisadas; revisar la asociación antes de exportar')
    X = pd.concat(X_parts, ignore_index=True)
    identifiers = pd.concat(id_parts, ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True).rename('y')
    if identifiers.duplicated(['incident_id', 'service']).any():
        raise ValueError('Hay muestras repetidas por incidente y servicio')
    if not X.index.equals(identifiers.index) or not X.index.equals(y.index):
        raise ValueError('X, identificadores y etiquetas perdieron alineación')
    if not y.isin([0, 1]).all() or not y.groupby(identifiers.incident_id).sum().eq(1).all():
        raise ValueError('Cada incidente asociado debe tener exactamente una causa raíz')
    if np.isinf(X.to_numpy()).any():
        raise ValueError('El dataset contiene features infinitas')
    return AssociatedDataset(
        FeatureBatch(X, identifiers), y, pd.DataFrame(incident_rows),
        pd.concat(window_parts, ignore_index=True), pd.concat(event_parts, ignore_index=True),
        perf_counter() - start,
    )


def save_associated_dataset(
    dataset: AssociatedDataset, output: str | Path,
    configuration: TopologyConfig, split_path: str | Path,
) -> dict[str, object]:
    # Exportar en un directorio nuevo, sin sobrescribir ni publicar datos
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    batch = dataset.features
    batch.X.to_parquet(destination / 'X.parquet', index=False)
    dataset.y.to_frame().to_parquet(destination / 'y.parquet', index=False)
    batch.groups.to_frame().to_parquet(destination / 'groups.parquet', index=False)
    batch.identifiers.to_parquet(destination / 'identifiers.parquet', index=False)
    dataset.incidents.to_csv(destination / 'incident_audit.csv', index=False)
    dataset.windows.to_csv(destination / 'window_audit.csv', index=False)
    dataset.events.to_csv(destination / 'event_audit.csv', index=False)
    manifest: dict[str, object] = {
        'schema_version': 1, 'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'dataset': configuration.dataset, 'partition': 'development',
        'kind': 'exploratory_supervised_dataset',
        'association_policy': 'first_complete_diagnosis_with_suspicion_and_window_start_at_or_after_injection',
        'injection_usage': 'offline_association_and_labels_only',
        'attempted_incidents': len(dataset.incidents),
        'associated_incidents': int(dataset.incidents.status.eq('associated').sum()),
        'incident_status_counts': dataset.incidents.status.value_counts().to_dict(),
        'false_alerts_before_injection': int(dataset.incidents.false_alerts_before_injection.sum()),
        'window_status_counts': dataset.windows.status.value_counts().to_dict(),
        'samples': len(batch.X), 'features': len(batch.X.columns),
        'feature_names': batch.X.columns.tolist(),
        'positive_samples': int(dataset.y.sum()),
        'negative_samples': int(dataset.y.eq(0).sum()),
        'missing_feature_values': int(batch.X.isna().sum().sum()),
        'elapsed_sec': dataset.elapsed_sec,
        'notes': [
            'Todos los servicios observados se conservan; sin submuestreo ni imputación.',
            'La asociación usa verdad de referencia offline; no es una regla de producción.',
            'Las omisiones y falsas alertas permanecen en las auditorías, fuera de X.',
            'No presentar ranking condicionado a la asociación como rendimiento integral del sistema.',
            'Antes de GroupKFold independiente, recalibrar detector solo con train y regenerar ventanas/features.',
            'No se cargaron métricas de test ni se modificó la partición.',
        ],
    }
    snapshot = json.loads(configuration.path.read_text())
    snapshot['data_config']['root'] = str(configuration.data_root)
    calibration = configuration.detector_calibration
    if calibration is not None:
        snapshot['detector_calibration_file'] = 'detector_calibration.json'
        (destination / 'detector_calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    (destination / 'topology.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (destination / 'incident_split.json').write_text(Path(split_path).read_text())
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest
