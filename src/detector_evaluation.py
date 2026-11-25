# Evaluación descriptiva del detector exclusivamente en desarrollo.

from collections import Counter
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from .configuration import TopologyConfig
from .evaluation import IncidentSplit
from .ingestion import IncidentMetadata, RCAEvalLoader
from .window_detector import DetectorState, WindowDetector


def missed_metric_diagnostics(
    configuration: TopologyConfig, metadata: IncidentMetadata, frame: pd.DataFrame,
) -> list[dict[str, object]]:
    """Describir las métricas de una omisión; no decidir alertas con etiquetas.

    Los rangos antes/después de inyección son información de evaluación. Un
    solapamiento no demuestra imposibilidad de detectar, solo limita la evidencia
    para atribuir el fallo usando cambios grandes en estas métricas.
    """
    params = configuration.detector_config
    calibration = configuration.detector_calibration or {}
    floors = calibration.get('minimum_deviation', {})
    scale_floors = calibration.get('minimum_reference_scale', {})
    initial = frame.iloc[:int(params['warmup_sec'])]
    before = frame.loc[frame.time < metadata.injection_timestamp]
    after = frame.loc[frame.time >= metadata.injection_timestamp]
    service = next(item for item in configuration.services if item['name'] == metadata.root_cause_service)
    records: list[dict[str, object]] = []
    for metric in service['required_metrics'] + service['optional_metrics']:
        column = f"{service['name']}_{metric}"
        if column not in frame:
            continue
        family = configuration.metric_families[metric]
        mean = float(initial[column].mean())
        std = float(initial[column].std(ddof=0))
        scale = max(abs(mean), std)
        records.append({
            'incident_id': metadata.incident_id, 'metric': column, 'family': family,
            'warmup_mean': mean, 'warmup_std': std,
            'warmup_cv': None if mean == 0 else std / abs(mean),
            'before_injection_min': before[column].min(),
            'before_injection_max': before[column].max(),
            'after_injection_min': after[column].min(),
            'after_injection_max': after[column].max(),
            'missing_before': int(before[column].isna().sum()),
            'missing_after': int(after[column].isna().sum()),
            'minimum_deviation': floors.get(column, 0.) * params.get('minimum_deviation_multiplier_by_family', {}).get(family, 1.),
            'relative_reference_eligible': bool(
                params.get('relative_increase_factor_by_family', {}).get(family, 0) > 1
                and initial[column].count() >= 2 and scale > 0
                and scale >= scale_floors.get(column, 0.)
                and std <= params.get('relative_reference_max_cv', .25) * abs(mean)
            ),
        })
    return records


def evaluate_development(
    configuration: TopologyConfig,
    split_path: str | Path,
    output: str | Path,
) -> dict[str, object]:
    # Evaluar  72 casos sin leer métricas de test; persistir CSV y resumen.

    loader = RCAEvalLoader(configuration=configuration)
    metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
    split = IncidentSplit.load(split_path, metadata)
    configuration.validate_calibration_split(split.development, split.test, split.metadata_sha256)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    triggers: Counter[str] = Counter()
    miss_diagnostics: list[dict[str, object]] = []
    for number, incident_id in enumerate(split.development, 1):
        meta = loader.load_metadata(incident_id)
        observations = loader.load_metrics(incident_id)
        injection = meta.injection_timestamp
        false_alerts = crossings = post_alerts = pre_seconds = pre_alert_seconds = 0
        unknown = recovery_events = 0
        active_at_injection = False
        first_post: int | None = None
        case_start = perf_counter()
        for update in WindowDetector(configuration).run(observations):
            if update.timestamp < injection and update.state != DetectorState.WARMUP:
                # WARMUP_COMPLETE expresa NORMAL en la última muestra de warmup;
                # aún no es una observación clasificada.
                if not any(event.kind == 'WARMUP_COMPLETE' for event in update.events):
                    pre_seconds += 1
                    pre_alert_seconds += update.state == DetectorState.ALERT
            if update.timestamp == injection:
                active_at_injection = update.state == DetectorState.ALERT
            unknown += update.status.value == 'UNKNOWN'
            for event in update.events:
                if event.kind in ('WARMUP_COMPLETE', 'SEGMENT_CLOSED', 'SEGMENT_INCOMPLETE'):
                    continue
                event_rows.append({
                    'incident_id': incident_id, 'event': event.kind,
                    'timestamp': event.timestamp, 'injection_timestamp': injection,
                    'offset_sec': event.timestamp - injection, 'alert_id': event.alert_id,
                    'start_timestamp': event.start_timestamp, 'recovered': event.recovered,
                    'trigger_metrics': json.dumps(update.anomalous_metrics),
                })
                if event.kind == 'ALERT_CONFIRMED':
                    if event.timestamp < injection:
                        false_alerts += 1
                        triggers.update(update.anomalous_metrics)
                    elif event.start_timestamp is not None and event.start_timestamp < injection:
                        crossings += 1
                    else:
                        post_alerts += 1
                        if first_post is None:
                            first_post = event.timestamp
                recovery_events += event.kind == 'RECOVERED'
        rows.append({
            'incident_id': incident_id, 'service': meta.root_cause_service,
            'fault': meta.fault_type, 'false_alerts': false_alerts,
            'crossing_alerts': crossings, 'post_injection_alerts': post_alerts,
            'detected': first_post is not None,
            'detection_delay_sec': None if first_post is None else first_post - injection,
            'alert_at_injection': active_at_injection,
            'pre_injection_seconds': pre_seconds, 'pre_injection_alert_seconds': pre_alert_seconds,
            'unknown_samples': unknown, 'recovery_events': recovery_events,
            'final_state': update.state.value,
            'processing_sec': perf_counter() - case_start,
        })
        if first_post is None:
            miss_diagnostics.extend(missed_metric_diagnostics(configuration, meta, observations.frame))
        if number % 12 == 0:
            print(f'Desarrollo: {number}/{len(split.development)} casos', flush=True)
    cases = pd.DataFrame(rows)
    calibration = configuration.detector_calibration
    if calibration:
        internal_validation = set(calibration['validation_incidents'])
        cases['calibration_subset'] = cases.incident_id.map(
            lambda key: 'validation' if key in internal_validation else 'fit'
        )
    delays = cases['detection_delay_sec'].dropna()
    monitored = int(cases['pre_injection_seconds'].sum())
    summary: dict[str, object] = {
        'partition': 'development', 'cases': len(cases),
        'false_alerts': int(cases.false_alerts.sum()),
        'cases_with_false_alerts': int(cases.false_alerts.gt(0).sum()),
        'false_alerts_per_normal_hour': float(cases.false_alerts.sum() * 3600 / monitored),
        'pre_injection_alert_fraction': float(cases.pre_injection_alert_seconds.sum() / monitored),
        'cases_alert_at_injection': int(cases.alert_at_injection.sum()),
        'cases_with_new_post_injection_detection': int(cases.detected.sum()),
        'cases_without_new_post_injection_detection': int((~cases.detected).sum()),
        'median_detection_delay_sec': None if delays.empty else float(delays.median()),
        'p90_detection_delay_sec': None if delays.empty else float(delays.quantile(.9)),
        'cases_ending_alert': int(cases.final_state.eq('ALERT').sum()),
        'false_alert_trigger_metrics': dict(triggers.most_common()),
        'elapsed_sec': perf_counter() - started,
        'detector_config': configuration.detector_config,
        'notes': [
            'Desarrollo usado para calibración: no es una estimación independiente.',
            'Alertas preinyección se consideran falsas respecto a la inyección etiquetada.',
            'Una alerta previa que cruza la inyección no cuenta como nueva detección.',
            'Retrasos calculados solo sobre los casos detectados; reportar también omisiones.',
            'No existe etiqueta de fin de fallo para validar recuperación real.',
        ],
    }
    if calibration:
        subset_summary: dict[str, dict[str, object]] = {}
        for name, subset in cases.groupby('calibration_subset'):
            subset_delays = subset['detection_delay_sec'].dropna()
            subset_summary[name] = {
                'cases': len(subset),
                'cases_with_false_alerts': int(subset.false_alerts.gt(0).sum()),
                'false_alerts': int(subset.false_alerts.sum()),
                'detected': int(subset.detected.sum()),
                'median_delay_sec': None if subset_delays.empty else float(subset_delays.median()),
            }
        summary['internal_subsets'] = subset_summary
    cases.to_csv(destination / 'cases.csv', index=False)
    pd.DataFrame(event_rows).to_csv(destination / 'events.csv', index=False)
    pd.DataFrame(miss_diagnostics).to_csv(destination / 'miss_diagnostics.csv', index=False)
    by_fault = cases.groupby('fault').agg(
        cases=('incident_id', 'size'), detected=('detected', 'sum'),
        false_alerts=('false_alerts', 'sum'),
        median_delay_sec=('detection_delay_sec', 'median'),
    )
    by_fault.to_csv(destination / 'by_fault.csv')
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Snapshot completo para reproducir cada ejecución aunque cambie el JSON principal.
    snapshot = json.loads(configuration.path.read_text())
    snapshot['data_config']['root'] = str(configuration.data_root)
    if calibration:
        snapshot['detector_calibration_file'] = 'calibration_snapshot.json'
    (destination / 'topology_snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (destination / 'split_snapshot.json').write_text(Path(split_path).read_text())
    if calibration:
        (destination / 'calibration_snapshot.json').write_text(json.dumps(calibration, indent=2) + '\n')
    return summary
