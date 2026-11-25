"""Features causales por servicio a partir de un evento de diagnóstico.

No ajusta imputadores ni consulta metadatos para extraer variables. Las etiquetas
se generan mediante una función separada y nunca forman parte de X.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .configuration import TopologyConfig
from .ingestion import IncidentMetadata
from .window_detector import DetectorEvent

FEATURE_SCHEMA_VERSION = 1
_LOCAL_STATS = (
    'present', 'baseline_count', 'window_count',
    'baseline_mean', 'baseline_std', 'window_mean', 'window_std',
    'window_min', 'window_max', 'mean_delta', 'normalized_mean_shift',
    'peak_abs_normalized_shift', 'outside_3sigma_fraction',
    'first_deviation_sec', 'baseline_missing_fraction', 'window_missing_fraction',
)


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Un diagnóstico: una fila por servicio observado, con IDs fuera de X.

    Los DataFrames son resultados de trabajo, no objetos inmutables. Mantener
    su alineación al concatenar casos para entrenamiento o evaluación.
    """

    X: pd.DataFrame
    identifiers: pd.DataFrame

    @property
    def groups(self) -> pd.Series:
        """Grupos para GroupKFold; distintas ventanas nunca separan un incidente."""
        return self.identifiers['incident_id'].rename('incident_id').copy()


def _finite(series: pd.Series) -> pd.Series:
    return series.loc[np.isfinite(series)]


def _metric_features(
    baseline: pd.DataFrame, window: pd.DataFrame, column: str,
) -> dict[str, float]:
    result = {key: np.nan for key in _LOCAL_STATS}
    present = column in window.columns
    result['present'] = float(present)
    if not present:
        result['baseline_count'] = result['window_count'] = 0.
        result['baseline_missing_fraction'] = result['window_missing_fraction'] = 1.
        return result
    before, current = _finite(baseline[column]), _finite(window[column])
    result.update(
        baseline_count=float(len(before)), window_count=float(len(current)),
        baseline_missing_fraction=1 - len(before) / len(baseline),
        window_missing_fraction=1 - len(current) / len(window),
    )
    if len(before):
        result['baseline_mean'] = float(before.mean())
    if len(before) >= 2:
        result['baseline_std'] = float(before.std(ddof=0))
    if len(current):
        result.update(window_mean=float(current.mean()), window_std=float(current.std(ddof=0)),
                      window_min=float(current.min()), window_max=float(current.max()))
    if len(before) < 2 or not len(current):
        return result
    mean, std = result['baseline_mean'], result['baseline_std']
    result['mean_delta'] = result['window_mean'] - mean
    # Cambio adimensional acotado: evita dividir por cero en métricas constantes.
    denominator = abs(result['window_mean']) + abs(mean) + std
    result['normalized_mean_shift'] = result['mean_delta'] / denominator if denominator else 0.
    denominators = current.abs().to_numpy() + abs(mean) + std
    shifts = np.divide(
        np.abs(current.to_numpy() - mean), denominators,
        out=np.zeros(len(current)), where=denominators > 0,
    )
    result['peak_abs_normalized_shift'] = float(shifts.max())
    different = ~np.isclose(current.to_numpy(), mean, rtol=1e-9, atol=0)
    outside = (np.abs(current.to_numpy() - mean) > 3 * std) & different
    result['outside_3sigma_fraction'] = float(outside.mean())
    if outside.any():
        first_index = current.index[np.flatnonzero(outside)[0]]
        result['first_deviation_sec'] = float(window.loc[first_index, 'time'] - window.time.iloc[0])
    return result


def _peer_mean(values: pd.Series) -> pd.Series:
    """Promedio de otros servicios, ignorando NaN sin convertirlos en salud."""
    count = values.notna().sum() - values.notna().astype(int)
    total = values.sum(min_count=1) - values.fillna(0)
    return total.div(count.where(count > 0))


class FeatureBuilder:
    """Mismo esquema de variables en todos los servicios y en inferencia.

    Se incluyen componentes con alguna columna presente, incluso si sus valores
    son NaN. Un componente enteramente ausente no se inventa como candidato.
    No usa aristas de topología no verificadas, nombres de servicio como features
    ni tolerancias aprendidas globalmente para transformar X.
    """

    def __init__(self, configuration: TopologyConfig) -> None:
        self._configuration = configuration
        self._services = sorted(configuration.services, key=lambda item: item['name'])
        self._metrics = sorted(configuration.metric_families)

    def _validate(self, event: DetectorEvent) -> tuple[pd.DataFrame, pd.DataFrame]:
        if event.kind != 'DIAGNOSIS_DUE' or event.baseline is None or event.observations is None:
            raise ValueError('Se requiere DIAGNOSIS_DUE con baseline y observaciones')
        if not event.incident_id or not event.alert_id:
            raise ValueError('El diagnóstico necesita identificadores de incidente y alerta')
        before, current = event.baseline.copy(), event.observations.copy()
        for frame in (before, current):
            if frame.empty or not frame.columns.is_unique:
                raise ValueError('Ventanas vacías o con columnas duplicadas')
            if any(not isinstance(name, str) for name in frame.columns):
                raise ValueError('Los nombres de columnas deben ser textos')
            self._configuration.validate_columns(frame.columns.tolist())
            if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                   or pd.api.types.is_complex_dtype(dtype) for dtype in frame.dtypes):
                raise ValueError('Las ventanas deben contener columnas numéricas reales')
            if np.isinf(frame.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError('Las ventanas no deben contener infinitos')
            times = frame.time.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(times).all() or (times % 1 != 0).any() or (np.diff(times) <= 0).any():
                raise ValueError('Timestamps enteros, únicos y crecientes requeridos')
        if set(before.columns) != set(current.columns):
            raise ValueError('Baseline y observaciones deben tener las mismas columnas')
        if before.time.max() >= current.time.min() or current.time.max() != event.timestamp:
            raise ValueError('Ventanas solapadas o incompatibles con el instante del diagnóstico')
        if not current.time.diff().dropna().eq(1).all():
            raise ValueError('Las observaciones del diagnóstico deben conservar resolución de 1 s')
        # Los índices de Pandas no son timestamps; normalizarlos conserva los datos.
        return before.reset_index(drop=True), current.reset_index(drop=True)

    def build(self, event: DetectorEvent) -> FeatureBatch:
        before, current = self._validate(event)
        features: list[dict[str, float]] = []
        identifiers: list[dict[str, str | int]] = []
        for service in self._services:
            name = service['name']
            available = service['required_metrics'] + service['optional_metrics']
            if not any(f'{name}_{metric}' in current for metric in available):
                continue
            row: dict[str, float] = {}
            local: list[dict[str, float]] = []
            for metric in self._metrics:
                stats = _metric_features(before, current, f'{name}_{metric}')
                row.update({f'{metric}__{key}': value for key, value in stats.items()})
                local.append(stats)
            severity = [item['peak_abs_normalized_shift'] for item in local
                        if np.isfinite(item['peak_abs_normalized_shift'])]
            onsets = [item['first_deviation_sec'] for item in local if np.isfinite(item['first_deviation_sec'])]
            fractions = [item['outside_3sigma_fraction'] for item in local
                         if np.isfinite(item['outside_3sigma_fraction'])]
            row.update(
                local_severity=max(severity) if severity else np.nan,
                changed_metric_fraction=float(np.mean(np.array(fractions) > 0)) if fractions else np.nan,
                first_deviation_sec=min(onsets) if onsets else np.nan,
                observed_metric_count=float(sum(item['present'] for item in local)),
                window_duration_sec=float(current.time.iloc[-1] - current.time.iloc[0] + 1),
            )
            features.append(row)
            identifiers.append({
                'incident_id': event.incident_id, 'alert_id': event.alert_id, 'service': name,
                'diagnosis_timestamp': event.timestamp,
                'window_start_timestamp': int(current.time.iloc[0]),
            })
        if not features:
            raise ValueError('No hay servicios observados para construir candidatos')
        X = pd.DataFrame(features, dtype=float)
        X['peer_severity_mean'] = _peer_mean(X.local_severity)
        X['severity_excess'] = X.local_severity - X.peer_severity_mean
        X['severity_percentile'] = X.local_severity.rank(method='average', pct=True)
        X['onset_delay_from_first_service_sec'] = X.first_deviation_sec - X.first_deviation_sec.min()
        for metric in self._metrics:
            shift = X[f'{metric}__normalized_mean_shift']
            X[f'{metric}__shift_excess_vs_peers'] = shift - _peer_mean(shift)
            X[f'{metric}__shift_percentile'] = shift.rank(method='average', pct=True)
        if np.isinf(X.to_numpy()).any():
            raise ValueError('Se generaron features infinitas; revisar escala de entrada')
        return FeatureBatch(X, pd.DataFrame(identifiers))


def assign_root_cause_labels(batch: FeatureBatch, metadata: IncidentMetadata) -> pd.Series:
    """Etiquetado offline separado, solo para ventanas posteriores a inyección.

    No asignar una causa raíz futura a falsas alertas anteriores a la inyección.
    La asociación/selección de ventanas para el dataset de entrenamiento sigue
    siendo responsabilidad de la orquestación y debe reportar las omisiones.
    """
    ids = batch.identifiers
    if not ids.index.equals(batch.X.index) or not ids.incident_id.eq(metadata.incident_id).all():
        raise ValueError('Metadatos y features no corresponden al mismo incidente o índice')
    if ids.empty or ids.service.duplicated().any() or ids.alert_id.nunique() != 1:
        raise ValueError('Se requiere un único diagnóstico y una fila por servicio')
    if ids.window_start_timestamp.lt(metadata.injection_timestamp).any():
        raise ValueError('La ventana incluye datos anteriores a inyección; no asignar causa raíz futura')
    if not ids.service.eq(metadata.root_cause_service).any():
        raise ValueError('La causa raíz no está entre los candidatos observados')
    return ids.service.eq(metadata.root_cause_service).astype('int64').rename('y')
