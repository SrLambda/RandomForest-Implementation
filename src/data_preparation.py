# Imputación aprendida en train y submuestreo exclusivo de sus negativos

from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PreparedSamples:
    X: pd.DataFrame
    y: pd.Series
    identifiers: pd.DataFrame

    @property
    def groups(self) -> pd.Series:
        return self.identifiers.incident_id.rename('incident_id').copy()


def _validate_X(X: pd.DataFrame) -> None:
    if X.empty or not X.columns.is_unique or not X.index.is_unique:
        raise ValueError('X requiere filas, columnas e índices únicos')
    if any(not isinstance(name, str) for name in X.columns):
        raise ValueError('Los nombres de features deben ser textos')
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
           or pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
        raise ValueError('X debe contener solo features numéricas reales')
    if np.isinf(X.to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError('X contiene infinitos')


class TrainMedianImputer:
    # Medianas del train completo, variables sin datos usan constante 0.

    def __init__(self) -> None:
        self._statistics: pd.Series | None = None
        self._empty_columns: list[str] = []

    def fit(self, X: pd.DataFrame) -> 'TrainMedianImputer':
        _validate_X(X)
        medians = X.median()
        self._empty_columns = medians.index[medians.isna()].tolist()
        self._statistics = medians.fillna(0.).astype(float)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        _validate_X(X)
        if self._statistics is None:
            raise ValueError('Ajustar el imputador con train antes de transformar')
        if X.columns.tolist() != self._statistics.index.tolist():
            raise ValueError('El esquema de features difiere del train del imputador')
        return X.astype(float).fillna(self._statistics)

    def state(self) -> dict[str, Any]:
        """Estado serializable para reproducir la transformación en inferencia."""
        if self._statistics is None:
            raise ValueError('El imputador aún no está ajustado')
        return {'schema_version': 1, 'strategy': 'train_median', 'empty_fill_value': 0.,
                'feature_names': self._statistics.index.tolist(),
                'statistics': self._statistics.to_dict(),
                'all_missing_train_columns': list(self._empty_columns)}

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> 'TrainMedianImputer':
        # Restaurar medianas sin consultar nuevos datos
        if state.get('schema_version') != 1 or state.get('strategy') != 'train_median':
            raise ValueError('Estado de imputación no soportado')
        names = state['feature_names']
        statistics = state['statistics']
        if (not names or len(set(names)) != len(names) or set(names) != set(statistics)
                or not set(state['all_missing_train_columns']) <= set(names)
                or state.get('empty_fill_value') != 0.):
            raise ValueError('Estado de imputación incompatible')
        values = pd.Series([statistics[key] for key in names], index=names, dtype=float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError('Las medianas deben ser finitas')
        result = cls()
        result._statistics = values
        result._empty_columns = list(state['all_missing_train_columns'])
        return result


def _validate_sampling(negative_ratio: float, seed: int) -> None:
    if isinstance(negative_ratio, bool) or not math.isfinite(negative_ratio) or negative_ratio <= 0:
        raise ValueError('negative_ratio debe ser finito y positivo')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed debe ser un entero no negativo')


def undersample_training_indices(
    y: pd.Series, *, negative_ratio: float = 1., seed: int = 42,
) -> np.ndarray:
    # Posiciones de train: todos los positivos y negativos elegidos al azar.

    _validate_sampling(negative_ratio, seed)
    if y.isna().any() or not y.isin([0, 1]).all() or y.nunique() != 2:
        raise ValueError('Train requiere ambas clases binarias sin faltantes')
    positives = np.flatnonzero(y.to_numpy() == 1)
    negatives = np.flatnonzero(y.to_numpy() == 0)
    # Acotar antes de multiplicar evita desbordamientos con razones muy grandes.
    count = math.ceil(len(positives) * min(negative_ratio, len(negatives) / len(positives)))
    count = min(count, len(negatives))
    selected = np.random.default_rng(seed).choice(negatives, size=count, replace=False)
    return np.sort(np.concatenate([positives, selected]))


def _load_samples(path: Path) -> PreparedSamples:
    X = pd.read_parquet(path / 'X.parquet')
    y = pd.read_parquet(path / 'y.parquet')['y']
    ids = pd.read_parquet(path / 'identifiers.parquet')
    groups = pd.read_parquet(path / 'groups.parquet')['incident_id']
    _validate_X(X)
    if (not X.index.equals(y.index) or not X.index.equals(ids.index)
            or not groups.equals(ids.incident_id) or ids.empty
            or ids[['incident_id', 'service']].isna().any().any()
            or ids.duplicated(['incident_id', 'service']).any()):
        raise ValueError(f'Filas o grupos incompatibles en {path}')
    if y.isna().any() or not y.isin([0, 1]).all() or not y.groupby(ids.incident_id).sum().eq(1).all():
        raise ValueError(f'Etiquetas inválidas en {path}')
    return PreparedSamples(X, y, ids)


def _save_samples(samples: PreparedSamples, destination: Path, positions: np.ndarray) -> None:
    destination.mkdir()
    samples.X.iloc[positions].reset_index(drop=True).to_parquet(destination / 'X.parquet', index=False)
    samples.y.iloc[positions].reset_index(drop=True).to_frame('y').to_parquet(destination / 'y.parquet', index=False)
    ids = samples.identifiers.iloc[positions].reset_index(drop=True)
    ids.to_parquet(destination / 'identifiers.parquet', index=False)
    ids.incident_id.to_frame().to_parquet(destination / 'groups.parquet', index=False)
    pd.DataFrame({'source_row': positions}).to_parquet(destination / 'source_rows.parquet', index=False)


def prepare_training_folds(
    source: str | Path, output: str | Path, *, negative_ratio: float = 1., seed: int = 42,
) -> dict[str, Any]:
    # Consumir folds calibrados; exportar train balanceado y validación completa.

    started = perf_counter()
    # Validar parámetros antes de crear la salida o leer los datasets completos.
    _validate_sampling(negative_ratio, seed)
    source, destination = Path(source).resolve(), Path(output).resolve()
    folds_doc = json.loads((source / 'folds.json').read_text())
    split = json.loads((source / 'incident_split.json').read_text())
    summary = json.loads((source / 'summary.json').read_text())
    if summary.get('status') != 'complete' or folds_doc.get('metadata_sha256') != split.get('metadata_sha256'):
        raise ValueError('La preparación de folds de origen está incompleta o es incompatible')
    folds = folds_doc['folds']
    validation_ids = [key for fold in folds for key in fold['validation']]
    if (len(validation_ids) != len(set(validation_ids)) or set(validation_ids) != set(split['development'])
            or set(validation_ids) & set(split['test'])):
        raise ValueError('Los folds no cubren desarrollo exactamente una vez en validación')
    destination.mkdir(parents=True, exist_ok=False)
    results = []
    for fold in folds:
        train_ids, val_ids = set(fold['train']), set(fold['validation'])
        if train_ids & val_ids or train_ids | val_ids != set(split['development']):
            raise ValueError('Train y validación deben ser disjuntos y cubrir desarrollo')
        folder = source / f"fold_{fold['fold']}"
        calibration = json.loads((folder / 'detector_calibration.json').read_text())
        if (set(calibration['fit_incidents']) != train_ids
                or set(calibration['validation_incidents']) != val_ids
                or calibration.get('metadata_sha256') != split['metadata_sha256']):
            raise ValueError('La calibración no corresponde al train del fold')
        train, validation = _load_samples(folder / 'train'), _load_samples(folder / 'validation')
        for role, samples, allowed in [('train', train, train_ids), ('validation', validation, val_ids)]:
            manifest = json.loads((folder / role / 'manifest.json').read_text())
            audit = pd.read_csv(folder / role / 'incident_audit.csv')
            if (manifest.get('role') != role or manifest.get('fold') != fold['fold']
                    or set(audit.incident_id) != allowed or len(audit) != len(allowed)
                    or set(samples.groups) != set(audit.loc[audit.status.eq('associated'), 'incident_id'])):
                raise ValueError('IDs de muestras, auditoría o rol incompatibles con el fold')
        imputer = TrainMedianImputer().fit(train.X)
        train_X, val_X = imputer.transform(train.X), imputer.transform(validation.X)
        selected = undersample_training_indices(train.y, negative_ratio=negative_ratio, seed=seed)
        target = destination / folder.name
        target.mkdir()
        _save_samples(PreparedSamples(train_X, train.y, train.identifiers), target / 'train', selected)
        _save_samples(PreparedSamples(val_X, validation.y, validation.identifiers), target / 'validation', np.arange(len(val_X)))
        (target / 'imputer.json').write_text(json.dumps(imputer.state(), indent=2) + '\n')
        report = {
            'fold': fold['fold'], 'source': str(folder), 'seed': seed, 'negative_ratio': negative_ratio,
            'train_incidents': fold['train'], 'validation_incidents': fold['validation'],
            'train_rows_before': len(train.X), 'train_rows_after': len(selected),
            'train_positive_rows': int(train.y.iloc[selected].sum()),
            'train_negative_rows': int(train.y.iloc[selected].eq(0).sum()),
            'validation_rows': len(validation.X), 'validation_positive_rows': int(validation.y.sum()),
            'all_missing_train_columns': imputer.state()['all_missing_train_columns'],
            'feature_names': train.X.columns.tolist(),
            'imputer_fitted_on': 'all_associated_train_rows_before_undersampling',
            'audit_source': str(folder),
        }
        (target / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        results.append(report)
        print(f"Fold {fold['fold']}: train {len(train.X)} → {len(selected)}; validación completa {len(val_X)}", flush=True)
    for filename in ('folds.json', 'incident_split.json'):
        (destination / filename).write_text((source / filename).read_text())
    result = {'schema_version': 1, 'stage': 'train_data_preparation', 'status': 'complete',
              'source': str(source), 'folds': results, 'elapsed_sec': perf_counter() - started,
              'notes': ['Validación sin submuestreo; IDs y etiquetas fuera de X.',
                        'Sin entrenamiento. Puntuaciones futuras no necesariamente calibradas.',
                        'CV exploratoria por selección previa de reglas del detector en desarrollo.']}
    (destination / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
