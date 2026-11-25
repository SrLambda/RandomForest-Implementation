"""Random Forest binario por servicio; sin balanceo, imputación ni etiquetas en X."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import pickle
from time import perf_counter
from typing import Self

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True, slots=True)
class ForestParameters:
    """Configuración base reproducible, sin selección mediante validación."""

    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | None = 'sqrt'
    random_state: int = 42
    n_jobs: int = 1

    def __post_init__(self) -> None:
        for name in ('n_estimators', 'min_samples_leaf'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} debe ser un entero positivo')
        if self.max_depth is not None and (
            isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int) or self.max_depth < 1
        ):
            raise ValueError('max_depth debe ser null o un entero positivo')
        if self.max_features not in ('sqrt', 'log2', None):
            raise ValueError('max_features debe ser sqrt, log2 o null')
        if (isinstance(self.random_state, bool) or not isinstance(self.random_state, int)
                or not 0 <= self.random_state < 2**32):
            raise ValueError('random_state debe ser un entero entre 0 y 2**32-1')
        if isinstance(self.n_jobs, bool) or not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ValueError('n_jobs debe ser un entero distinto de cero')

    @classmethod
    def load(cls, path: str | Path) -> 'ForestParameters':
        document = json.loads(Path(path).read_text())
        expected = set(cls.__dataclass_fields__) | {'schema_version'}
        if not isinstance(document, dict) or set(document) != expected or document['schema_version'] != 1:
            raise ValueError('Configuración Random Forest no soportada o incompleta')
        return cls(**{key: value for key, value in document.items() if key != 'schema_version'})

    def to_dict(self) -> dict[str, object]:
        return {'schema_version': 1, **asdict(self)}


class RootCauseForest:
    """Aprende y puntúa candidatos de causa raíz a partir de features preparadas.

    predict_proba devuelve la puntuación de clase 1 por fila, sin umbral 0.5 ni
    normalización entre servicios. El submuestreo altera la prevalencia: estas
    puntuaciones no deben interpretarse como probabilidades calibradas.
    """

    def __init__(self, parameters: ForestParameters | None = None) -> None:
        self.parameters = parameters or ForestParameters()
        self._forest = RandomForestClassifier(**asdict(self.parameters))
        self._columns: list[str] | None = None
        self.fit_elapsed_sec: float | None = None

    @staticmethod
    def _validate_X(X: pd.DataFrame) -> None:
        if X.empty or not X.columns.is_unique or not X.index.is_unique:
            raise ValueError('X debe tener filas, columnas e índices únicos')
        if any(not isinstance(key, str) for key in X.columns):
            raise ValueError('Los nombres de features deben ser textos')
        forbidden = {'y', 'y_true', 'incident_id', 'service', 'alert_id', 'injection_timestamp',
                     'root_cause_service', 'fault_type', 'diagnosis_timestamp', 'window_start_timestamp'}
        if set(X.columns) & forbidden:
            raise ValueError('X contiene etiquetas o identificadores')
        if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
               or pd.api.types.is_complex_dtype(dtype) for dtype in X.dtypes):
            raise ValueError('Las features deben ser numéricas reales')
        if not np.isfinite(X.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError('Aplicar imputación antes del modelo; X requiere valores finitos')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self._validate_X(X)
        if (not X.index.equals(y.index) or y.isna().any()
                or not y.isin([0, 1]).all() or y.nunique() != 2):
            raise ValueError('Train requiere etiquetas binarias alineadas y ambas clases')
        start = perf_counter()
        # Un nuevo estimador en cada fit evita acumular árboles o estado anterior.
        forest = RandomForestClassifier(**asdict(self.parameters))
        forest.fit(X, y)
        self._forest = forest
        self._columns = X.columns.tolist()
        self.fit_elapsed_sec = perf_counter() - start
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Score de clase positiva, preservando el índice de los candidatos."""
        if self._columns is None:
            raise ValueError('Entrenar el modelo antes de puntuar')
        self._validate_X(X)
        if X.columns.tolist() != self._columns:
            raise ValueError('Las columnas o su orden difieren de train')
        positive = np.flatnonzero(self._forest.classes_ == 1)
        scores = self._forest.predict_proba(X)[:, int(positive[0])]
        return pd.Series(scores, index=X.index, name='root_cause_score')

    def feature_importances(self) -> pd.Series:
        """Importancia por reducción de impureza; no demuestra causalidad."""
        if self._columns is None:
            raise ValueError('Entrenar el modelo antes de consultar importancias')
        return pd.Series(self._forest.feature_importances_, index=self._columns,
                         name='importance').sort_values(ascending=False, kind='stable')

    def save(self, path: str | Path) -> None:
        """Guardar un modelo entrenado en un archivo local nuevo."""
        if self._columns is None:
            raise ValueError('No guardar un modelo sin entrenar')
        with Path(path).open('xb') as stream:
            pickle.dump(self, stream, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Cargar exclusivamente artefactos pickle locales de confianza."""
        with Path(path).open('rb') as stream:
            model = pickle.load(stream)
        if not isinstance(model, cls) or model._columns is None:
            raise ValueError('El artefacto no contiene un RootCauseForest entrenado')
        return model
