# Contratos y carga local de métricas y metadatos RCAEval en Parquet

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .configuration import TopologyConfig


@dataclass(frozen=True, slots=True)
class IncidentMetadata:
    # Verdad de referencia de un caso, separada de las entradas del modelo
    # incident_id  corresponde a case en cases.parquet
    # injection_timestamp corresponde a inject_time (Unix, segundos)
    # El instante de inyección sirve para evaluación, no para activar el detector.

    incident_id: str
    dataset: str
    root_cause_service: str
    fault_type: str
    injection_timestamp: int

    def __post_init__(self) -> None:
        for name in ("incident_id", "dataset", "root_cause_service", "fault_type"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} debe ser un texto no vacío")
        if isinstance(self.injection_timestamp, bool) or not isinstance(
            self.injection_timestamp, (int, np.integer)
        ):
            raise ValueError("injection_timestamp debe ser un entero Unix en segundos")


class MetricTimeSeries:
    # Métricas de un caso en formato ancho.
    # Contrato: time contiene timestamps Unix enteros en segundos, únicos, crecientes y separados por un segundo. 
    # Las otras columnas son solo métricas numéricas
    # Se copia el DataFrame al recibirlo y al exponerlo para evitar que cambios externos invaliden el contrato. 
    

    def __init__(self, incident_id: str, frame: pd.DataFrame) -> None:
        if not isinstance(incident_id, str) or not incident_id.strip():
            raise ValueError("incident_id debe ser un texto no vacío")
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("Las métricas deben tener filas y columnas únicas")
        if not all(isinstance(column, str) for column in frame.columns):
            raise ValueError("Los nombres de columnas deben ser textos")
        if "time" not in frame or len(frame.columns) < 2:
            raise ValueError("Se requiere time y al menos una métrica")
        if any(
            not is_numeric_dtype(dtype) or is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
            for dtype in frame.dtypes
        ):
            raise ValueError("Todas las columnas deben ser numéricas reales")
        values = frame.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ValueError("Las métricas no pueden contener infinitos")
        timestamps = frame["time"].to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(timestamps).all() or (timestamps % 1 != 0).any():
            raise ValueError("time debe contener segundos Unix enteros sin faltantes")
        if not np.all(np.diff(timestamps) == 1):
            raise ValueError("time debe ser creciente, único y tener intervalos de 1 s")
        self._incident_id = incident_id
        self._frame = frame.copy(deep=True)

    @property
    def incident_id(self) -> str:
        # Identificador para asociación con metadatos
        return self._incident_id

    @property
    def frame(self) -> pd.DataFrame:
        # Copia de las observaciones
        return self._frame.copy(deep=True)


class RCAEvalLoader:
    # Acceso a los casos de RE2-SS.

    DATASET = "RE2-SS"
    INDEX_COLUMNS = (
        "case", "dataset", "root_cause_service", "fault", "inject_time"
    )

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        configuration: TopologyConfig | None = None,
    ) -> None:
        self._configuration = configuration
        default_root = configuration.data_root if configuration else Path("data/raw/RCAEval")
        self._root = Path(root if root is not None else default_root).expanduser().resolve()
        data_config = configuration.data_config if configuration else {}
        self._metrics_file = data_config.get("metrics_file", "metrics.parquet")
        index_path = self._root / data_config.get("index_file", "cases.parquet")
        if not index_path.is_file():
            raise FileNotFoundError(f"No se encontró el índice: {index_path}")
        index = pd.read_parquet(index_path)
        missing = set(self.INDEX_COLUMNS) - set(index.columns)
        if missing:
            raise ValueError(f"Faltan columnas en cases.parquet: {sorted(missing)}")
        selected = index.loc[index["dataset"].eq(self.DATASET), list(self.INDEX_COLUMNS)]
        if selected.empty:
            raise ValueError(f"El índice no contiene casos de {self.DATASET}")
        if selected.isna().any().any():
            raise ValueError("Los metadatos RE2-SS contienen campos obligatorios vacíos")
        if selected["case"].duplicated().any():
            raise ValueError("El índice contiene identificadores de incidente duplicados")

        self._metadata: dict[str, IncidentMetadata] = {}
        for row in selected.itertuples(index=False):
            metadata = IncidentMetadata(
                incident_id=row.case,
                dataset=row.dataset,
                root_cause_service=row.root_cause_service,
                fault_type=row.fault,
                injection_timestamp=row.inject_time,
            )
            if configuration and metadata.root_cause_service not in configuration.service_names:
                raise ValueError(f"Causa raíz no configurada: {metadata.root_cause_service}")
            # Los identificadores deben ser nombres de carpetas, no rutas.
            if (
                Path(metadata.incident_id).name != metadata.incident_id
                or metadata.incident_id in {".", ".."}
                or "\\" in metadata.incident_id
            ):
                raise ValueError(f"Identificador de incidente inválido: {metadata.incident_id!r}")
            self._metadata[metadata.incident_id] = metadata

    def list_incidents(self) -> tuple[str, ...]:
        # Identificadores ordenados de RE2-SS, sin filtros
        return tuple(sorted(self._metadata))

    def load_metadata(self, incident_id: str) -> IncidentMetadata:
        # Obtener la referencia y contrastar la inyección

        metadata = self._get_metadata(incident_id)
        injection_path = self._case_directory(incident_id) / "inject_time.txt"
        if injection_path.is_file():
            try:
                injection = int(injection_path.read_text(encoding="utf-8").strip())
            except ValueError as exc:
                raise ValueError(f"Timestamp de inyección inválido: {injection_path}") from exc
            if injection != metadata.injection_timestamp:
                raise ValueError(f"Inyección distinta entre índice y archivo: {incident_id}")
        return metadata

    def load_metrics(self, incident_id: str) -> MetricTimeSeries:
        # Leer métricas completas, preservando faltantes, sin añadir etiquetas

        self._get_metadata(incident_id)
        metrics_path = self._case_directory(incident_id) / self._metrics_file
        if not metrics_path.is_file():
            raise FileNotFoundError(f"No se encontraron métricas: {metrics_path}")
        frame = pd.read_parquet(metrics_path)
        try:
            if self._configuration:
                self._configuration.validate_columns(frame.columns.tolist())
            return MetricTimeSeries(incident_id, frame)
        except ValueError as exc:
            raise ValueError(f"Métricas inválidas para {incident_id}: {exc}") from exc

    def _get_metadata(self, incident_id: str) -> IncidentMetadata:
        try:
            return self._metadata[incident_id]
        except KeyError:
            raise KeyError(f"Incidente no registrado en {self.DATASET}: {incident_id!r}") from None

    def _case_directory(self, incident_id: str) -> Path:
        # Mantener los archivos de cada caso dentro de la carpeta raíz del dataset, evitando escapes de ruta
        directory = (self._root / incident_id).resolve()
        if not directory.is_relative_to(self._root):
            raise ValueError(f"La carpeta del incidente está fuera del dataset: {incident_id}")
        return directory
