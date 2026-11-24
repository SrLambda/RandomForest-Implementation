# Lectura y validación de la configuración externa del pipeline RCA

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any


class TopologyConfig:
    # Configuración schema_version 3
    #La validación comprueba estructura y compatibilidad, no calibra parámetros.
    
    def __init__(self, path: str | Path = "config/topology.json") -> None:
        self.path = Path(path).expanduser().resolve()
        with self.path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        self._validate(document)
        self._document = document

    @property
    def dataset(self) -> str:
        return self._document["dataset"]

    @property
    def data_config(self) -> dict[str, Any]:
        return deepcopy(self._document["data_config"])

    @property
    def detector_config(self) -> dict[str, float | int]:
        return dict(self._document["detector_config"])

    @property
    def data_root(self) -> Path:
        root = Path(self._document["data_config"]["root"]).expanduser()
        if not root.is_absolute():
            root = self.path.parent.parent / root
        return root.resolve()

    @property
    def service_names(self) -> tuple[str, ...]:
        return tuple(service["name"] for service in self._document["services"])

    @property
    def services(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(self._document["services"]))

    def metric_columns(self, *, required_only: bool = False) -> frozenset[str]:
        # Columnas del inventario, incluidas opcionales
        return frozenset(
            f"{service['name']}_{metric}"
            for service in self._document["services"]
            for metric in (
                service["required_metrics"]
                + ([] if required_only else service["optional_metrics"])
            )
        )

    def validate_columns(self, columns: list[str]) -> None:
        # Rechazar requeridas ausentes y columnas ajenas al inventario

        actual = set(columns)
        missing = (self.metric_columns(required_only=True) | {"time"}) - actual
        unknown = actual - (self.metric_columns() | {"time"})
        if missing or unknown:
            raise ValueError(
                f"Columnas requeridas ausentes: {sorted(missing)}; "
                f"columnas no configuradas: {sorted(unknown)}"
            )

    @staticmethod
    def _names(value: Any, field: str) -> list[str]:
        if not isinstance(value, list) or any(
            not isinstance(name, str) or not name.strip() for name in value
        ):
            raise ValueError(f"{field} debe ser una lista de nombres no vacíos")
        if len(set(value)) != len(value):
            raise ValueError(f"{field} contiene duplicados")
        return value

    @classmethod
    def _validate(cls, doc: Any) -> None:
        if not isinstance(doc, dict) or doc.get("schema_version") != 3:
            raise ValueError("Se requiere un objeto JSON con schema_version 3")
        if doc.get("dataset") != "RE2-SS" or doc.get("system") != "Sock Shop":
            raise ValueError("La ingesta actual soporta RE2-SS / Sock Shop")
        data = doc.get("data_config")
        if not isinstance(data, dict):
            raise ValueError("Falta data_config")
        if not isinstance(data.get("root"), str) or not data["root"].strip():
            raise ValueError("data_config.root debe ser una ruta no vacía")
        for field in ("index_file", "metrics_file"):
            name = data.get(field)
            if (
                not isinstance(name, str) or not name.strip()
                or Path(name).name != name or name in {".", ".."} or "\\" in name
            ):
                raise ValueError(f"{field} debe ser un nombre de archivo")
        supported = {
            "time_column": "time",
            "timestamp_unit": "unix_seconds",
            "metric_column_pattern": "{service}_{metric}",
            "missing_values_policy": "preserve_nan",
            "missing_optional_columns_policy": "preserve_as_unavailable",
            "unit_conversion": "none",
        }
        for field, expected in supported.items():
            if data.get(field) != expected:
                raise ValueError(f"Política no soportada: {field}; se requiere {expected!r}")
        detector = doc.get("detector_config")
        if not isinstance(detector, dict):
            raise ValueError("Falta detector_config")
        for field in (
            "sampling_interval_sec", "baseline_window_sec", "z_score_threshold",
            "recovery_consecutive_sec", "max_anomaly_window_sec",
        ):
            value = detector.get(field)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field} debe ser un número finito positivo")
        if detector["sampling_interval_sec"] != 1:
            raise ValueError("El contrato de ingesta actual requiere resolución de 1 s")
        for field in ("baseline_window_sec", "recovery_consecutive_sec", "max_anomaly_window_sec"):
            if detector[field] % 1:
                raise ValueError(f"{field} debe representar segundos enteros")
        families = doc.get("metric_families")
        if not isinstance(families, dict) or not families or any(
            not isinstance(k, str) or not k.strip()
            or not isinstance(v, str) or not v.strip() for k, v in families.items()
        ):
            raise ValueError("metric_families debe asociar métricas a familias no vacías")
        services = doc.get("services")
        if not isinstance(services, list) or not services or any(
            not isinstance(service, dict) for service in services
        ):
            raise ValueError("services debe ser una lista no vacía de objetos")
        names = cls._names([service.get("name") for service in services], "services.name")
        allow_external = doc.get("allow_external_service_references")
        if not isinstance(allow_external, bool):
            raise ValueError("allow_external_service_references debe ser booleano")
        for service in services:
            name = service["name"]
            required = cls._names(service.get("required_metrics"), f"{name}.required_metrics")
            optional = cls._names(service.get("optional_metrics"), f"{name}.optional_metrics")
            if not required and not optional:
                raise ValueError(f"{name} no tiene métricas configuradas")
            if set(required) & set(optional):
                raise ValueError(f"{name} repite métricas entre requeridas y opcionales")
            if (set(required) | set(optional)) - families.keys():
                raise ValueError(f"{name} utiliza métricas sin familia definida")
            for field in ("upstream", "downstream"):
                if field not in service:
                    raise ValueError(f"Falta {name}.{field}; usar null si no se conoce")
                if service[field] is not None:
                    refs = cls._names(service[field], f"{name}.{field}")
                    if not allow_external and set(refs) - set(names):
                        raise ValueError(f"{name}.{field} referencia servicios desconocidos")
