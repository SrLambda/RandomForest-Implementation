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
        self._calibration: dict[str, Any] | None = None
        calibration_file = document.get("detector_calibration_file")
        if calibration_file is not None:
            if not isinstance(calibration_file, str) or not calibration_file.strip():
                raise ValueError("detector_calibration_file debe ser una ruta no vacía")
            calibration_path = Path(calibration_file)
            if not calibration_path.is_absolute():
                calibration_path = self.path.parent / calibration_path
            with calibration_path.open(encoding="utf-8") as stream:
                calibration = json.load(stream)
            if not isinstance(calibration, dict) or calibration.get("schema_version") != 1 or calibration.get("dataset") != "RE2-SS":
                raise ValueError("Calibración del detector no soportada")
            if calibration.get("method") != "normal_deviation_from_warmup_mean":
                raise ValueError("Método de calibración no soportado")
            floors = calibration.get("minimum_deviation")
            if not isinstance(floors, dict) or not floors:
                raise ValueError("La calibración no contiene tolerancias por métrica")
            reference_scales = calibration.get("minimum_reference_scale", {})
            if not isinstance(reference_scales, dict):
                raise ValueError("minimum_reference_scale debe ser un mapa de métricas")
            if (set(floors) | set(reference_scales)) - self.metric_columns():
                raise ValueError("La calibración contiene métricas no configuradas")
            for value in (*floors.values(), *reference_scales.values()):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Las tolerancias deben ser finitas y no negativas")
            for key in ("fit_incidents", "validation_incidents"):
                self._names(calibration.get(key), key)
            if calibration.get("warmup_sec") != document["detector_config"]["warmup_sec"]:
                raise ValueError("El calentamiento no coincide con el usado al calibrar")
            self._calibration = calibration

    @property
    def detector_calibration(self) -> dict[str, Any] | None:
        return deepcopy(self._calibration)

    def validate_calibration_split(
        self, development: tuple[str, ...], test: tuple[str, ...], metadata_sha256: str,
    ) -> None:
        """Evitar activar tolerancias ajustadas con incidentes del test reservado."""
        if self._calibration is None:
            return
        fitted = set(self._calibration["fit_incidents"])
        validation = set(self._calibration["validation_incidents"])
        if (
            len(fitted) != 48 or len(validation) != 24 or fitted & validation
            or fitted | validation != set(development)
            or fitted & set(test) or validation & set(test)
            or self._calibration.get("metadata_sha256") != metadata_sha256
        ):
            raise ValueError("La calibración no corresponde a desarrollo 48/24 del manifiesto")

    def validate_fold_calibration(
        self, train: tuple[str, ...], validation: tuple[str, ...],
        development: tuple[str, ...], metadata_sha256: str,
    ) -> None:
        """Exigir tolerancias ajustadas exactamente con train de este fold."""
        if (not train or not validation or len(set(train)) != len(train)
                or len(set(validation)) != len(validation) or set(train) & set(validation)
                or set(train) | set(validation) != set(development)):
            raise ValueError('Fold incompatible con la partición de desarrollo')
        calibration = self._calibration
        if (calibration is None or set(calibration['fit_incidents']) != set(train)
                or set(calibration['validation_incidents']) != set(validation)
                or calibration.get('metadata_sha256') != metadata_sha256):
            raise ValueError('La calibración debe ajustarse solo con el train del fold')

    @property
    def dataset(self) -> str:
        return self._document["dataset"]

    @property
    def data_config(self) -> dict[str, Any]:
        return deepcopy(self._document["data_config"])

    @property
    def detector_config(self) -> dict[str, Any]:
        return deepcopy(self._document["detector_config"])

    @property
    def metric_families(self) -> dict[str, str]:
        return dict(self._document["metric_families"])

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
            "warmup_sec", "anomaly_consecutive_sec", "diagnosis_delay_sec",
        ):
            value = detector.get(field)
            if (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{field} debe ser un número finito positivo")
        if detector["sampling_interval_sec"] != 1:
            raise ValueError("El contrato de ingesta actual requiere resolución de 1 s")
        for field in (
            "baseline_window_sec", "recovery_consecutive_sec", "max_anomaly_window_sec",
            "warmup_sec", "anomaly_consecutive_sec", "diagnosis_delay_sec",
        ):
            if detector[field] % 1:
                raise ValueError(f"{field} debe representar segundos enteros")
        if detector["baseline_window_sec"] < 2 or detector["warmup_sec"] < 2:
            raise ValueError("El baseline y el calentamiento requieren al menos 2 segundos")
        families = doc.get("metric_families")
        if not isinstance(families, dict) or not families or any(
            not isinstance(k, str) or not k.strip()
            or not isinstance(v, str) or not v.strip() for k, v in families.items()
        ):
            raise ValueError("metric_families debe asociar métricas a familias no vacías")
        directions = detector.get("anomaly_direction_by_family", {})
        multipliers = detector.get("minimum_deviation_multiplier_by_family", {})
        decrease_multipliers = detector.get("decrease_deviation_multiplier_by_family", {})
        fixed_families = cls._names(detector.get("fixed_reference_families", []), "fixed_reference_families")
        if set(fixed_families) - set(families.values()):
            raise ValueError("fixed_reference_families contiene familias desconocidas")
        relative_factors = detector.get("relative_increase_factor_by_family", {})
        if not isinstance(relative_factors, dict) or set(relative_factors) - set(families.values()):
            raise ValueError("relative_increase_factor_by_family debe referenciar familias configuradas")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 1 for value in relative_factors.values()
        ):
            raise ValueError("Los factores de aumento relativo deben ser finitos y > 1")
        max_cv = detector.get("relative_reference_max_cv", .25)
        if isinstance(max_cv, bool) or not isinstance(max_cv, (int, float)) or not math.isfinite(max_cv) or max_cv <= 0:
            raise ValueError("relative_reference_max_cv debe ser finito y positivo")
        for name, mapping in (("anomaly_direction_by_family", directions),
                              ("minimum_deviation_multiplier_by_family", multipliers),
                              ("decrease_deviation_multiplier_by_family", decrease_multipliers)):
            if not isinstance(mapping, dict) or set(mapping) - set(families.values()):
                raise ValueError(f"{name} debe referenciar familias configuradas")
        if any(value not in ("both", "increase", "decrease") for value in directions.values()):
            raise ValueError("Dirección de anomalía no soportada")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 1
            for value in (*multipliers.values(), *decrease_multipliers.values())
        ):
            raise ValueError("Los multiplicadores de tolerancia deben ser finitos y >= 1")
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
