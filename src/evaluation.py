"""Reserva persistida por incidente; folds en cross_validation y balanceo pendiente."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from .ingestion import IncidentMetadata


def _metadata_index(metadata: Iterable[IncidentMetadata]) -> dict[str, IncidentMetadata]:
    index: dict[str, IncidentMetadata] = {}
    for item in metadata:
        if item.incident_id in index:
            raise ValueError(f"Incidente duplicado: {item.incident_id}")
        if item.dataset != "RE2-SS":
            raise ValueError("La partición actual requiere RE2-SS")
        index[item.incident_id] = item
    if len(index) != 90:
        raise ValueError(f"Se requieren los 90 incidentes; recibidos: {len(index)}")
    return index


def _fingerprint(index: dict[str, IncidentMetadata]) -> str:
    # Detecta cambios de identidad o etiquetas; no utiliza señales temporales.
    rows = [(key, item.dataset, item.root_cause_service, item.fault_type)
            for key, item in sorted(index.items())]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class IncidentSplit:
    """Identificadores de desarrollo/test; nunca filas ni servicios submuestreados.

    Persistir una vez y reutilizar el manifiesto. No regenerarlo en función de
    resultados del detector o modelo. El fingerprint identifica metadatos, no
    certifica integridad de los archivos de métricas.
    """

    seed: int
    metadata_sha256: str
    development: tuple[str, ...]
    test: tuple[str, ...]

    @classmethod
    def create(cls, metadata: Iterable[IncidentMetadata], seed: int = 42) -> "IncidentSplit":
        """Reservar 18 casos con 3 por fallo y 3 o 4 por servicio causa raíz.

        Se permutan servicios y fallos con semilla fija. Se asignan cíclicamente
        tres servicios distintos a cada fallo y se elige una de las tres
        repeticiones por pareja. Así ninguna pareja queda ausente en desarrollo.
        La selección usa únicamente etiquetas, nunca métricas o calidad del RCA.
        """
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed debe ser un entero")
        index = _metadata_index(metadata)
        pairs: dict[tuple[str, str], list[str]] = defaultdict(list)
        for incident_id, item in sorted(index.items()):
            pairs[item.root_cause_service, item.fault_type].append(incident_id)
        services = sorted({service for service, _ in pairs})
        faults = sorted({fault for _, fault in pairs})
        if len(services) != 5 or len(faults) != 6 or any(
            len(pairs.get((service, fault), [])) != 3
            for service in services for fault in faults
        ):
            raise ValueError("Se esperan 5 servicios × 6 fallos × 3 repeticiones")
        rng = random.Random(seed)
        rng.shuffle(services)
        rng.shuffle(faults)
        selected = {
            rng.choice(pairs[services[(3 * i + j) % 5], fault])
            for i, fault in enumerate(faults) for j in range(3)
        }
        result = cls(seed, _fingerprint(index), tuple(sorted(set(index) - selected)), tuple(sorted(selected)))
        result.validate(index.values())
        return result

    def validate(self, metadata: Iterable[IncidentMetadata]) -> None:
        """Comprobar cobertura, disjunción, proporciones e identidad de los casos."""
        index = _metadata_index(metadata)
        development, test = set(self.development), set(self.test)
        if len(self.development) != 72 or len(development) != 72 or len(self.test) != 18 or len(test) != 18:
            raise ValueError("La partición debe contener 72/18 identificadores únicos")
        if development & test or development | test != set(index):
            raise ValueError("La partición comparte incidentes o no coincide con el índice")
        if self.metadata_sha256 != _fingerprint(index):
            raise ValueError("Los metadatos difieren de los usados para crear la partición")
        service_counts = Counter(index[key].root_cause_service for key in test)
        fault_counts = Counter(index[key].fault_type for key in test)
        pair_counts = Counter((index[key].root_cause_service, index[key].fault_type) for key in test)
        if sorted(service_counts.values()) != [3, 3, 4, 4, 4] or sorted(fault_counts.values()) != [3] * 6:
            raise ValueError("La partición no conserva el equilibrio acordado")
        if max(pair_counts.values()) != 1:
            raise ValueError("Test debe usar parejas servicio/fallo distintas")

    def save(self, path: str | Path) -> None:
        """Crear el manifiesto sin sobrescribir una reserva previa de test."""
        document = {
            "schema_version": 1,
            "dataset": "RE2-SS",
            "method": "balanced_cyclic_service_fault",
            "seed": self.seed,
            "metadata_sha256": self.metadata_sha256,
            "development": list(self.development),
            "test": list(self.test),
        }
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

    @classmethod
    def load(cls, path: str | Path, metadata: Iterable[IncidentMetadata]) -> "IncidentSplit":
        """Reutilizar y validar un manifiesto sin volver a seleccionar incidentes."""
        with Path(path).open(encoding="utf-8") as stream:
            doc = json.load(stream)
        if not isinstance(doc, dict) or doc.get("schema_version") != 1 or doc.get("dataset") != "RE2-SS":
            raise ValueError("Manifiesto de partición no soportado")
        if doc.get("method") != "balanced_cyclic_service_fault":
            raise ValueError("Método de partición no soportado")
        if isinstance(doc.get("seed"), bool) or not isinstance(doc.get("seed"), int):
            raise ValueError("La semilla del manifiesto debe ser un entero")
        for field in ("development", "test"):
            if not isinstance(doc.get(field), list) or any(not isinstance(key, str) for key in doc[field]):
                raise ValueError(f"{field} debe contener una lista de identificadores")
        result = cls(doc["seed"], doc.get("metadata_sha256", ""), tuple(doc["development"]), tuple(doc["test"]))
        result.validate(metadata)
        return result
