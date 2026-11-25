# Ventanas causales con baseline protegido, sin consultar etiquetas de fallo

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from .configuration import TopologyConfig
from .ingestion import MetricTimeSeries


class DetectorState(str, Enum):
    WARMUP = "WARMUP"
    NORMAL = "NORMAL"
    OBSERVATION = "OBSERVATION"
    ALERT = "ALERT"


class SampleStatus(str, Enum):
    HEALTHY = "HEALTHY"
    ANOMALOUS = "ANOMALOUS"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DetectorEvent:
    # Se adjuntan copias del baseline anterior a la sospecha y de las observaciones hasta ese instante, sin etiquetas.    

    kind: str
    timestamp: int
    incident_id: str
    alert_id: str | None = None
    start_timestamp: int | None = None
    recovered: bool = False
    baseline: pd.DataFrame | None = None
    observations: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class DetectionUpdate:
    timestamp: int
    state: DetectorState
    status: SampleStatus
    anomalous_metrics: tuple[str, ...]
    unavailable_metrics: tuple[str, ...]
    events: tuple[DetectorEvent, ...]


@dataclass(slots=True)
class _PendingDiagnosis:
    alert_id: str
    due: int
    start: int
    baseline: list[tuple[int, np.ndarray]]
    observations: list[tuple[int, np.ndarray]] = field(default_factory=list)
    recovered: bool = False


class WindowDetector:
    # Detector a 1 Hz, cada run inicia un caso independiente.
    # Baseline: últimas muestras aceptadas, congeladas ante sospechas/faltantes.
    # El calentamiento presupone un arranque sano; no consulta la inyección.
    
    def __init__(self, configuration: TopologyConfig) -> None:
        self._configuration = configuration
        p = configuration.detector_config
        self._baseline_size = int(p["baseline_window_sec"])
        self._warmup = int(p["warmup_sec"])
        self._threshold = float(p["z_score_threshold"])
        self._confirmation = int(p["anomaly_consecutive_sec"])
        self._recovery = int(p["recovery_consecutive_sec"])
        self._delay = int(p["diagnosis_delay_sec"])
        self._maximum = int(p["max_anomaly_window_sec"])

    def _classify(self, values: np.ndarray, baseline: deque[tuple[int, np.ndarray]],) -> tuple[SampleStatus, np.ndarray, np.ndarray]:
        history = np.stack([sample for _, sample in baseline])
        finite = np.isfinite(history)
        counts = finite.sum(axis=0)
        means = np.divide(
            np.where(finite, history, 0).sum(axis=0), counts,
            out=np.zeros(len(values)), where=counts > 0,
        )
        variance = np.divide(
            np.where(finite, (history - means) ** 2, 0).sum(axis=0), counts,
            out=np.zeros(len(values)), where=counts > 0,
        )
        std = np.sqrt(variance)
        available = np.isfinite(values) & (counts >= 2)
        anomalous = available & (
            ((std > 0) & (np.abs(values - means) > self._threshold * std))
            | ((std == 0) & ~np.isclose(values, means, rtol=1e-9, atol=0))
        )
        status = (SampleStatus.ANOMALOUS if anomalous.any() else
                  SampleStatus.HEALTHY if available.all() else SampleStatus.UNKNOWN)
        return status, anomalous, ~available

    @staticmethod
    def _frame(rows: list[tuple[int, np.ndarray]], names: list[str]) -> pd.DataFrame:
        result = pd.DataFrame([values for _, values in rows], columns=names)
        result.insert(0, "time", [timestamp for timestamp, _ in rows])
        return result

    def run(self, metrics: MetricTimeSeries) -> Iterator[DetectionUpdate]:
        # Emitir una actualización por muestra, sin pausas artificiales.

        frame = metrics.frame
        self._configuration.validate_columns(frame.columns.tolist())
        names = [name for name in frame.columns if name != "time"]
        baseline: deque[tuple[int, np.ndarray]] = deque(maxlen=self._baseline_size)
        pending: list[_PendingDiagnosis] = []
        state = DetectorState.WARMUP
        anomaly_count = recovery_count = alert_number = 0
        # Acotar memoria si OBSERVATION se prolonga sin confirmar ni recuperar.
        suspicion: deque[tuple[int, np.ndarray]] = deque(maxlen=self._maximum)
        suspicion_start: int | None = None
        reference: list[tuple[int, np.ndarray]] = []
        active_id: str | None = None
        segment_start: int | None = None
        segment_count = 0

        for position, row in enumerate(frame[["time", *names]].itertuples(index=False, name=None)):
            timestamp = int(row[0])
            values = np.asarray(row[1:], dtype=float)
            current = (timestamp, values)
            events: list[DetectorEvent] = []
            for diagnosis in pending:
                diagnosis.observations.append(current)

            if position < self._warmup:
                baseline.append(current)
                status = SampleStatus.UNKNOWN
                anomalous = np.zeros(len(names), dtype=bool)
                unavailable = ~np.isfinite(values)
                if position + 1 == self._warmup:
                    state = DetectorState.NORMAL
                    events.append(DetectorEvent("WARMUP_COMPLETE", timestamp, metrics.incident_id))
            else:
                status, anomalous, unavailable = self._classify(values, baseline)
                if state == DetectorState.NORMAL:
                    if status == SampleStatus.ANOMALOUS:
                        state = DetectorState.OBSERVATION
                        reference = list(baseline)
                        suspicion.clear()
                        suspicion.append(current)
                        suspicion_start = timestamp
                        anomaly_count = 1
                        recovery_count = 0
                    elif status == SampleStatus.HEALTHY:
                        baseline.append(current)
                else:
                    if state == DetectorState.OBSERVATION:
                        suspicion.append(current)
                    anomaly_count = anomaly_count + 1 if status == SampleStatus.ANOMALOUS else 0
                    recovery_count = recovery_count + 1 if status == SampleStatus.HEALTHY else 0

                if state == DetectorState.OBSERVATION and anomaly_count >= self._confirmation:
                    state = DetectorState.ALERT
                    alert_number += 1
                    active_id = f"{metrics.incident_id}:alert-{alert_number}"
                    pending.append(_PendingDiagnosis(
                        active_id, timestamp + self._delay, int(suspicion_start),
                        reference, list(suspicion),
                    ))
                    events.append(DetectorEvent(
                        "ALERT_CONFIRMED", timestamp, metrics.incident_id,
                        active_id, suspicion_start,
                    ))
                    segment_start = timestamp
                    segment_count = 0
                    suspicion.clear()
                    reference = []

                if state == DetectorState.ALERT:
                    segment_count += 1

                if state in (DetectorState.OBSERVATION, DetectorState.ALERT) and recovery_count >= self._recovery:
                    if state == DetectorState.ALERT:
                        for diagnosis in pending:
                            if diagnosis.alert_id == active_id:
                                diagnosis.recovered = True
                        events.append(DetectorEvent(
                            "RECOVERED", timestamp, metrics.incident_id,
                            active_id, segment_start, recovered=True,
                        ))
                    else:
                        events.append(DetectorEvent("OBSERVATION_DISCARDED", timestamp, metrics.incident_id))
                    state = DetectorState.NORMAL
                    anomaly_count = recovery_count = 0
                    suspicion.clear()
                    suspicion_start = None
                    reference = []
                    active_id = None
                    segment_start = None
                    segment_count = 0
                    # Descartar también muestras usadas para confirmar recuperación.
                    # La siguiente muestra sana reanudará el baseline.
                elif state == DetectorState.ALERT and segment_count >= self._maximum:
                    events.append(DetectorEvent(
                        "SEGMENT_CLOSED", timestamp, metrics.incident_id,
                        active_id, segment_start,
                    ))
                    segment_start = timestamp + 1
                    segment_count = 0

            for diagnosis in pending:
                if timestamp >= diagnosis.due:
                    events.append(DetectorEvent(
                        "DIAGNOSIS_DUE", timestamp, metrics.incident_id,
                        diagnosis.alert_id, diagnosis.start, diagnosis.recovered,
                        self._frame(diagnosis.baseline, names),
                        self._frame(diagnosis.observations, names),
                    ))
            pending = [diagnosis for diagnosis in pending if diagnosis.due > timestamp]
            if position + 1 == len(frame):
                for diagnosis in pending:
                    events.append(DetectorEvent(
                        "DIAGNOSIS_INCOMPLETE", timestamp, metrics.incident_id,
                        diagnosis.alert_id, diagnosis.start, diagnosis.recovered,
                    ))
                if position + 1 < self._warmup:
                    events.append(DetectorEvent("WARMUP_INCOMPLETE", timestamp, metrics.incident_id))
                if state == DetectorState.OBSERVATION:
                    events.append(DetectorEvent(
                        "OBSERVATION_INCOMPLETE", timestamp, metrics.incident_id,
                        start_timestamp=suspicion_start,
                    ))
                if state == DetectorState.ALERT and segment_count:
                    events.append(DetectorEvent(
                        "SEGMENT_INCOMPLETE", timestamp, metrics.incident_id,
                        active_id, segment_start,
                    ))
            yield DetectionUpdate(
                timestamp, state, status,
                tuple(name for name, flag in zip(names, anomalous) if flag),
                tuple(name for name, flag in zip(names, unavailable) if flag),
                tuple(events),
            )
