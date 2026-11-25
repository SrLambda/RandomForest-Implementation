# Tolerancias por métrica ajustadas 

from collections import defaultdict
from pathlib import Path
import random
from typing import Any

import numpy as np

from .configuration import TopologyConfig
from .evaluation import IncidentSplit
from .ingestion import RCAEvalLoader


def fit_deviation_floors(
    configuration: TopologyConfig, split_path: str | Path,
    seed: int = 42, quantile: float = .999, margin: float = 2.,
    reference_scale_quantile: float = .05,
) -> dict[str, Any]:
    #Ajustar con 48 casos y reservar 24 de desarrollo para comprobación.
    #Por cada tipo de fallo se reservan 4 casos para validación interna. 
    
    if not 0 < quantile <= 1:
        raise ValueError('El cuantil debe estar entre 0 y 1')
    if not np.isfinite(margin) or margin < 1:
        raise ValueError('El margen debe ser finito y al menos 1')
    if not 0 <= reference_scale_quantile <= 1:
        raise ValueError('El cuantil de escala debe estar entre 0 y 1')
    loader = RCAEvalLoader(configuration=configuration)
    metadata = [loader.load_metadata(key) for key in loader.list_incidents()]
    split = IncidentSplit.load(split_path, metadata)
    by_id = {item.incident_id: item for item in metadata}
    by_fault: dict[str, list[str]] = defaultdict(list)
    for key in split.development:
        by_fault[by_id[key].fault_type].append(key)
    rng = random.Random(seed)
    validation: list[str] = []
    for fault in sorted(by_fault):
        group = sorted(by_fault[fault])
        rng.shuffle(group)
        validation.extend(group[:4])
    fitting = sorted(set(split.development) - set(validation))
    deviations: dict[str, list[np.ndarray]] = defaultdict(list)
    reference_scales: dict[str, list[float]] = defaultdict(list)
    warmup = int(configuration.detector_config['warmup_sec'])
    for key in fitting:
        frame = loader.load_metrics(key).frame
        normal = frame.loc[frame.time < by_id[key].injection_timestamp].drop(columns='time')
        if len(normal) <= warmup:
            raise ValueError(f'No hay histórico normal suficiente: {key}')
        centers = normal.iloc[:warmup].mean()
        initial_std = normal.iloc[:warmup].std(ddof=0)
        for column in normal:
            if np.isfinite(centers[column]):
                scale = max(abs(centers[column]), initial_std[column])
                if np.isfinite(scale):
                    reference_scales[column].append(float(scale))
                delta = np.abs(normal[column].iloc[warmup:].to_numpy() - centers[column])
                finite = delta[np.isfinite(delta)]
                if finite.size:
                    deviations[column].append(finite)
    return {
        'schema_version': 1, 'dataset': 'RE2-SS',
        'method': 'normal_deviation_from_warmup_mean',
        'seed': seed, 'quantile': quantile, 'margin': margin,
        'warmup_sec': warmup,
        'reference_scale_quantile': reference_scale_quantile,
        'metadata_sha256': split.metadata_sha256,
        'fit_incidents': fitting, 'validation_incidents': sorted(validation),
        'minimum_deviation': {
            column: float(margin * np.quantile(np.concatenate(parts), quantile))
            for column, parts in sorted(deviations.items())
        },
        # Piso para no amplificar un arranque numéricamente cercano a cero.
        'minimum_reference_scale': {
            column: float(np.quantile(values, reference_scale_quantile))
            for column, values in sorted(reference_scales.items())
        },
        'uncalibrated_metrics': sorted(configuration.metric_columns() - deviations.keys()),
    }
