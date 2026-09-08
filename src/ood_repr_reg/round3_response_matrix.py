"""Build raw and finite-difference risk-response matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3_shift_ensemble import PopulationEnvironment, ShiftProbe
from .round3_trained_models import TrainedModel, risk

Array = np.ndarray


@dataclass(frozen=True)
class ResponseMatrices:
    model_ids: tuple[str, ...]
    shift_ids: tuple[str, ...]
    raw: Array
    normalized_by_model: Array
    normalized_by_shift: Array
    first_order: Array
    second_order: Array
    probes: tuple[ShiftProbe, ...]


def _standardize_columns(matrix: Array) -> Array:
    values = np.asarray(matrix, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def _standardize_rows(matrix: Array) -> Array:
    values = np.asarray(matrix, dtype=float)
    mean = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return (values - mean) / scale


def build_response_matrices(models: tuple[TrainedModel, ...], source: PopulationEnvironment, probes: tuple[ShiftProbe, ...]) -> ResponseMatrices:
    raw = np.asarray([[risk(model.weights, probe.target) - risk(model.weights, source) for probe in probes] for model in models], dtype=float)
    pair_lookup = {probe.pair_id: {} for probe in probes if probe.pair_id is not None}
    for index, probe in enumerate(probes):
        if probe.pair_id is not None:
            pair_lookup[probe.pair_id][probe.sign] = index
    first = np.zeros_like(raw)
    second = np.zeros_like(raw)
    for pair in pair_lookup.values():
        if -1 not in pair or 1 not in pair:
            continue
        minus, plus = pair[-1], pair[1]
        first[:, minus] = (raw[:, plus] - raw[:, minus]) / 2.0
        first[:, plus] = first[:, minus]
        second[:, minus] = (raw[:, plus] + raw[:, minus]) / 2.0
        second[:, plus] = second[:, minus]
    return ResponseMatrices(
        tuple(model.model_id for model in models), tuple(probe.shift_id for probe in probes), raw,
        _standardize_rows(raw), _standardize_columns(raw), first, second, tuple(probes)
    )


def column_signatures(matrices: ResponseMatrices, normalized: str = "raw") -> Array:
    values = getattr(matrices, normalized)
    return np.asarray(values, dtype=float).T


def source_only_metadata(models: tuple[TrainedModel, ...]) -> dict[str, object]:
    return {
        "model_count": len(models),
        "source_risks": [float(model.source_risk) for model in models],
        "methods": [model.method for model in models],
        "uses_target_risk": False,
    }

