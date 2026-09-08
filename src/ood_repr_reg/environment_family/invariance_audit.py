"""Audits separating chart changes from genuine family changes."""

from __future__ import annotations

import numpy as np

from .base import EnvironmentFamily
from .geometry import build_task_geometry
from .metrics import coordinate_transport, metric_operator_summary


def source_span_projector(states: np.ndarray, reference_index: int = 0,
                          tolerance: float = 1e-10) -> np.ndarray:
    value = np.asarray(states, dtype=float)
    if value.ndim != 2 or not 0 <= reference_index < value.shape[0]:
        raise ValueError("states and reference index are incompatible")
    contrasts = (value - value[reference_index]).T
    if contrasts.size == 0:
        return np.zeros((value.shape[1], value.shape[1]))
    u, singular, _ = np.linalg.svd(contrasts, full_matrices=False)
    rank = 0 if singular.size == 0 or singular[0] <= 0 else int(np.sum(singular > tolerance * singular[0]))
    return u[:, :rank] @ u[:, :rank].T


def family_coordinate_audit(family: EnvironmentFamily) -> dict[str, object]:
    geometry = build_task_geometry(family)
    dimension = geometry.spec.dimension
    rng = np.random.default_rng(17)
    recoding = rng.normal(size=(dimension, dimension)) + 2.0 * np.eye(dimension)
    response, observation, metric = coordinate_transport(
        geometry.response, geometry.observation, geometry.spec.metric, recoding,
    )
    original = metric_operator_summary(geometry.response, geometry.observation, geometry.spec.metric)
    transformed = metric_operator_summary(response, observation, metric)
    return {
        "family": geometry.spec.family_name,
        "general_invertible_recoding_error": abs(original["alpha"] - transformed["alpha"]),
        "invariant": bool(np.isclose(original["alpha"], transformed["alpha"], atol=1e-10)),
        "metric_transported": True,
        "uncertainty_geometry_changed_without_transport": False,
    }


def failure_classification(*, in_family: bool, source_visible: bool,
                           response_correct: bool) -> str:
    if not in_family:
        return "family_misspecification_or_omission"
    if not source_visible:
        return "information_failure"
    return "algorithm_failure" if not response_correct else "no_failure"


__all__ = ["source_span_projector", "family_coordinate_audit", "failure_classification"]
