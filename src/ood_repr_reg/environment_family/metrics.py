"""Metric-aware finite-dimensional operator geometry for family tangents."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def validate_metric(metric: Array | None, dimension: int) -> Array:
    value = np.eye(dimension) if metric is None else np.asarray(metric, dtype=float)
    if value.shape != (dimension, dimension):
        raise ValueError("metric has incompatible shape")
    value = (value + value.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(value)
    if not np.all(np.isfinite(value)) or (eigenvalues.size and eigenvalues.min() <= 0.0):
        raise ValueError("metric must be finite and positive definite")
    return value


def _roots(metric: Array) -> tuple[Array, Array, Array]:
    value = validate_metric(metric, np.asarray(metric).shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_root = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return root, inverse_root, eigenvalues


def world_whiten(matrix: Array, metric: Array | None = None) -> Array:
    """Map a world-coordinate operator to a Euclidean unit-ball chart."""
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if metric is None:
        return value.copy()
    _, inverse_root, _ = _roots(np.asarray(metric, dtype=float))
    if inverse_root.shape[0] != value.shape[1]:
        raise ValueError("metric has incompatible world dimension")
    return value @ inverse_root


def metric_operator_norm(matrix: Array, metric: Array | None = None) -> float:
    value = world_whiten(matrix, metric)
    singular = np.linalg.svd(value, compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


def _null_basis_euclidean(observation: Array, tolerance: float) -> Array:
    value = np.asarray(observation, dtype=float)
    _, singular, vh = np.linalg.svd(value, full_matrices=True)
    rank = 0 if singular.size == 0 or singular[0] <= 0.0 else int(np.sum(singular > tolerance * singular[0]))
    return vh[rank:].T.copy()


def metric_nullspace_basis(observation: Array, metric: Array | None = None,
                           tolerance: float = 1e-10) -> Array:
    value = np.asarray(observation, dtype=float)
    if value.ndim != 2:
        raise ValueError("observation must be a matrix")
    basis = _null_basis_euclidean(world_whiten(value, metric), tolerance)
    if metric is None or basis.shape[1] == 0:
        return basis
    _, inverse_root, _ = _roots(np.asarray(metric, dtype=float))
    # The null basis above is in the Euclidean whitened coordinates.  Mapping it
    # back gives a basis for the original tangent kernel.
    return inverse_root @ basis


def metric_minimax_error(response: Array, observation: Array, metric: Array | None = None,
                         tolerance: float = 1e-10) -> float:
    a = world_whiten(response, metric)
    o = world_whiten(observation, metric)
    if a.shape[1] != o.shape[1]:
        raise ValueError("response and observation must share the world dimension")
    null = _null_basis_euclidean(o, tolerance)
    singular = np.linalg.svd(a @ null, compute_uv=False) if null.shape[1] else np.zeros(0)
    return float(singular[0]) if singular.size else 0.0


def metric_information_floor(response: Array, observation: Array, metric: Array | None = None,
                             tolerance: float = 1e-10) -> float:
    return 0.5 * metric_minimax_error(response, observation, metric, tolerance) ** 2


def metric_operator_summary(response: Array, observation: Array, metric: Array | None = None,
                            tolerance: float = 1e-10) -> dict[str, object]:
    a = np.asarray(response, dtype=float)
    o = np.asarray(observation, dtype=float)
    if a.ndim != 2 or o.ndim != 2 or a.shape[1] != o.shape[1]:
        raise ValueError("response and observation must be compatible matrices")
    whitened_a = world_whiten(a, metric)
    whitened_o = world_whiten(o, metric)
    _, singular_o, _ = np.linalg.svd(whitened_o, full_matrices=True)
    rank_o = 0 if singular_o.size == 0 or singular_o[0] <= 0.0 else int(np.sum(singular_o > tolerance * singular_o[0]))
    null = _null_basis_euclidean(whitened_o, tolerance)
    singular_a = np.linalg.svd(whitened_a, compute_uv=False)
    alpha = metric_minimax_error(a, o, metric, tolerance)
    response_norm = float(singular_a[0]) if singular_a.size else 0.0
    return {
        "dim_U": int(o.shape[1]),
        "source_dimension": int(o.shape[0]),
        "response_dimension": int(a.shape[0]),
        "rank_O": rank_o,
        "kernel_dimension": int(null.shape[1]),
        "rank_A": int(np.sum(singular_a > tolerance * singular_a[0])) if singular_a.size and singular_a[0] > 0 else 0,
        "alpha": alpha,
        "normalized_alpha": alpha / response_norm if response_norm > 0 else 0.0,
        "information_floor": 0.5 * alpha * alpha,
        "ambiguity_diameter": 2.0 * alpha,
        "metric": validate_metric(metric, o.shape[1]).tolist(),
        "tolerance": float(tolerance),
    }


def coordinate_transport(response: Array, observation: Array, metric: Array,
                         recoding: Array) -> tuple[Array, Array, Array]:
    """Transport `(A,O,G)` under the new coordinates ``u_new=B u_old``."""
    a = np.asarray(response, dtype=float)
    o = np.asarray(observation, dtype=float)
    g = validate_metric(metric, a.shape[1])
    b = np.asarray(recoding, dtype=float)
    if b.shape != (a.shape[1], a.shape[1]) or abs(np.linalg.det(b)) <= 1e-12:
        raise ValueError("recoding must be invertible on the world tangent")
    inverse = np.linalg.inv(b)
    transported_metric = inverse.T @ g @ inverse
    return a @ inverse, o @ inverse, transported_metric


__all__ = [
    "validate_metric", "world_whiten", "metric_operator_norm", "metric_nullspace_basis",
    "metric_minimax_error", "metric_information_floor", "metric_operator_summary",
    "coordinate_transport",
]
