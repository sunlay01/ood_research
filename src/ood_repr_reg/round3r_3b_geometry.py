"""Intrinsic first-order relevance geometry for 3B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class GeometryTolerance:
    rank_relative: float = 1e-9
    relevance_relative: float = 1e-10
    residual_absolute: float = 1e-9


def _symmetric(matrix: Array) -> Array:
    value = np.asarray(matrix, dtype=float)
    return (value + value.T) / 2.0


def _inverse_sqrt(matrix: Array) -> Array:
    values, vectors = np.linalg.eigh(_symmetric(matrix))
    if values.size == 0 or np.min(values) <= 0:
        raise ValueError("Hessian must be positive definite")
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def whitened_responses(hessian: Array, gradients: Array) -> Array:
    """Return columns ``H^{-1/2} g_s``."""
    values = np.asarray(gradients, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] != np.asarray(hessian).shape[0]:
        raise ValueError("gradient matrix has incompatible shape")
    return _inverse_sqrt(hessian) @ values


def relevance_gram(hessian: Array, gradients: Array) -> Array:
    """Compute the intrinsic Gram matrix ``G_ij = g_i' H^-1 g_j``."""
    values = np.asarray(gradients, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    return values.T @ np.linalg.solve(_symmetric(hessian), values)


def filter_relevant_shifts(gram: Array, threshold: float = 1e-10) -> dict[str, object]:
    value = _symmetric(np.asarray(gram, dtype=float))
    diagonal = np.maximum(np.diag(value), 0.0)
    scale = max(float(np.max(diagonal, initial=0.0)), 1e-30)
    keep = diagonal > threshold * scale
    return {"mask": keep, "indices": np.flatnonzero(keep), "threshold": threshold, "scale": scale,
            "relevance_squared": diagonal, "n_filtered": int(np.sum(~keep))}


def matrix_rank(matrix: Array, tolerance: GeometryTolerance = GeometryTolerance()) -> int:
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    return int(np.sum(singular > tolerance.rank_relative * singular[0])) if singular.size and singular[0] > 0 else 0


def subspace_basis(matrix: Array, tolerance: GeometryTolerance = GeometryTolerance()) -> Array:
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if min(value.shape) == 0:
        return np.zeros((value.shape[0], 0))
    u, singular, _ = np.linalg.svd(value, full_matrices=False)
    rank = int(np.sum(singular > tolerance.rank_relative * singular[0])) if singular[0] > 0 else 0
    return u[:, :rank]


def module_subspaces(responses: Array, assignments: Array, tolerance: GeometryTolerance = GeometryTolerance()) -> dict[str, Array]:
    value = np.asarray(responses, dtype=float)
    labels = np.asarray(assignments, dtype=int)
    if value.ndim != 2 or value.shape[1] != labels.size:
        raise ValueError("responses and assignments have incompatible shapes")
    return {str(label): subspace_basis(value[:, labels == label], tolerance) for label in sorted(set(labels.tolist()))}


def principal_angles(first: Array, second: Array, tolerance: GeometryTolerance = GeometryTolerance()) -> Array:
    a, b = subspace_basis(first, tolerance), subspace_basis(second, tolerance)
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must share ambient dimension")
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.zeros(0)
    return np.arccos(np.clip(np.linalg.svd(a.T @ b, compute_uv=False), -1.0, 1.0))


def intersection_dimension(first: Array, second: Array, tolerance: GeometryTolerance = GeometryTolerance()) -> int:
    a, b = subspace_basis(first, tolerance), subspace_basis(second, tolerance)
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must share ambient dimension")
    return a.shape[1] + b.shape[1] - matrix_rank(np.column_stack((a, b)), tolerance)


def inclusion_residual(subspace: Array, ambient: Array,
                       tolerance: GeometryTolerance = GeometryTolerance()) -> float:
    """Distance of columns of ``subspace`` from the span of ``ambient``."""
    a, b = subspace_basis(subspace, tolerance), subspace_basis(ambient, tolerance)
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must share ambient dimension")
    if a.shape[1] == 0:
        return 0.0
    projection = b @ (b.T @ a) if b.shape[1] else np.zeros_like(a)
    return float(np.linalg.norm(a - projection))


def incremental_rank(first: Array, second: Array,
                     tolerance: GeometryTolerance = GeometryTolerance()) -> int:
    """Rank contributed by ``first`` after the span of ``second``."""
    return matrix_rank(np.column_stack((np.asarray(first), np.asarray(second))), tolerance) - matrix_rank(second, tolerance)


def mixed_shift_decomposition(response: Array, modules: dict[str, Array], tolerance: GeometryTolerance = GeometryTolerance()) -> dict[str, object]:
    value = np.asarray(response, dtype=float)
    bases = [modules[key] for key in sorted(modules)]
    joint = np.column_stack(bases) if bases else np.zeros((value.size, 0))
    if joint.shape[1] == 0:
        coefficients = np.zeros(0)
        reconstruction = np.zeros_like(value)
    else:
        coefficients, *_ = np.linalg.lstsq(joint, value, rcond=None)
        reconstruction = joint @ coefficients
    return {"coefficients": coefficients, "reconstruction": reconstruction,
            "residual_norm": float(np.linalg.norm(value - reconstruction)),
            "joint_rank": matrix_rank(joint, tolerance),
            "joint_columns": int(joint.shape[1]),
            "unique_if_direct_sum": bool(matrix_rank(joint, tolerance) == joint.shape[1])}


def mixed_shift_reconstruction(response: Array, modules: dict[str, Array],
                               tolerance: GeometryTolerance = GeometryTolerance()) -> dict[str, object]:
    """Named API for the minimum-norm module reconstruction diagnostic."""
    return mixed_shift_decomposition(response, modules, tolerance)


def identifiability_diagnostics(modules: dict[str, Array],
                                tolerance: GeometryTolerance = GeometryTolerance()) -> dict[str, object]:
    keys = sorted(modules)
    intersections = {}
    for index, first in enumerate(keys):
        for second in keys[index + 1:]:
            intersections[f"{first}|{second}"] = intersection_dimension(modules[first], modules[second], tolerance)
    joint = np.column_stack([modules[key] for key in keys]) if keys else np.zeros((0, 0))
    return {"intersection_dimensions": intersections, "joint_rank": matrix_rank(joint, tolerance),
            "joint_columns": int(joint.shape[1]),
            "unique_if_direct_sum": bool(matrix_rank(joint, tolerance) == joint.shape[1])}


def transformed_geometry(hessian: Array, gradients: Array, transform: Array) -> dict[str, Array]:
    """Apply consistent predictor reparameterization and return its geometry."""
    t = np.asarray(transform, dtype=float)
    if t.ndim != 2 or t.shape[0] != t.shape[1] or abs(np.linalg.det(t)) < 1e-12:
        raise ValueError("transform must be invertible")
    inv = np.linalg.inv(t)
    hp = inv.T @ np.asarray(hessian) @ inv
    gp = inv.T @ np.asarray(gradients)
    return {"hessian": hp, "gradients": gp, "gram": relevance_gram(hp, gp),
            "responses": whitened_responses(hp, gp)}


def rotation_diagnostics(hessian: Array, gradients: Array, transforms: list[Array]) -> dict[str, object]:
    """Audit Gram invariance under a list of invertible predictor transforms."""
    reference = relevance_gram(hessian, gradients)
    rows = []
    for index, transform in enumerate(transforms):
        current = transformed_geometry(hessian, gradients, transform)["gram"]
        rows.append({"index": index, "max_gram_error": float(np.max(np.abs(current - reference)))})
    return {"rows": rows, "max_gram_error": max((row["max_gram_error"] for row in rows), default=0.0)}
