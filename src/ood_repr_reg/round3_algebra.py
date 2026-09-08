"""Algebraic analysis of the raw OOD risk-response matrix (round 3A).

This module intentionally contains no clustering, mechanism naming, response
order analysis, or regularizer interpretation.  It treats a model and a shift
only through their lifted vectors in the common quadratic-risk feature space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .round3_response_matrix import ResponseMatrices
from .round3_shift_ensemble import PopulationEnvironment, ShiftProbe
from .round3_trained_models import TrainedModel, _stats, risk

Array = np.ndarray


@dataclass(frozen=True)
class AlgebraTolerance:
    """Numerical tolerances used by every rank and subspace calculation."""

    rank_relative: float = 1e-9
    residual_absolute: float = 1e-9
    angle_absolute: float = 1e-8


def svec_symmetric(matrix: Array) -> Array:
    """Frobenius-isometric symmetric vectorization.

    Diagonal entries are copied and strict upper-triangular entries are scaled
    by sqrt(2), so the Euclidean inner product equals the Frobenius product.
    """
    value = np.asarray(matrix, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    if not np.allclose(value, value.T, atol=1e-10):
        raise ValueError("matrix must be symmetric")
    entries = [value[i, i] for i in range(value.shape[0])]
    entries.extend(np.sqrt(2.0) * value[i, j] for i in range(value.shape[0]) for j in range(i + 1, value.shape[1]))
    return np.asarray(entries, dtype=float)


def smat_symmetric(vector: Array, dimension: int) -> Array:
    """Inverse of :func:`svec_symmetric`."""
    values = np.asarray(vector, dtype=float)
    expected = dimension * (dimension + 1) // 2
    if values.ndim != 1 or values.size != expected:
        raise ValueError("vector has incompatible symmetric dimension")
    matrix = np.zeros((dimension, dimension), dtype=float)
    cursor = 0
    for index in range(dimension):
        matrix[index, index] = values[cursor]
        cursor += 1
    for index in range(dimension):
        for other in range(index + 1, dimension):
            matrix[index, other] = matrix[other, index] = values[cursor] / np.sqrt(2.0)
            cursor += 1
    return matrix


def risk_statistic_labels(dimension: int) -> tuple[str, ...]:
    """Coordinate labels matching ``svec_symmetric`` plus linear statistics."""
    labels = [f"dM[{index},{index}]" for index in range(dimension)]
    labels.extend(f"sqrt2*dM[{index},{other}]" for index in range(dimension) for other in range(index + 1, dimension))
    labels.extend(f"dmXY[{index}]" for index in range(dimension))
    labels.append("dmY2")
    return tuple(labels)


def right_nullspace(matrix: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> Array:
    """Orthonormal basis for the right nullspace of a matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    _, singular, vh = np.linalg.svd(values, full_matrices=True)
    if singular.size == 0 or singular[0] == 0.0:
        rank = 0
    else:
        rank = int(np.sum(singular > tolerance.rank_relative * singular[0]))
    return vh[rank:].T


def coordinate_nullspace(matrix: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> dict[str, object]:
    """Return a deterministic, coordinate-anchored nullspace basis.

    Unlike an SVD basis, each vector fixes one non-pivot coordinate to one.
    This makes a null functional easier to read.  The basis is not canonical;
    the orthonormal basis from :func:`right_nullspace` remains the invariant
    object used for dimensions and geometry.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    n_columns = values.shape[1]
    selected: list[int] = []
    current_rank = 0
    for column in range(n_columns):
        candidate = values[:, selected + [column]]
        candidate_rank = rank_with_tolerance(candidate, tolerance)
        if candidate_rank > current_rank:
            selected.append(column)
            current_rank = candidate_rank
    pivots = tuple(selected)
    free = tuple(column for column in range(n_columns) if column not in pivots)
    basis = np.zeros((n_columns, len(free)), dtype=float)
    if pivots:
        pivot_block = values[:, pivots]
        for index, free_column in enumerate(free):
            basis[free_column, index] = 1.0
            basis[list(pivots), index] = np.linalg.lstsq(pivot_block, -values[:, free_column], rcond=None)[0]
    else:
        for index, free_column in enumerate(free):
            basis[free_column, index] = 1.0
    return {"basis": basis, "pivot_columns": pivots, "free_columns": free, "residual": values @ basis}


def risk_statistic_functional(coordinates: Array, matrix_dimension: int) -> dict[str, Array | float]:
    """Decode a null functional into risk-statistic coordinates.

    A returned vector ``n`` represents the functional
    ``<N_M, delta_M> + n_xy @ delta_mXY + n_y2 * delta_mY2``.
    """
    values = np.asarray(coordinates, dtype=float)
    expected = matrix_dimension * (matrix_dimension + 1) // 2 + matrix_dimension + 1
    if values.ndim != 1 or values.size != expected:
        raise ValueError("coordinates have incompatible risk-statistic dimension")
    matrix_size = matrix_dimension * (matrix_dimension + 1) // 2
    matrix_coordinates = values[:matrix_size]
    vector_coordinates = values[matrix_size : matrix_size + matrix_dimension]
    return {
        "delta_mxx_functional": smat_symmetric(matrix_coordinates, matrix_dimension),
        "delta_mxy_functional": vector_coordinates.copy(),
        "delta_my2_functional": float(values[-1]),
        "coordinate_vector": values.copy(),
        "frobenius_norm_mxx": float(np.linalg.norm(smat_symmetric(matrix_coordinates, matrix_dimension), ord="fro")),
        "l2_norm_mxy": float(np.linalg.norm(vector_coordinates)),
    }


def nullspace_functionals(
    shift_liftings: Array,
    matrix_dimension: int,
    tolerance: AlgebraTolerance = AlgebraTolerance(),
) -> dict[str, object]:
    """Return all shift-unexcited linear risk-statistic combinations."""
    values = np.asarray(shift_liftings, dtype=float)
    basis = right_nullspace(values, tolerance)
    return {
        "basis": basis,
        "dimension": int(basis.shape[1]),
        "coordinate_labels": risk_statistic_labels(matrix_dimension),
        "functionals": [risk_statistic_functional(basis[:, index], matrix_dimension) for index in range(basis.shape[1])],
        "max_annihilation_residual": float(np.max(np.abs(values @ basis))) if basis.size else 0.0,
    }


def symmetric_matrix_basis(dimension: int) -> tuple[Array, ...]:
    """Coordinate basis of symmetric matrices, including off-diagonal axes."""
    basis: list[Array] = []
    for index in range(dimension):
        for other in range(index, dimension):
            matrix = np.zeros((dimension, dimension), dtype=float)
            matrix[index, other] = 1.0
            matrix[other, index] = 1.0
            basis.append(matrix)
    return tuple(basis)


def pure_family_liftings(
    source: PopulationEnvironment,
    step: float = 0.08,
) -> tuple[dict[str, Array], dict[str, tuple[dict[str, object], ...]]]:
    """Build finite, one-structural-axis-at-a-time shift lifting families.

    Both signs are retained.  This is a finite-shift algebra audit, not a
    first/second-order decomposition; response-order interpretation is left
    for 3B.
    """
    eye_c = tuple(np.eye(source.c_dim)[index] for index in range(source.c_dim))
    eye_a = tuple(np.eye(source.a_dim)[index] for index in range(source.a_dim))
    specs: dict[str, tuple[tuple[str, Array], ...]] = {
        "core_mean": tuple(("mu_c", vector) for vector in eye_c),
        "core_covariance": tuple(("sigma_c", matrix) for matrix in symmetric_matrix_basis(source.c_dim)),
        "relation": tuple(("gamma", np.eye(source.a_dim * source.c_dim).reshape(source.a_dim * source.c_dim, source.a_dim, source.c_dim)[index]) for index in range(source.a_dim * source.c_dim)),
        "b": tuple(("b", vector) for vector in eye_a),
        "mu_xi": tuple(("mu_xi", vector) for vector in eye_a),
        "sigma_xi": tuple(("sigma_xi", matrix) for matrix in symmetric_matrix_basis(source.a_dim)),
        "task": tuple(("beta", vector) for vector in eye_c),
    }
    liftings: dict[str, Array] = {}
    metadata: dict[str, tuple[dict[str, object], ...]] = {}
    for family, axes in specs.items():
        columns: list[Array] = []
        records: list[dict[str, object]] = []
        for axis_index, (parameter, direction) in enumerate(axes):
            for sign in (-1, 1):
                target = source.with_delta({parameter: direction}, scale=sign * step)
                columns.append(shift_lift(target, source))
                records.append({"axis_index": axis_index, "parameter": parameter, "sign": sign, "step": step})
        liftings[family] = np.column_stack(columns) if columns else np.zeros((21, 0))
        metadata[family] = tuple(records)
    return liftings, metadata


def structural_reachability_liftings(
    source: PopulationEnvironment,
    count: int = 512,
    seed: int = 20260906,
    vary_task: bool = True,
) -> Array:
    """Sample valid structural points to audit design-only null directions.

    This is a numerical span diagnostic.  It is intentionally separate from
    the exact invariant that the intercept second moment is always constant.
    """
    rng = np.random.default_rng(seed)
    points: list[Array] = []
    for _ in range(count):
        core_factor = rng.normal(scale=0.65, size=(source.c_dim, source.c_dim))
        nuisance_factor = rng.normal(scale=0.55, size=(source.a_dim, source.a_dim))
        target = PopulationEnvironment(
            mu_c=source.mu_c + rng.normal(scale=0.55, size=source.c_dim),
            sigma_c=source.sigma_c + core_factor @ core_factor.T,
            gamma=source.gamma + rng.normal(scale=0.55, size=source.gamma.shape),
            b=source.b + rng.normal(scale=0.55, size=source.a_dim),
            mu_xi=source.mu_xi + rng.normal(scale=0.55, size=source.a_dim),
            sigma_xi=source.sigma_xi + nuisance_factor @ nuisance_factor.T,
            beta=source.beta + rng.normal(scale=0.45, size=source.c_dim) if vary_task else source.beta,
            noise_variance=source.noise_variance,
        )
        points.append(shift_lift(target, source))
    return np.asarray(points, dtype=float)


def model_lift(weights: Array) -> Array:
    """Return phi(w) for X=(1,C,A)."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    return np.concatenate((svec_symmetric(np.outer(w, w)), -2.0 * w, np.ones(1)))


def _moment_differences(target: PopulationEnvironment, source: PopulationEnvironment) -> tuple[Array, Array, float]:
    target_xx, target_xy, target_y2 = _stats(target)
    source_xx, source_xy, source_y2 = _stats(source)
    return target_xx - source_xx, target_xy - source_xy, float(target_y2 - source_y2)


def shift_lift(target: PopulationEnvironment, source: PopulationEnvironment) -> Array:
    """Return psi(T) in the same feature ordering as ``model_lift``."""
    delta_xx, delta_xy, delta_y2 = _moment_differences(target, source)
    return np.concatenate((svec_symmetric(delta_xx), delta_xy, np.asarray([delta_y2])))


def risk_response_pairing(weights: Array, target: PopulationEnvironment, source: PopulationEnvironment) -> float:
    return float(model_lift(weights) @ shift_lift(target, source))


def rank_with_tolerance(matrix: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> int:
    values = np.asarray(matrix, dtype=float)
    singular = np.linalg.svd(values, compute_uv=False)
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    return int(np.sum(singular > tolerance.rank_relative * singular[0]))


def subspace_basis(matrix: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> Array:
    """Orthonormal basis for the column space of a matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if min(values.shape) == 0:
        return np.zeros((values.shape[0], 0))
    u, singular, _ = np.linalg.svd(values, full_matrices=False)
    rank = 0 if singular.size == 0 or singular[0] == 0 else int(np.sum(singular > tolerance.rank_relative * singular[0]))
    return u[:, :rank]


def intersection_dimension(first: Array, second: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> int:
    """Dimension of the intersection of two column spaces."""
    a = subspace_basis(first, tolerance)
    b = subspace_basis(second, tolerance)
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must share an ambient dimension")
    return int(a.shape[1] + b.shape[1] - rank_with_tolerance(np.column_stack((a, b)), tolerance))


def principal_angles(first: Array, second: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> Array:
    """Principal angles in radians between two column spaces."""
    a = subspace_basis(first, tolerance)
    b = subspace_basis(second, tolerance)
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must share an ambient dimension")
    if a.shape[1] == 0 or b.shape[1] == 0:
        return np.zeros(0)
    singular = np.linalg.svd(a.T @ b, compute_uv=False)
    return np.arccos(np.clip(singular, -1.0, 1.0))


def inclusion_residual(subspace: Array, ambient: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> float:
    """Relative Frobenius residual after projecting one span into another."""
    a = subspace_basis(subspace, tolerance)
    b = subspace_basis(ambient, tolerance)
    if a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must share an ambient dimension")
    if a.shape[1] == 0:
        return 0.0
    if b.shape[1] == 0:
        return 1.0
    residual = a - b @ (b.T @ a)
    return float(np.linalg.norm(residual, ord="fro") / max(np.linalg.norm(a, ord="fro"), 1e-30))


def incremental_rank(first: Array, second: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> int:
    """New column-space dimension contributed by ``first`` after ``second``."""
    return rank_with_tolerance(np.column_stack((np.asarray(second), np.asarray(first))), tolerance) - rank_with_tolerance(second, tolerance)


def response_factorization(
    models: tuple[TrainedModel, ...],
    source: PopulationEnvironment,
    probes: tuple[ShiftProbe, ...],
) -> tuple[Array, Array, Array]:
    """Return Phi, Psi, and the direct raw response matrix."""
    phi = np.asarray([model_lift(model.weights) for model in models], dtype=float)
    psi = np.asarray([shift_lift(probe.target, source) for probe in probes], dtype=float)
    response = phi @ psi.T
    return phi, psi, response


def factorize_response(
    models: tuple[TrainedModel, ...],
    source: PopulationEnvironment,
    probes: tuple[ShiftProbe, ...],
) -> tuple[Array, Array, Array]:
    """Compatibility name for the public 3A factorization API."""
    return response_factorization(models, source, probes)


def response_rank(response: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> int:
    """Rank of a raw response matrix under the shared 3A tolerance."""
    return rank_with_tolerance(response, tolerance)


def factorization_from_matrices(
    models: tuple[TrainedModel, ...],
    source: PopulationEnvironment,
    probes: tuple[ShiftProbe, ...],
    matrices: ResponseMatrices | None = None,
) -> dict[str, object]:
    phi, psi, factored = response_factorization(models, source, probes)
    direct = np.asarray(matrices.raw if matrices is not None else [[risk(model.weights, probe.target) - risk(model.weights, source) for probe in probes] for model in models], dtype=float)
    residual = direct - factored
    tolerance = AlgebraTolerance()
    return {
        "phi": phi,
        "psi": psi,
        "response": direct,
        "factored_response": factored,
        "residual": residual,
        "residual_frobenius": float(np.linalg.norm(residual, ord="fro")),
        "residual_max_absolute": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "phi_rank": rank_with_tolerance(phi, tolerance),
        "psi_rank": rank_with_tolerance(psi, tolerance),
        "response_rank": rank_with_tolerance(direct, tolerance),
        "feature_dimension": int(phi.shape[1]),
        "tolerance": tolerance,
    }


def quotient_diagnostics(phi: Array, psi: Array, response: Array | None = None, tolerance: AlgebraTolerance = AlgebraTolerance()) -> dict[str, int | bool]:
    """Report the two algebraic kernels and the visible quotient dimension."""
    value = np.asarray(response if response is not None else np.asarray(phi) @ np.asarray(psi).T)
    phi_rank = rank_with_tolerance(phi, tolerance)
    psi_rank = rank_with_tolerance(psi, tolerance)
    response_rank = rank_with_tolerance(value, tolerance)
    return {
        "phi_rank": phi_rank,
        "psi_rank": psi_rank,
        "response_rank": response_rank,
        "model_blind_dimension": phi_rank - response_rank,
        "shift_blind_dimension": psi_rank - response_rank,
        "quotient_dimension": response_rank,
        "shift_full_visible": response_rank == psi_rank,
        "model_full_visible": response_rank == phi_rank,
    }


def grouped_response_geometry(response: Array, groups: dict[str, tuple[int, ...]], axis: str, tolerance: AlgebraTolerance = AlgebraTolerance()) -> dict[str, object]:
    """Compute ranks and pairwise geometry for response row/column groups.

    ``axis='shift'`` uses response[:, indices] as subspaces in model-response
    space. ``axis='model'`` uses response[indices, :].T in shift-response
    space. Group names are bookkeeping labels only.
    """
    value = np.asarray(response, dtype=float)
    spans: dict[str, Array] = {}
    ranks: dict[str, int] = {}
    for name, indices in groups.items():
        if axis == "shift":
            block = value[:, indices]
        elif axis == "model":
            block = value[indices, :].T
        else:
            raise ValueError("axis must be 'shift' or 'model'")
        spans[name] = block
        ranks[name] = rank_with_tolerance(block, tolerance)
    pairwise: dict[str, object] = {}
    names = list(groups)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            key = f"{first}__{second}"
            pairwise[key] = {
                "intersection_dimension": intersection_dimension(spans[first], spans[second], tolerance),
                "principal_angles": principal_angles(spans[first], spans[second], tolerance),
                "first_in_second_residual": inclusion_residual(spans[first], spans[second], tolerance),
                "second_in_first_residual": inclusion_residual(spans[second], spans[first], tolerance),
                "incremental_first_given_second": incremental_rank(spans[first], spans[second], tolerance),
                "incremental_second_given_first": incremental_rank(spans[second], spans[first], tolerance),
            }
    joint = np.column_stack([spans[name] for name in names]) if names else np.zeros((value.shape[0] if axis == "shift" else value.shape[1], 0))
    return {"axis": axis, "group_ranks": ranks, "joint_rank": rank_with_tolerance(joint, tolerance), "pairwise": pairwise, "spans": spans}


def family_incremental_ranks(response: Array, ordered_groups: Iterable[tuple[str, tuple[int, ...]]], axis: str, tolerance: AlgebraTolerance = AlgebraTolerance()) -> list[dict[str, int | str]]:
    value = np.asarray(response, dtype=float)
    current: Array | None = None
    rows: list[dict[str, int | str]] = []
    for name, indices in ordered_groups:
        block = value[:, indices] if axis == "shift" else value[indices, :].T
        before = rank_with_tolerance(current, tolerance) if current is not None else 0
        current = block if current is None else np.column_stack((current, block))
        after = rank_with_tolerance(current, tolerance)
        rows.append({"group": name, "rank_before": before, "rank_after": after, "incremental_rank": after - before})
    return rows
