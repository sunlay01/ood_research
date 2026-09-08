"""Rank filtration utilities for the round 3B response-order audit."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .round3_3b_derivatives import Array
from .round3_algebra import AlgebraTolerance, inclusion_residual, intersection_dimension, rank_with_tolerance, subspace_basis


def order_ranks(order_matrices: dict[int, Array], tolerance: AlgebraTolerance = AlgebraTolerance()) -> dict[str, object]:
    """Compute individual and cumulative ranks and quotient increments."""
    individual: dict[str, int] = {}
    cumulative: dict[str, int] = {}
    increments: dict[str, int] = {}
    stacked: Array | None = None
    for order in sorted(order_matrices):
        matrix = np.asarray(order_matrices[order], dtype=float)
        individual[str(order)] = rank_with_tolerance(matrix, tolerance)
        before = rank_with_tolerance(stacked, tolerance) if stacked is not None else 0
        stacked = matrix if stacked is None else np.column_stack((stacked, matrix))
        after = rank_with_tolerance(stacked, tolerance)
        cumulative[str(order)] = after
        increments[str(order)] = after - before
    return {
        "individual_ranks": individual,
        "cumulative_ranks": cumulative,
        "incremental_ranks": increments,
        "cumulative_matrix": stacked if stacked is not None else np.zeros((0, 0)),
    }


def order_overlap(first: Array, second: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> dict[str, object]:
    """Report overlap and the genuinely new part of the second order space."""
    first_basis = subspace_basis(first, tolerance)
    second_basis = subspace_basis(second, tolerance)
    intersection = intersection_dimension(first, second, tolerance)
    combined_rank = rank_with_tolerance(np.column_stack((first, second)), tolerance)
    return {
        "first_rank": int(first_basis.shape[1]),
        "second_rank": int(second_basis.shape[1]),
        "intersection_dimension": int(intersection),
        "new_second_dimension": int(second_basis.shape[1] - intersection),
        "combined_rank": int(combined_rank),
        "second_in_first_residual": inclusion_residual(second, first, tolerance),
        "first_in_second_residual": inclusion_residual(first, second, tolerance),
    }


def project_residual(ambient: Array, subspace: Array, tolerance: AlgebraTolerance = AlgebraTolerance()) -> float:
    """Relative residual of columns of ``ambient`` outside ``span(subspace)``."""
    value = np.asarray(ambient, dtype=float)
    basis = subspace_basis(subspace, tolerance)
    if value.size == 0:
        return 0.0
    projected = basis @ (basis.T @ value) if basis.shape[1] else np.zeros_like(value)
    return float(np.linalg.norm(value - projected) / max(np.linalg.norm(value), 1e-30))


def rank_profile(matrix: Array, tolerances: Iterable[float]) -> list[dict[str, float | int]]:
    """Rank sensitivity profile for an array of relative SVD thresholds."""
    rows = []
    for relative in tolerances:
        tolerance = AlgebraTolerance(rank_relative=float(relative))
        rows.append({"rank_relative": float(relative), "rank": rank_with_tolerance(matrix, tolerance)})
    return rows

