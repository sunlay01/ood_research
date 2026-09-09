"""Source-design exposure geometry for 3D."""

from __future__ import annotations

import numpy as np

from .round3r_3b_benchmark import MomentState
from .round3r_3b_geometry import GeometryTolerance, matrix_rank, subspace_basis
from .round3r_3d_state import state_components, state_difference, task_state

Array = np.ndarray


def _tolerance(value: GeometryTolerance | float) -> GeometryTolerance:
    """Preserve the legacy float API while plumbing rank tolerance correctly."""
    if isinstance(value, GeometryTolerance):
        return value
    return GeometryTolerance(rank_relative=float(value))


def _inverse_sqrt(matrix: Array) -> Array:
    value = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.size == 0 or eigenvalues.min() <= 0:
        raise ValueError("source Hessian must be positive definite")
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def _state_matrix(state: MomentState | Array) -> Array:
    return np.asarray(state if isinstance(state, np.ndarray) else task_state(state), dtype=float)


def source_design_span(source_states: list[MomentState | Array] | tuple[MomentState | Array, ...], reference_index: int = 0,
                       tolerance: GeometryTolerance | float = 1e-9) -> dict[str, object]:
    """Compute the source contrast span in observable state coordinates."""
    if not source_states or not 0 <= reference_index < len(source_states):
        raise ValueError("at least one source state is required")
    vectors = [_state_matrix(state) for state in source_states]
    reference = vectors[reference_index]
    contrasts = np.column_stack([value - reference for value in vectors]) if len(vectors) > 1 else np.zeros((reference.size, 0))
    tol = _tolerance(tolerance)
    basis = subspace_basis(contrasts, tol)
    return {
        "reference": reference,
        "states": np.column_stack(vectors),
        "contrasts": contrasts,
        "basis": basis,
        "rank": int(basis.shape[1]),
        "reference_index": int(reference_index),
        "tolerance": tol.rank_relative,
    }


def response_operator(source_state: MomentState, source_optimum: Array,
                      shift_state: MomentState | Array | None = None) -> Array:
    """Return the whitened response of a shift, or the linear operator matrix.

    For a concrete ``shift_state`` this returns
    ``H_S^{-1/2}(2 DeltaM w* - 2 Delta m)``.  With no shift it returns the
    matrix mapping the concatenated state coordinates to that response.
    """
    source_matrix = np.asarray(source_state.second, dtype=float)
    weights = np.asarray(source_optimum, dtype=float)
    inverse_root = _inverse_sqrt(2.0 * source_matrix)
    dimension = source_matrix.shape[0]
    state_length = dimension * (dimension + 1) // 2 + dimension + 1
    if shift_state is not None:
        if isinstance(shift_state, MomentState):
            target_matrix, target_cross = shift_state.second, shift_state.xy
        else:
            target_matrix, target_cross, _ = state_components(shift_state)
        delta_matrix = target_matrix - source_state.second
        delta_cross = target_cross - source_state.xy
        return inverse_root @ (2.0 * delta_matrix @ weights - 2.0 * delta_cross)
    operator = np.zeros((dimension, state_length), dtype=float)
    cursor = 0
    for row in range(dimension):
        operator[:, cursor] = 2.0 * inverse_root[:, row] * weights[row]
        cursor += 1
        for col in range(row):
            operator[:, cursor] = 2.0 * inverse_root[:, row] * weights[col] / np.sqrt(2.0)
            operator[:, cursor] += 2.0 * inverse_root[:, col] * weights[row] / np.sqrt(2.0)
            cursor += 1
    operator[:, cursor:cursor + dimension] = -2.0 * inverse_root
    return operator


def exposed_response_basis(source_contrasts: Array, response_map: Array,
                           tolerance: GeometryTolerance | float = 1e-9) -> dict[str, object]:
    contrasts = np.asarray(source_contrasts, dtype=float)
    response_map = np.asarray(response_map, dtype=float)
    if contrasts.ndim != 2 or response_map.ndim != 2 or response_map.shape[1] != contrasts.shape[0]:
        raise ValueError("source contrasts and response map have incompatible shapes")
    image = response_map @ contrasts
    tol = _tolerance(tolerance)
    basis = subspace_basis(image, tol)
    return {"responses": image, "basis": basis, "rank": int(basis.shape[1]),
            "input_rank": matrix_rank(contrasts, tol), "tolerance": tol.rank_relative}


def target_exposure_fraction(target_response: Array, exposed_basis: Array, tolerance: float = 1e-12) -> float:
    value = np.asarray(target_response, dtype=float)
    basis = np.asarray(exposed_basis, dtype=float)
    denominator = float(value @ value)
    if denominator <= tolerance:
        return 1.0
    orthogonal = subspace_basis(basis)
    projection = orthogonal @ (orthogonal.T @ value) if orthogonal.shape[1] else np.zeros_like(value)
    return float(projection @ projection / denominator)


def exact_exposure_membership(target_response: Array, exposed_basis: Array, tolerance: float = 1e-9) -> bool:
    value = np.asarray(target_response, dtype=float)
    basis = subspace_basis(np.asarray(exposed_basis, dtype=float))
    projection = basis @ (basis.T @ value) if basis.shape[1] else np.zeros_like(value)
    return bool(np.linalg.norm(value - projection) <= tolerance * max(1.0, np.linalg.norm(value)))


def quotient_dimensions(total_basis: Array, exposed_basis: Array, tolerance: GeometryTolerance | float = 1e-9) -> dict[str, object]:
    tol = _tolerance(tolerance)
    total = subspace_basis(np.asarray(total_basis, dtype=float), tol)
    exposed = subspace_basis(np.asarray(exposed_basis, dtype=float), tol)
    if total.shape[0] != exposed.shape[0]:
        raise ValueError("bases must share ambient response dimension")
    combined = np.column_stack((exposed, total))
    rank_total = matrix_rank(total, tol)
    rank_exposed = matrix_rank(exposed, tol)
    # The image is expected to be contained in the total response space; the
    # residual is reported so a malformed caller is visible.
    projection = total @ (total.T @ exposed) if total.shape[1] else np.zeros_like(exposed)
    return {"total_dimension": int(rank_total), "exposed_dimension": int(rank_exposed),
            "unexposed_quotient_dimension": int(rank_total - rank_exposed),
            "inclusion_residual": float(np.linalg.norm(exposed - projection)),
            "combined_rank": int(matrix_rank(combined, tol)), "tolerance": tol.rank_relative}


def source_reference_audit(source_states: list[MomentState | Array] | tuple[MomentState | Array, ...],
                           response_map: Array, tolerance: GeometryTolerance | float = 1e-9) -> dict[str, object]:
    tol = _tolerance(tolerance)
    first = source_design_span(source_states, 0, tol)
    last = source_design_span(source_states, len(source_states) - 1, tol)
    first_image = subspace_basis(np.asarray(response_map) @ first["contrasts"], tol)
    last_image = subspace_basis(np.asarray(response_map) @ last["contrasts"], tol)
    projection = last_image @ (last_image.T @ first_image) if last_image.shape[1] else np.zeros_like(first_image)
    return {"rank_first": int(first["rank"]), "rank_last": int(last["rank"]),
            "exposed_rank_first": int(first_image.shape[1]), "exposed_rank_last": int(last_image.shape[1]),
            "response_basis_residual": float(np.linalg.norm(first_image - projection)), "invariant": bool(np.allclose(first_image @ first_image.T, last_image @ last_image.T, atol=1e-8))}


__all__ = ["response_operator", "source_design_span", "exposed_response_basis", "target_exposure_fraction", "exact_exposure_membership", "quotient_dimensions", "source_reference_audit"]
