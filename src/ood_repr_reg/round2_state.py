"""Symmetric risk-quotient and ambiguity utilities for round two.

The quotient is prediction-level: it retains only how a quadratic error state
is paired with the declared target moment family.  Symmetric vectorization is
used so duplicated upper/lower-triangular entries do not create artificial
dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _matrix(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.all(np.isfinite(result)) or not np.allclose(result, result.T, atol=1e-10):
        raise ValueError(f"{name} must be finite and symmetric")
    return result


def symmetric_vectorize(matrix: Array) -> Array:
    """Return the isometric upper-triangle vectorization of a symmetric matrix."""
    matrix = _matrix(matrix, "matrix")
    indices = np.triu_indices(matrix.shape[0])
    values = matrix[indices].copy()
    off_diagonal = indices[0] != indices[1]
    values[off_diagonal] *= np.sqrt(2.0)
    return values


def symmetric_devectorize(vector: Array, dimension: int) -> Array:
    """Inverse of :func:`symmetric_vectorize`."""
    vector = np.asarray(vector, dtype=float)
    expected = dimension * (dimension + 1) // 2
    if vector.ndim != 1 or vector.size != expected:
        raise ValueError("vector has incompatible symmetric dimension")
    result = np.zeros((dimension, dimension), dtype=float)
    indices = np.triu_indices(dimension)
    off_diagonal = indices[0] != indices[1]
    values = vector.copy()
    values[off_diagonal] /= np.sqrt(2.0)
    result[indices] = values
    result[(indices[1], indices[0])] = values
    return result


def _stack_vectors(matrices: tuple[Array, ...], name: str) -> tuple[Array, int]:
    if not matrices:
        raise ValueError(f"at least one {name} is required")
    checked = tuple(_matrix(matrix, name) for matrix in matrices)
    dimension = checked[0].shape[0]
    if any(matrix.shape != (dimension, dimension) for matrix in checked):
        raise ValueError(f"all {name}s must have the same dimension")
    return np.stack([symmetric_vectorize(matrix) for matrix in checked], axis=1), dimension


def _basis(matrix: Array, tolerance: float) -> Array:
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0))
    left, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    cutoff = tolerance * (singular[0] if singular.size else 0.0)
    return left[:, singular > cutoff]


@dataclass(frozen=True)
class RiskQuotient:
    dimension: int
    visible_basis: Array
    target_vector_matrix: Array

    @property
    def ambient_dimension(self) -> int:
        return self.dimension * (self.dimension + 1) // 2

    @property
    def annihilator_basis(self) -> Array:
        _, singular, vh = np.linalg.svd(self.target_vector_matrix, full_matrices=True)
        cutoff = 1e-10 * (singular[0] if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > cutoff))
        return vh[rank:].T


def risk_visible_basis(target_moments: tuple[Array, ...], tolerance: float = 1e-10) -> RiskQuotient:
    """Construct the target-risk-visible span in symmetric matrix coordinates."""
    vectors, dimension = _stack_vectors(target_moments, "target moment")
    return RiskQuotient(dimension, _basis(vectors, tolerance), vectors)


def target_annihilator_basis(target_moments: tuple[Array, ...], tolerance: float = 1e-10) -> Array:
    """Return symmetric states orthogonal to every target moment."""
    quotient = risk_visible_basis(target_moments, tolerance)
    return quotient.annihilator_basis


@dataclass(frozen=True)
class SourceObservation:
    matrix: Array
    centered: bool

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    @property
    def kernel_basis(self) -> Array:
        _, singular, vh = np.linalg.svd(self.matrix, full_matrices=True)
        cutoff = 1e-10 * (singular[0] if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > cutoff))
        return vh[rank:].T


def source_observation_map(source_moments: tuple[Array, ...], centered: bool = False) -> SourceObservation:
    """Return absolute or centered source-risk observations."""
    vectors, _ = _stack_vectors(source_moments, "source moment")
    if centered:
        if vectors.shape[1] < 2:
            raise ValueError("centered source observations require two environments")
        vectors = vectors[:, 1:] - vectors[:, [0]]
    return SourceObservation(vectors.T, centered)


@dataclass(frozen=True)
class TargetRecovery:
    coefficients: Array
    residual_norm: float
    exact: bool


def recover_target_moment_from_sources(
    source_moments: tuple[Array, ...], target_moment: Array, tolerance: float = 1e-10
) -> TargetRecovery:
    """Recover a target moment from the source span when possible."""
    source_vectors, _ = _stack_vectors(source_moments, "source moment")
    target_vector = symmetric_vectorize(_matrix(target_moment, "target moment"))
    coefficients, _, _, _ = np.linalg.lstsq(source_vectors, target_vector, rcond=None)
    residual = float(np.linalg.norm(source_vectors @ coefficients - target_vector))
    return TargetRecovery(coefficients, residual, residual <= tolerance)


@dataclass(frozen=True)
class AmbiguityReport:
    target_rank: int
    source_rank_in_quotient: int
    quotient_dimension: int
    ambiguity_dimension: int
    source_kernel_dimension: int
    source_visible_basis: Array


def source_target_ambiguity(
    source_moments: tuple[Array, ...], target_moments: tuple[Array, ...], tolerance: float = 1e-10
) -> AmbiguityReport:
    """Compute source-unidentified directions after removing target annihilators."""
    quotient = risk_visible_basis(target_moments, tolerance)
    source = source_observation_map(source_moments)
    source_vectors, _ = _stack_vectors(source_moments, "source moment")
    visible_source = quotient.visible_basis.T @ source_vectors
    source_basis = _basis(visible_source, tolerance)
    ambiguity = max(quotient.visible_basis.shape[1] - source_basis.shape[1], 0)
    return AmbiguityReport(
        target_rank=quotient.visible_basis.shape[1],
        source_rank_in_quotient=source_basis.shape[1],
        quotient_dimension=quotient.visible_basis.shape[1],
        ambiguity_dimension=ambiguity,
        source_kernel_dimension=source.kernel_basis.shape[1],
        source_visible_basis=source_basis,
    )


def ambiguity_support(delta_state: Array, shift_space: Array, radius: float) -> float:
    """Exact support function for a Frobenius/Hilbert shift ball."""
    if radius < 0:
        raise ValueError("radius must be nonnegative")
    state = np.asarray(delta_state, dtype=float)
    shifts = np.asarray(shift_space, dtype=float)
    if state.ndim != 1 or shifts.ndim != 2 or shifts.shape[1] != state.size:
        raise ValueError("state and shift space have incompatible shapes")
    basis = _basis(shifts.T, 1e-10)
    return float(radius * np.linalg.norm(basis.T @ state))


def quotient_coordinates(matrix: Array, quotient: RiskQuotient) -> Array:
    """Coordinates of a symmetric state in the target-visible quotient."""
    return quotient.visible_basis.T @ symmetric_vectorize(matrix)
