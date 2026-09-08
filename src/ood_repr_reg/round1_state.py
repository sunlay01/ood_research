"""Risk-state and source-exposure utilities for the first research round.

The objects in this module are deliberately prediction-level.  They do not
assign meanings to hidden coordinates.  In the linear Gaussian model, the
quadratic error state ``Q = vv.T`` is paired with environment moments ``M``;
the only directions that matter for a declared environment family are the
directions visible through these pairings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    augmented_moment,
    residual_vector,
)

Array = np.ndarray


def _array(value: Array | list[float] | list[list[float]], name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _matrix(value: Array, name: str) -> Array:
    result = _array(value, name)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.allclose(result, result.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return result


def _flatten(matrix: Array) -> Array:
    return np.asarray(matrix, dtype=float).reshape(-1)


@dataclass(frozen=True)
class RiskState:
    """Finite-dimensional prediction state for population squared loss."""

    vector: Array
    quadratic: Array
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        vector = _array(self.vector, "vector")
        quadratic = _matrix(self.quadratic, "quadratic")
        if vector.ndim != 1 or quadratic.shape != (vector.size, vector.size):
            raise ValueError("vector and quadratic state have incompatible shapes")
        if not np.allclose(quadratic, np.outer(vector, vector), atol=1e-10):
            raise ValueError("quadratic must equal vector outer product")
        if len(self.labels) != vector.size:
            raise ValueError("labels must match vector dimension")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "quadratic", quadratic)


@dataclass(frozen=True)
class ObservationOperator:
    """Linear source observation map on flattened symmetric states."""

    matrix: Array
    labels: tuple[str, ...]
    state_dimension: int

    def __post_init__(self) -> None:
        matrix = _array(self.matrix, "matrix")
        if matrix.ndim != 2 or matrix.shape[1] != self.state_dimension**2:
            raise ValueError("observation matrix has incompatible state dimension")
        if len(self.labels) != matrix.shape[0]:
            raise ValueError("labels must match observation rows")
        object.__setattr__(self, "matrix", matrix)

    @property
    def rank(self) -> int:
        return int(np.linalg.matrix_rank(self.matrix))

    @property
    def singular_values(self) -> Array:
        return np.linalg.svd(self.matrix, compute_uv=False)

    @property
    def kernel_basis(self) -> Array:
        """Orthonormal basis for states invisible to all source rows."""
        _, singular, vh = np.linalg.svd(self.matrix, full_matrices=True)
        tolerance = np.finfo(float).eps * max(self.matrix.shape) * (singular[0] if singular.size else 0.0)
        rank = int(np.count_nonzero(singular > tolerance))
        return vh[rank:].T


@dataclass(frozen=True)
class ExposureReport:
    """Exposure and target alignment diagnostics for a source operator."""

    source_rank: int
    source_dimension: int
    target_alignment: Array
    target_blind_norm: Array
    source_kernel_dimension: int


@dataclass(frozen=True)
class TargetRecovery:
    """Least-squares recovery of a target moment from source moments."""

    coefficients: Array
    residual_norm: float
    exact: bool


def risk_state(
    scm: LinearGaussianSCM,
    w: Array,
    environment: GaussianEnvironment,
) -> RiskState:
    """Build ``Q_f`` from the exact residual state of an affine predictor."""
    vector = residual_vector(scm, environment, w)
    labels = ("intercept", "task_state", "observation_noise", "nuisance")
    expanded = (labels[0],) + tuple(f"C_{i}" for i in range(scm.c_dim))
    expanded += tuple(f"xi_{i}" for i in range(scm.u_dim))
    expanded += tuple(f"A_{i}" for i in range(environment.a_dim))
    return RiskState(vector, np.outer(vector, vector), expanded)


def environment_state(scm: LinearGaussianSCM, environment: GaussianEnvironment) -> Array:
    """Return the augmented environment moment ``M_e = E[D D.T]``."""
    return augmented_moment(scm, environment)


def risk_pairing(state: RiskState | Array, environment_moment: Array) -> float:
    """Return the quadratic excess-risk pairing ``<Q, M>``."""
    quadratic = state.quadratic if isinstance(state, RiskState) else _matrix(_array(state, "state"), "state")
    moment = _matrix(environment_moment, "environment_moment")
    if moment.shape != quadratic.shape:
        raise ValueError("state and environment moment have incompatible shapes")
    return float(np.sum(quadratic * moment))


def conditional_loss_risk(loss_values: Array, environment_weights: Array) -> float:
    """Integrate a shared conditional loss over an environment law."""
    loss_values = _array(loss_values, "loss_values")
    environment_weights = _array(environment_weights, "environment_weights")
    if loss_values.ndim != 1 or environment_weights.shape != loss_values.shape:
        raise ValueError("loss values and environment weights must be matching vectors")
    if np.any(environment_weights < 0) or not np.isclose(environment_weights.sum(), 1.0):
        raise ValueError("environment weights must be nonnegative and sum to one")
    return float(loss_values @ environment_weights)


def risk_quotient_basis(environment_moments: tuple[Array, ...], tolerance: float = 1e-10) -> Array:
    """Return an orthonormal basis of the risk-visible moment span.

    Columns live in the flattened matrix space.  Projection onto this basis is
    the canonical quotient for a declared linear moment environment family;
    it is not a semantic latent-space decomposition.
    """
    if not environment_moments:
        raise ValueError("at least one environment moment is required")
    matrices = [_matrix(moment, "environment_moment") for moment in environment_moments]
    dimension = matrices[0].shape[0]
    if any(moment.shape != (dimension, dimension) for moment in matrices):
        raise ValueError("all environment moments must have the same shape")
    stacked = np.stack([_flatten(moment) for moment in matrices], axis=1)
    left, singular, _ = np.linalg.svd(stacked, full_matrices=False)
    cutoff = tolerance * (singular[0] if singular.size else 0.0)
    return left[:, singular > cutoff]


def project_quadratic_state(quadratic: Array, basis: Array) -> Array:
    """Project a quadratic state onto the declared risk-visible span."""
    quadratic = _matrix(quadratic, "quadratic")
    basis = _array(basis, "basis")
    if basis.ndim != 2 or basis.shape[0] != quadratic.size:
        raise ValueError("basis has incompatible shape")
    projected = basis @ (basis.T @ _flatten(quadratic))
    return projected.reshape(quadratic.shape)


def source_observation_operator(
    environment_moments: tuple[Array, ...],
    extra_rows: Array | None = None,
    labels: tuple[str, ...] | None = None,
) -> ObservationOperator:
    """Create the source map ``Q -> (<Q,M_e>)_e`` plus optional rows."""
    if not environment_moments:
        raise ValueError("at least one source moment is required")
    moments = [_matrix(moment, "environment_moment") for moment in environment_moments]
    dimension = moments[0].shape[0]
    if any(moment.shape != (dimension, dimension) for moment in moments):
        raise ValueError("all source moments must have the same shape")
    rows = [_flatten(moment) for moment in moments]
    row_labels = list(labels or [f"source_risk_{i}" for i in range(len(rows))])
    if extra_rows is not None:
        extra_rows = _array(extra_rows, "extra_rows")
        if extra_rows.ndim != 2 or extra_rows.shape[1] != dimension**2:
            raise ValueError("extra rows have incompatible shape")
        rows.extend(extra_rows)
        row_labels.extend(f"extra_{i}" for i in range(extra_rows.shape[0]))
    return ObservationOperator(np.stack(rows), tuple(row_labels), dimension)


def exposure_identifiability(
    operator: ObservationOperator,
    target_moments: tuple[Array, ...],
) -> ExposureReport:
    """Measure which target moment directions are exposed by source rows."""
    if not target_moments:
        raise ValueError("at least one target moment is required")
    target_vectors = np.stack([_flatten(_matrix(moment, "target_moment")) for moment in target_moments])
    row_basis = np.linalg.svd(operator.matrix.T, full_matrices=False)[0][:, : operator.rank]
    if operator.rank == 0:
        aligned = np.zeros(len(target_moments))
        blind = np.linalg.norm(target_vectors, axis=1)
    else:
        projections = target_vectors @ row_basis
        target_norms = np.linalg.norm(target_vectors, axis=1)
        aligned = np.divide(
            np.linalg.norm(projections, axis=1), target_norms, out=np.zeros_like(target_norms), where=target_norms > 0
        )
        blind = np.linalg.norm(target_vectors - projections @ row_basis.T, axis=1)
    return ExposureReport(
        source_rank=operator.rank,
        source_dimension=operator.matrix.shape[1],
        target_alignment=aligned,
        target_blind_norm=blind,
        source_kernel_dimension=operator.kernel_basis.shape[1],
    )


def recover_target_moment_from_sources(
    source_moments: tuple[Array, ...],
    target_moment: Array,
    tolerance: float = 1e-10,
) -> TargetRecovery:
    """Test whether a target risk functional lies in the source span."""
    if not source_moments:
        raise ValueError("at least one source moment is required")
    source_vectors = np.stack([_flatten(_matrix(moment, "source_moment")) for moment in source_moments], axis=1)
    target_vector = _flatten(_matrix(target_moment, "target_moment"))
    coefficients, _, _, _ = np.linalg.lstsq(source_vectors, target_vector, rcond=None)
    residual = float(np.linalg.norm(source_vectors @ coefficients - target_vector))
    return TargetRecovery(coefficients, residual, residual <= tolerance)


def risk_equivalence_gap(
    quadratic_difference: Array,
    environment_moments: tuple[Array, ...],
) -> float:
    """Return the largest risk-pairing difference over a declared family."""
    difference = _matrix(quadratic_difference, "quadratic_difference")
    return float(max(abs(np.sum(difference * _matrix(moment, "environment_moment"))) for moment in environment_moments))


def component_transport(
    vector: Array,
    delta_moment: Array,
    projectors: tuple[Array, ...],
) -> tuple[Array, Array]:
    """Exact diagonal and pairwise interaction transport for vector components."""
    vector = _array(vector, "vector")
    delta_moment = _matrix(delta_moment, "delta_moment")
    if delta_moment.shape != (vector.size, vector.size):
        raise ValueError("delta moment and vector have incompatible shapes")
    if not projectors:
        raise ValueError("at least one projector is required")
    components = []
    for projector in projectors:
        projector = _matrix(projector, "projector")
        if projector.shape != delta_moment.shape or not np.allclose(projector @ projector, projector, atol=1e-8):
            raise ValueError("projectors must be square idempotent matrices")
        components.append(projector @ vector)
    diagonal = np.array([component @ delta_moment @ component for component in components])
    interaction = np.array(
        [
            2.0 * components[i] @ delta_moment @ components[j]
            for i in range(len(components))
            for j in range(i + 1, len(components))
        ]
    )
    return diagonal, interaction


def componentwise_bound(vector: Array, delta_moment: Array, projectors: tuple[Array, ...]) -> float:
    """Spectral-norm bound matching ``component_transport`` terms."""
    vector = _array(vector, "vector")
    delta_moment = _matrix(delta_moment, "delta_moment")
    norm_delta = float(np.linalg.norm(delta_moment, ord=2))
    norms = [float(np.linalg.norm(projector @ vector)) for projector in projectors]
    return float(norm_delta * (sum(norm_value * norm_value for norm_value in norms) + 2.0 * sum(
        norms[i] * norms[j] for i in range(len(norms)) for j in range(i + 1, len(norms))
    )))


def source_environment_moments(
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
) -> tuple[Array, ...]:
    """Convenience wrapper used by the round-one runner and tests."""
    return tuple(environment_state(scm, environment) for environment in environments)
