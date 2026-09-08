"""Population calculations for task-preserving intervention robustness.

This module is intentionally separate from the earlier multi-task exploratory
generator.  It implements the partially observed linear Gaussian SCM used by
the intervention-calibration claims:

    C ~ N(0, I), U = L C + xi, Y = beta^T C + epsilon,
    A = R C + mu + eta, X = (U, A).

Only ``R``, ``mu``, and ``Cov(eta)`` vary between environments.  The task
mechanism and the observation channel for ``U`` are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _as_vector(value: Array | list[float], name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _as_matrix(value: Array | list[list[float]], name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return result


def _symmetric(value: Array, name: str) -> Array:
    if not np.allclose(value, value.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(value).min(initial=0.0) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")
    return value


@dataclass(frozen=True)
class LinearGaussianSCM:
    """Shared task and partially observed task-state mechanism."""

    loading: Array
    beta: Array
    sigma_xi: Array
    sigma_y: float

    def __post_init__(self) -> None:
        loading = _as_matrix(self.loading, "loading")
        beta = _as_vector(self.beta, "beta")
        sigma_xi = _symmetric(_as_matrix(self.sigma_xi, "sigma_xi"), "sigma_xi")
        if loading.shape[1] != beta.shape[0]:
            raise ValueError("loading and beta have incompatible task-state dimensions")
        if sigma_xi.shape != (loading.shape[0], loading.shape[0]):
            raise ValueError("sigma_xi has incompatible observation dimensions")
        if self.sigma_y < 0:
            raise ValueError("sigma_y must be nonnegative")
        object.__setattr__(self, "loading", loading)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "sigma_xi", sigma_xi)

    @property
    def c_dim(self) -> int:
        return self.beta.shape[0]

    @property
    def u_dim(self) -> int:
        return self.loading.shape[0]


@dataclass(frozen=True)
class GaussianEnvironment:
    """Environment-specific nuisance mechanism A = R C + mu + eta."""

    relation: Array
    mean: Array
    sigma_a: Array
    name: str = "environment"

    def __post_init__(self) -> None:
        relation = _as_matrix(self.relation, "relation")
        mean = _as_vector(self.mean, "mean")
        sigma_a = _symmetric(_as_matrix(self.sigma_a, "sigma_a"), "sigma_a")
        if relation.shape[0] != mean.shape[0]:
            raise ValueError("relation and mean have incompatible nuisance dimensions")
        if sigma_a.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("sigma_a has incompatible nuisance dimensions")
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma_a", sigma_a)

    @property
    def a_dim(self) -> int:
        return self.mean.shape[0]


@dataclass(frozen=True)
class TransportComponents:
    """Exact sequential target-risk changes from nuisance mechanism changes."""

    relation: float
    mean: float
    covariance: float

    @property
    def total(self) -> float:
        return self.relation + self.mean + self.covariance


@dataclass(frozen=True)
class ProjectedTransportComponents:
    """Exact risk transport split for a pre-specified nuisance projector."""

    controlled: float
    blind: float
    interaction: float

    @property
    def total(self) -> float:
        return self.controlled + self.blind + self.interaction


@dataclass(frozen=True)
class IRMv1RadialResponse:
    """Per-environment radial and tangential risk-gradient diagnostics."""

    derivatives: Array
    radial_gradient_norms: Array
    tangential_gradient_norms: Array
    source_mixture_gradient: Array

    @property
    def penalty(self) -> float:
        return float(np.mean(self.derivatives * self.derivatives))


@dataclass(frozen=True)
class ScalarRelationResponse:
    """Unscaled IRMv1 response q(r)=q0+q1*r+q2*r^2 in the scalar SCM."""

    coefficients: Array
    design: Array
    values: Array
    singular_value: float


def validate_environment(scm: LinearGaussianSCM, environment: GaussianEnvironment) -> None:
    if environment.relation.shape[1] != scm.c_dim:
        raise ValueError("environment relation has incompatible task-state dimensions")


def _slices(scm: LinearGaussianSCM, a_dim: int) -> tuple[slice, slice, slice, slice]:
    start = 1
    c_slice = slice(start, start + scm.c_dim)
    u_slice = slice(c_slice.stop, c_slice.stop + scm.u_dim)
    a_slice = slice(u_slice.stop, u_slice.stop + a_dim)
    return slice(0, 1), c_slice, u_slice, a_slice


def augmented_moment(scm: LinearGaussianSCM, environment: GaussianEnvironment) -> Array:
    """Return E[D D^T] for D = (1, C, xi, A)."""
    validate_environment(scm, environment)
    intercept, c_slice, u_slice, a_slice = _slices(scm, environment.a_dim)
    dimension = a_slice.stop
    mean = np.zeros(dimension)
    mean[intercept] = 1.0
    mean[a_slice] = environment.mean

    covariance = np.zeros((dimension, dimension))
    covariance[c_slice, c_slice] = np.eye(scm.c_dim)
    covariance[u_slice, u_slice] = scm.sigma_xi
    covariance[a_slice, a_slice] = environment.relation @ environment.relation.T + environment.sigma_a
    covariance[c_slice, a_slice] = environment.relation.T
    covariance[a_slice, c_slice] = environment.relation
    return covariance + np.outer(mean, mean)


def feature_map(scm: LinearGaussianSCM, a_dim: int) -> Array:
    """Map D = (1, C, xi, A) to augmented observed feature (1, U, A)."""
    intercept, c_slice, xi_slice, a_slice = _slices(scm, a_dim)
    feature_dimension = 1 + scm.u_dim + a_dim
    mapping = np.zeros((feature_dimension, a_slice.stop))
    mapping[0, intercept] = 1.0
    mapping[1 : 1 + scm.u_dim, c_slice] = scm.loading
    mapping[1 : 1 + scm.u_dim, xi_slice] = np.eye(scm.u_dim)
    mapping[1 + scm.u_dim :, a_slice] = np.eye(a_dim)
    return mapping


def residual_vector(scm: LinearGaussianSCM, environment: GaussianEnvironment, w: Array) -> Array:
    """Return b with Y - f_w(X) = b^T D + epsilon_Y."""
    validate_environment(scm, environment)
    w = _as_vector(w, "w")
    expected = 1 + scm.u_dim + environment.a_dim
    if w.shape != (expected,):
        raise ValueError(f"w must have shape {(expected,)}")
    intercept, c_slice, xi_slice, a_slice = _slices(scm, environment.a_dim)
    w_u = w[1 : 1 + scm.u_dim]
    w_a = w[1 + scm.u_dim :]
    b = np.zeros(a_slice.stop)
    b[intercept] = -w[0]
    b[c_slice] = scm.beta - scm.loading.T @ w_u
    b[xi_slice] = -w_u
    b[a_slice] = -w_a
    return b


def risk(scm: LinearGaussianSCM, environment: GaussianEnvironment, w: Array) -> float:
    b = residual_vector(scm, environment, w)
    return float(scm.sigma_y**2 + b @ augmented_moment(scm, environment) @ b)


def feature_moments(scm: LinearGaussianSCM, environment: GaussianEnvironment) -> tuple[Array, Array]:
    """Return E[x_aug x_aug^T] and E[x_aug Y] for x_aug=(1,U,A)."""
    validate_environment(scm, environment)
    mapping = feature_map(scm, environment.a_dim)
    moment = augmented_moment(scm, environment)
    d_y = np.zeros(moment.shape[0])
    _, c_slice, _, a_slice = _slices(scm, environment.a_dim)
    d_y[c_slice] = scm.beta
    d_y[a_slice] = environment.relation @ scm.beta
    return mapping @ moment @ mapping.T, mapping @ d_y


def source_moments(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], weights: Array | None = None
) -> tuple[Array, Array]:
    if not environments:
        raise ValueError("at least one source environment is required")
    if weights is None:
        weights = np.full(len(environments), 1.0 / len(environments))
    weights = _as_vector(weights, "weights")
    if weights.shape != (len(environments),) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be nonnegative and sum to one")
    moments = [feature_moments(scm, environment) for environment in environments]
    sigma = sum(weight * pair[0] for weight, pair in zip(weights, moments, strict=True))
    cross = sum(weight * pair[1] for weight, pair in zip(weights, moments, strict=True))
    return sigma, cross


def affine_erm(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], weights: Array | None = None
) -> Array:
    """Population affine ERM on the source mixture."""
    sigma, cross = source_moments(scm, environments, weights)
    return np.linalg.solve(sigma, cross)


def ridge_solution(
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    penalty: float,
    weights: Array | None = None,
) -> Array:
    """Population ridge solution with an unpenalized intercept."""
    if penalty < 0:
        raise ValueError("penalty must be nonnegative")
    sigma, cross = source_moments(scm, environments, weights)
    regularizer = np.eye(sigma.shape[0])
    regularizer[0, 0] = 0.0
    return np.linalg.solve(sigma + penalty * regularizer, cross)


def source_risk(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array, weights: Array | None = None
) -> float:
    if weights is None:
        weights = np.full(len(environments), 1.0 / len(environments))
    return float(sum(weight * risk(scm, environment, w) for weight, environment in zip(weights, environments, strict=True)))


def observational_oracle_risk(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], weights: Array | None = None
) -> float:
    return source_risk(scm, environments, affine_erm(scm, environments, weights), weights)


def causal_oracle_risk(scm: LinearGaussianSCM) -> float:
    return float(scm.sigma_y**2)


def l2_penalty(w: Array) -> float:
    w = _as_vector(w, "w")
    return float(w[1:] @ w[1:])


def risk_gradient(scm: LinearGaussianSCM, environment: GaussianEnvironment, w: Array) -> Array:
    sigma, cross = feature_moments(scm, environment)
    return 2.0 * (sigma @ w - cross)


def source_mixture_gradient(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array, weights: Array | None = None
) -> Array:
    """Return the gradient of population source-mixture risk at ``w``."""
    sigma, cross = source_moments(scm, environments, weights)
    return 2.0 * (sigma @ w - cross)


def erm_stationarity_residual(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array, weights: Array | None = None
) -> float:
    """Norm of the only first-order condition imposed by source-mixture ERM."""
    return float(np.linalg.norm(source_mixture_gradient(scm, environments, w, weights)))


def irmv1_radial_response(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array, weights: Array | None = None
) -> IRMv1RadialResponse:
    """Decompose per-environment gradients into predictor-ray and tangent parts.

    Standard scalar-scale IRMv1 penalizes only ``w.T @ grad R_e(w)``.  It
    therefore controls the radial component in this function, not the
    tangential component or the full environment-gradient discrepancy.
    """
    w = _as_vector(w, "w")
    gradients = np.stack([risk_gradient(scm, environment, w) for environment in environments])
    derivatives = gradients @ w
    norm_w = float(np.linalg.norm(w))
    if np.isclose(norm_w, 0.0):
        radial = np.zeros(len(environments))
    else:
        radial = np.abs(derivatives) / norm_w
    gradient_norms = np.linalg.norm(gradients, axis=1)
    tangential = np.sqrt(np.maximum(gradient_norms * gradient_norms - radial * radial, 0.0))
    return IRMv1RadialResponse(
        derivatives=derivatives,
        radial_gradient_norms=radial,
        tangential_gradient_norms=tangential,
        source_mixture_gradient=source_mixture_gradient(scm, environments, w, weights),
    )


def gradient_alignment_penalty(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array
) -> float:
    gradients = np.stack([risk_gradient(scm, environment, w) for environment in environments])
    centered = gradients - gradients.mean(axis=0, keepdims=True)
    return float(np.sum(centered * centered))


def irmv1_scale_penalty(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array
) -> float:
    """The standard dummy-scalar penalty d/d alpha R_e(alpha f_w)|_{alpha=1}."""
    return irmv1_radial_response(scm, environments, w).penalty


def gradient_response_operator(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...]
) -> tuple[Array, Array]:
    """Return H,d with centered gradient variation 2(H w - d)."""
    if len(environments) < 2:
        raise ValueError("at least two environments are required")
    pairs = [feature_moments(scm, environment) for environment in environments]
    mean_sigma = sum(pair[0] for pair in pairs) / len(pairs)
    mean_cross = sum(pair[1] for pair in pairs) / len(pairs)
    return (
        np.concatenate([pair[0] - mean_sigma for pair in pairs], axis=0),
        np.concatenate([pair[1] - mean_cross for pair in pairs], axis=0),
    )


def _smallest_singular_value(matrix: Array) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.svd(matrix, compute_uv=False).min())


def _column_space_basis(matrix: Array) -> Array:
    """Return an orthonormal basis for the numerical column space of ``matrix``."""
    if matrix.size == 0:
        return np.empty((matrix.shape[0], 0))
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((matrix.shape[0], 0))
    tolerance = np.finfo(float).eps * max(matrix.shape) * singular_values[0]
    return left[:, singular_values > tolerance]


def nuisance_observability(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...]
) -> tuple[float, float]:
    """Return direct and task-adjusted nuisance observability for gradient variation.

    The adjusted value projects nuisance response off the intercept/U response,
    which prevents a nuisance direction that can be cancelled by task-observation
    parameters from being incorrectly called observable.
    """
    h, _ = gradient_response_operator(scm, environments)
    base = h[:, : 1 + scm.u_dim]
    nuisance = h[:, 1 + scm.u_dim :]
    direct = _smallest_singular_value(nuisance)
    if base.size == 0:
        return direct, direct
    basis = _column_space_basis(base)
    residual = nuisance - basis @ (basis.T @ nuisance)
    return direct, _smallest_singular_value(residual)


def risk_transport_components(
    scm: LinearGaussianSCM, source: GaussianEnvironment, target: GaussianEnvironment, w: Array
) -> TransportComponents:
    """Exactly telescope target transport into relation, mean, and covariance shifts.

    The order is fixed: first replace the conditional relation, then nuisance
    mean, then nuisance covariance.  It is an accounting convention, not a
    claim that the three mechanisms are independent.
    """
    validate_environment(scm, source)
    validate_environment(scm, target)
    if source.a_dim != target.a_dim:
        raise ValueError("source and target nuisance dimensions must agree")
    relation_hybrid = GaussianEnvironment(
        target.relation, source.mean, source.sigma_a, "relation_hybrid"
    )
    mean_hybrid = GaussianEnvironment(target.relation, target.mean, source.sigma_a, "mean_hybrid")
    source_value = risk(scm, source, w)
    relation_value = risk(scm, relation_hybrid, w)
    mean_value = risk(scm, mean_hybrid, w)
    target_value = risk(scm, target, w)
    return TransportComponents(
        relation=relation_value - source_value,
        mean=mean_value - relation_value,
        covariance=target_value - mean_value,
    )


def projected_transport_components(
    scm: LinearGaussianSCM,
    source: GaussianEnvironment,
    target: GaussianEnvironment,
    w: Array,
    projector: Array,
) -> ProjectedTransportComponents:
    """Split exact transport by a pre-specified orthogonal nuisance projector.

    This function deliberately does not infer ``projector`` from target data or
    a regularizer.  A method-specific theorem must supply that bridge.
    """
    validate_environment(scm, source)
    validate_environment(scm, target)
    if source.a_dim != target.a_dim:
        raise ValueError("source and target nuisance dimensions must agree")
    projector = _as_matrix(projector, "projector")
    if projector.shape != (source.a_dim, source.a_dim):
        raise ValueError("projector has incompatible nuisance dimensions")
    if not np.allclose(projector, projector.T, atol=1e-10) or not np.allclose(
        projector @ projector, projector, atol=1e-10
    ):
        raise ValueError("projector must be orthogonal")

    w = _as_vector(w, "w")
    _, _, _, a_slice = _slices(scm, source.a_dim)
    base_w = w.copy()
    base_w[1 + scm.u_dim :] = 0.0
    base = residual_vector(scm, source, base_w)
    # In observed-feature coordinates w_A starts after intercept and U.
    nuisance = w[1 + scm.u_dim :]
    controlled = np.zeros_like(base)
    blind = np.zeros_like(base)
    controlled[a_slice] = -(projector @ nuisance)
    blind[a_slice] = -((np.eye(source.a_dim) - projector) @ nuisance)
    delta_moment = augmented_moment(scm, target) - augmented_moment(scm, source)
    controlled_value = float(2.0 * base @ delta_moment @ controlled + controlled @ delta_moment @ controlled)
    blind_value = float(2.0 * base @ delta_moment @ blind + blind @ delta_moment @ blind)
    interaction = float(2.0 * controlled @ delta_moment @ blind)
    return ProjectedTransportComponents(controlled_value, blind_value, interaction)


def nuisance_transport_bound(
    scm: LinearGaussianSCM,
    source: GaussianEnvironment,
    target: GaussianEnvironment,
    w: Array,
    nuisance_norm_bound: float,
) -> float:
    """Bound absolute target transport from a bound on effective nuisance use.

    This is a conditional population bound for the C001 model.  It requires
    that only the nuisance mechanism changes, as encoded by the common SCM.
    The caller must independently justify ``nuisance_norm_bound``; scalar
    IRMv1 with a full-rank relation design supplies one such bridge.
    """
    if nuisance_norm_bound < 0:
        raise ValueError("nuisance_norm_bound must be nonnegative")
    validate_environment(scm, source)
    validate_environment(scm, target)
    if source.a_dim != target.a_dim:
        raise ValueError("source and target nuisance dimensions must agree")
    w = _as_vector(w, "w")
    base_w = w.copy()
    base_w[1 + scm.u_dim :] = 0.0
    base = residual_vector(scm, source, base_w)
    delta_moment = augmented_moment(scm, target) - augmented_moment(scm, source)
    geometry = float(np.linalg.norm(delta_moment, ord=2))
    return geometry * (2.0 * np.linalg.norm(base) * nuisance_norm_bound + nuisance_norm_bound**2)


def scalar_relation_design(relations: Array | list[float]) -> Array:
    """Quadratic response design with rows ``(1, r, r^2)``."""
    relations = _as_vector(relations, "relations")
    if relations.size < 1:
        raise ValueError("at least one relation is required")
    return np.column_stack((np.ones(relations.size), relations, relations * relations))


def scalar_relation_response(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array
) -> ScalarRelationResponse:
    """Return the scalar IRMv1 response polynomial under the stated source model.

    Preconditions are one-dimensional task/observation/nuisance variables,
    zero nuisance means, a common nuisance variance, and zero predictor
    intercept.  The unscaled response is

        q(r) = E[(f_w(X)-Y) f_w(X)] = q0 + q1 r + q2 r^2,

    with ``q2 = w_A^2``.  The standard IRMv1 derivative is ``2*q(r)``.
    """
    if scm.c_dim != 1 or scm.u_dim != 1:
        raise ValueError("scalar relation response requires one-dimensional C and U")
    if not environments:
        raise ValueError("at least one environment is required")
    if any(environment.a_dim != 1 for environment in environments):
        raise ValueError("scalar relation response requires one-dimensional A")
    if any(not np.allclose(environment.mean, 0.0) for environment in environments):
        raise ValueError("scalar relation response requires zero nuisance means")
    variance = environments[0].sigma_a
    if any(not np.allclose(environment.sigma_a, variance) for environment in environments[1:]):
        raise ValueError("scalar relation response requires common nuisance variance")
    w = _as_vector(w, "w")
    if w.shape != (3,) or not np.isclose(w[0], 0.0):
        raise ValueError("scalar relation response requires a zero-intercept affine predictor")

    loading = float(scm.loading[0, 0])
    beta = float(scm.beta[0])
    observation_variance = float(scm.sigma_xi[0, 0])
    nuisance_variance = float(variance[0, 0])
    u, a = float(w[1]), float(w[2])
    coefficients = np.array(
        [
            loading * loading * u * u - loading * beta * u + observation_variance * u * u + a * a * nuisance_variance,
            a * (2.0 * loading * u - beta),
            a * a,
        ]
    )
    relations = np.array([float(environment.relation[0, 0]) for environment in environments])
    design = scalar_relation_design(relations)
    values = design @ coefficients
    return ScalarRelationResponse(
        coefficients=coefficients,
        design=design,
        values=values,
        singular_value=_smallest_singular_value(design),
    )


def scalar_irmv1_nuisance_squared_bound(
    scm: LinearGaussianSCM, environments: tuple[GaussianEnvironment, ...], w: Array
) -> float:
    """Bound ``w_A^2`` from scalar-scale IRMv1 under a full-rank relation design.

    For m source environments and V=[1,r,r^2], the exact relation response
    has q2=w_A^2 and Omega=4/m ||V q||^2.  Hence

        w_A^2 <= sqrt(m * Omega) / (2 * sigma_min(V)).
    """
    response = scalar_relation_response(scm, environments, w)
    if response.design.shape[0] < 3 or np.isclose(response.singular_value, 0.0):
        raise ValueError("a full-rank three-column relation design is required")
    penalty = irmv1_scale_penalty(scm, environments, w)
    return float(np.sqrt(len(environments) * penalty) / (2.0 * response.singular_value))


def effective_predictor(representation: Array, head: Array) -> Array:
    """Return theta=B.T w for z=B x and f=w.T z."""
    representation = _as_matrix(representation, "representation")
    head = _as_vector(head, "head")
    if representation.shape[0] != head.shape[0]:
        raise ValueError("representation and head have incompatible latent dimensions")
    return representation.T @ head


def representation_risk(
    scm: LinearGaussianSCM, environment: GaussianEnvironment, representation: Array, head: Array
) -> float:
    """Evaluate risk through the effective predictor, never encoder coordinates."""
    return risk(scm, environment, effective_predictor(representation, head))


def representation_irmv1_scale_penalty(
    scm: LinearGaussianSCM,
    environments: tuple[GaussianEnvironment, ...],
    representation: Array,
    head: Array,
) -> float:
    """Evaluate standard IRMv1 through the effective predictor theta=B.T w."""
    return irmv1_scale_penalty(scm, environments, effective_predictor(representation, head))


def reparameterize_linear_representation(
    representation: Array, head: Array, transform: Array
) -> tuple[Array, Array]:
    """Apply B -> T B and w -> T^{-T} w without changing the predictor."""
    representation = _as_matrix(representation, "representation")
    head = _as_vector(head, "head")
    transform = _as_matrix(transform, "transform")
    if transform.shape != (head.shape[0], head.shape[0]) or representation.shape[0] != head.shape[0]:
        raise ValueError("transform, representation, and head have incompatible latent dimensions")
    if np.isclose(np.linalg.det(transform), 0.0):
        raise ValueError("transform must be invertible")
    return transform @ representation, np.linalg.solve(transform.T, head)


def effective_nuisance_sensitivity(representation: Array, head: Array, nuisance_start: int) -> Array:
    """Return the coordinate-invariant effective nuisance coefficient w.T B_A."""
    theta = effective_predictor(representation, head)
    if nuisance_start < 0 or nuisance_start > theta.size:
        raise ValueError("nuisance_start is outside the observed-feature dimension")
    return theta[nuisance_start:]


def correlation_robust_risk(
    scm: LinearGaussianSCM,
    w: Array,
    relation_center: Array,
    relation_radius: float,
    sigma_a: Array,
) -> float:
    """Exact worst-case risk over ||R-R0||_F <= radius with zero nuisance mean."""
    if relation_radius < 0:
        raise ValueError("relation_radius must be nonnegative")
    environment = GaussianEnvironment(
        relation=relation_center,
        mean=np.zeros(relation_center.shape[0]),
        sigma_a=sigma_a,
    )
    b = residual_vector(scm, environment, w)
    _, c_slice, xi_slice, a_slice = _slices(scm, environment.a_dim)
    b0 = float(b[0])
    task_residual = b[c_slice] + relation_center.T @ b[a_slice]
    return float(
        scm.sigma_y**2
        + b0**2
        + b[xi_slice] @ scm.sigma_xi @ b[xi_slice]
        + b[a_slice] @ sigma_a @ b[a_slice]
        + (np.linalg.norm(task_residual) + relation_radius * np.linalg.norm(b[a_slice])) ** 2
    )


def moment_robust_risk(
    scm: LinearGaussianSCM,
    w: Array,
    relation: Array,
    mean_center: Array,
    mean_radius: float,
    sigma_a_center: Array,
    covariance_radius: float,
) -> float:
    """Exact worst-case risk over bounded nuisance mean/covariance perturbations."""
    if mean_radius < 0 or covariance_radius < 0:
        raise ValueError("moment radii must be nonnegative")
    environment = GaussianEnvironment(relation, mean_center, sigma_a_center)
    b = residual_vector(scm, environment, w)
    _, c_slice, xi_slice, a_slice = _slices(scm, environment.a_dim)
    b_a = b[a_slice]
    task_residual = b[c_slice] + relation.T @ b_a
    mean_residual = float(b[0] + b_a @ mean_center)
    return float(
        scm.sigma_y**2
        + (abs(mean_residual) + mean_radius * np.linalg.norm(b_a)) ** 2
        + b[xi_slice] @ scm.sigma_xi @ b[xi_slice]
        + np.linalg.norm(task_residual) ** 2
        + b_a @ sigma_a_center @ b_a
        + covariance_radius * (b_a @ b_a)
    )


def scalar_irmv1_zero_candidates(
    scm: LinearGaussianSCM,
    sources: tuple[GaussianEnvironment, GaussianEnvironment],
) -> tuple[Array, ...]:
    """Enumerate zeroes of scalar, zero-mean IRMv1 population equations.

    The function is deliberately limited to the smallest model used in the
    proof audit: one task state, one task observation, one nuisance, and two
    zero-mean sources with the same nuisance noise variance.  It returns both
    nuisance-free zeroes and every real nonzero-nuisance branch.
    """
    if scm.c_dim != 1 or scm.u_dim != 1:
        raise ValueError("the scalar zero-set calculation requires one-dimensional C and U")
    first, second = sources
    if first.a_dim != 1 or second.a_dim != 1:
        raise ValueError("the scalar zero-set calculation requires one-dimensional A")
    if not np.allclose(first.mean, 0.0) or not np.allclose(second.mean, 0.0):
        raise ValueError("the scalar zero-set calculation requires zero nuisance means")
    if not np.allclose(first.sigma_a, second.sigma_a):
        raise ValueError("the scalar zero-set calculation requires common nuisance variance")
    r1 = float(first.relation[0, 0])
    r2 = float(second.relation[0, 0])
    if np.isclose(r1, r2):
        raise ValueError("the source relations must differ")
    loading = float(scm.loading[0, 0])
    beta = float(scm.beta[0])
    if np.isclose(loading, 0.0):
        raise ValueError("the task observation loading must be nonzero")
    observation_variance = loading**2 + float(scm.sigma_xi[0, 0])

    candidates = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, loading * beta / observation_variance, 0.0]),
    ]
    relation_sum = r1 + r2

    def first_environment_equation(a: float) -> float:
        u = (beta - a * relation_sum) / (2.0 * loading)
        w = np.array([0.0, u, a])
        sigma, cross = feature_moments(scm, first)
        return float(w @ sigma @ w - w @ cross)

    coefficients = np.polyfit(np.array([0.0, 1.0, 2.0]), [
        first_environment_equation(0.0),
        first_environment_equation(1.0),
        first_environment_equation(2.0),
    ], 2)
    for root in np.roots(coefficients):
        if abs(root.imag) > 1e-9 or abs(root.real) < 1e-9:
            continue
        a = float(root.real)
        u = (beta - a * relation_sum) / (2.0 * loading)
        candidate = np.array([0.0, u, a])
        if (
            irmv1_scale_penalty(scm, sources, candidate) <= 1e-10
            and not any(np.allclose(candidate, prior, atol=1e-8) for prior in candidates)
        ):
            candidates.append(candidate)
    return tuple(candidates)
