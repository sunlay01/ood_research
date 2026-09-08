"""Structural mechanism family for round three.

The module keeps structural parameters, induced moments, and prediction risk
separate.  It is deliberately population-level and uses U=(C,A) with an
affine-free predictor; non-zero means are still represented through raw second
moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _vector(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result


def _matrix(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(result, result.T, atol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return result


@dataclass(frozen=True)
class StructuralMechanism:
    """Parameters for C ~ N(mu_c, sigma_c), A = gamma C + b + xi."""

    mu_c: Array
    sigma_c: Array
    gamma: Array
    b: Array
    mu_xi: Array
    sigma_xi: Array

    def __post_init__(self) -> None:
        mu_c = _vector(self.mu_c, "mu_c")
        sigma_c = _matrix(self.sigma_c, "sigma_c")
        gamma = np.asarray(self.gamma, dtype=float)
        b = _vector(self.b, "b")
        mu_xi = _vector(self.mu_xi, "mu_xi")
        sigma_xi = _matrix(self.sigma_xi, "sigma_xi")
        if sigma_c.shape != (mu_c.size, mu_c.size):
            raise ValueError("sigma_c has incompatible shape")
        if gamma.ndim != 2 or gamma.shape[1] != mu_c.size:
            raise ValueError("gamma has incompatible shape")
        if b.size != gamma.shape[0] or mu_xi.size != gamma.shape[0]:
            raise ValueError("nuisance vectors have incompatible shape")
        if sigma_xi.shape != (gamma.shape[0], gamma.shape[0]):
            raise ValueError("sigma_xi has incompatible shape")
        object.__setattr__(self, "mu_c", mu_c)
        object.__setattr__(self, "sigma_c", sigma_c)
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "mu_xi", mu_xi)
        object.__setattr__(self, "sigma_xi", sigma_xi)

    @property
    def c_dim(self) -> int:
        return self.mu_c.size

    @property
    def a_dim(self) -> int:
        return self.b.size

    @property
    def mu_a(self) -> Array:
        return self.gamma @ self.mu_c + self.b + self.mu_xi


@dataclass(frozen=True)
class MechanismDirection:
    """A tangent direction in structural parameter space."""

    d_mu_c: Array
    d_sigma_c: Array
    d_gamma: Array
    d_b: Array
    d_mu_xi: Array
    d_sigma_xi: Array
    name: str = "direction"


@dataclass(frozen=True)
class MomentCoordinates:
    """Raw first/second moments induced by a structural mechanism."""

    mu_c: Array
    mu_a: Array
    sigma_cc: Array
    sigma_ca: Array
    sigma_aa: Array

    def joint_second_moment(self) -> Array:
        dimension = self.mu_c.size + self.mu_a.size
        result = np.zeros((dimension, dimension), dtype=float)
        c = slice(0, self.mu_c.size)
        a = slice(self.mu_c.size, dimension)
        result[c, c] = self.sigma_cc + np.outer(self.mu_c, self.mu_c)
        result[c, a] = self.sigma_ca + np.outer(self.mu_c, self.mu_a)
        result[a, c] = result[c, a].T
        result[a, a] = self.sigma_aa + np.outer(self.mu_a, self.mu_a)
        return result


def moment_coordinates(mechanism: StructuralMechanism) -> MomentCoordinates:
    sigma_ca = mechanism.sigma_c @ mechanism.gamma.T
    sigma_aa = mechanism.gamma @ mechanism.sigma_c @ mechanism.gamma.T + mechanism.sigma_xi
    return MomentCoordinates(
        mechanism.mu_c.copy(),
        mechanism.mu_a,
        mechanism.sigma_c.copy(),
        sigma_ca,
        sigma_aa,
    )


def moment_matrix(mechanism: StructuralMechanism) -> Array:
    return moment_coordinates(mechanism).joint_second_moment()


def add_direction(mechanism: StructuralMechanism, direction: MechanismDirection, scale: float) -> StructuralMechanism:
    """Apply a structural tangent and return a new mechanism point."""
    return StructuralMechanism(
        mechanism.mu_c + scale * direction.d_mu_c,
        mechanism.sigma_c + scale * direction.d_sigma_c,
        mechanism.gamma + scale * direction.d_gamma,
        mechanism.b + scale * direction.d_b,
        mechanism.mu_xi + scale * direction.d_mu_xi,
        mechanism.sigma_xi + scale * direction.d_sigma_xi,
    )


def _directional_moments(mechanism: StructuralMechanism, direction: MechanismDirection) -> MomentCoordinates:
    """Analytic derivative of the induced raw moment coordinates."""
    mu_c, sigma, gamma = mechanism.mu_c, mechanism.sigma_c, mechanism.gamma
    mu_a = mechanism.mu_a
    dmu_c, dsigma, dgamma = direction.d_mu_c, direction.d_sigma_c, direction.d_gamma
    dmu_a = dgamma @ mu_c + gamma @ dmu_c + direction.d_b + direction.d_mu_xi
    d_sigma_ca = dsigma @ gamma.T + sigma @ dgamma.T
    d_sigma_aa = (
        dgamma @ sigma @ gamma.T
        + gamma @ dsigma @ gamma.T
        + gamma @ sigma @ dgamma.T
        + direction.d_sigma_xi
    )
    return MomentCoordinates(dmu_c, dmu_a, dsigma, d_sigma_ca, d_sigma_aa)


def directional_moment_matrix(mechanism: StructuralMechanism, direction: MechanismDirection) -> Array:
    """Analytic derivative D M[direction] of the raw joint second moment."""
    base = moment_coordinates(mechanism)
    derivative = _directional_moments(mechanism, direction)
    c = slice(0, mechanism.c_dim)
    a = slice(mechanism.c_dim, mechanism.c_dim + mechanism.a_dim)
    result = np.zeros((mechanism.c_dim + mechanism.a_dim,) * 2, dtype=float)
    result[c, c] = derivative.sigma_cc + np.outer(derivative.mu_c, base.mu_c) + np.outer(base.mu_c, derivative.mu_c)
    result[c, a] = derivative.sigma_ca + np.outer(derivative.mu_c, base.mu_a) + np.outer(base.mu_c, derivative.mu_a)
    result[a, c] = result[c, a].T
    result[a, a] = derivative.sigma_aa + np.outer(derivative.mu_a, base.mu_a) + np.outer(base.mu_a, derivative.mu_a)
    return result


def risk_state(beta: Array, w_c: Array, w_a: Array) -> Array:
    beta = _vector(beta, "beta")
    w_c = _vector(w_c, "w_c")
    w_a = _vector(w_a, "w_a")
    if beta.shape != w_c.shape:
        raise ValueError("beta and w_c must match")
    return np.concatenate((w_c - beta, w_a))


def risk(beta: Array, w_c: Array, w_a: Array, mechanism: StructuralMechanism, noise_variance: float = 0.0) -> float:
    v = risk_state(beta, w_c, w_a)
    return float(noise_variance + v @ moment_matrix(mechanism) @ v)


def risk_response(beta: Array, w_c: Array, w_a: Array, mechanism: StructuralMechanism, direction: MechanismDirection) -> float:
    v = risk_state(beta, w_c, w_a)
    return float(v @ directional_moment_matrix(mechanism, direction) @ v)


def interaction_terms(v: Array, delta_moment: Array, c_dim: int) -> dict[str, float]:
    v = _vector(v, "v")
    delta_moment = np.asarray(delta_moment, dtype=float)
    if delta_moment.shape != (v.size, v.size):
        raise ValueError("delta_moment has incompatible shape")
    delta_c = v[:c_dim]
    w_a = v[c_dim:]
    cc = delta_moment[:c_dim, :c_dim]
    ca = delta_moment[:c_dim, c_dim:]
    aa = delta_moment[c_dim:, c_dim:]
    terms = {
        "core": float(delta_c @ cc @ delta_c),
        "relation": float(2.0 * delta_c @ ca @ w_a),
        "nuisance": float(w_a @ aa @ w_a),
    }
    terms["total"] = sum(terms.values())
    return terms


def default_mechanism() -> StructuralMechanism:
    return StructuralMechanism(
        mu_c=np.array([0.0]),
        sigma_c=np.array([[1.0]]),
        gamma=np.array([[0.8]]),
        b=np.array([0.0]),
        mu_xi=np.array([0.0]),
        sigma_xi=np.array([[0.4]]),
    )


def mechanism_directions() -> dict[str, MechanismDirection]:
    z = np.zeros((1, 1))
    return {
        "core": MechanismDirection(np.array([1.0]), z, z, np.array([0.0]), np.array([0.0]), z, "core_mean"),
        "nuisance": MechanismDirection(np.array([0.0]), z, z, np.array([1.0]), np.array([0.0]), z, "nuisance_mean"),
        "relation": MechanismDirection(np.array([0.0]), z, np.array([[1.0]]), np.array([0.0]), np.array([0.0]), z, "relation"),
        "core_variance": MechanismDirection(np.array([0.0]), np.array([[1.0]]), z, np.array([0.0]), np.array([0.0]), z, "core_variance"),
        "nuisance_variance": MechanismDirection(np.array([0.0]), z, z, np.array([0.0]), np.array([0.0]), np.array([[1.0]]), "nuisance_variance"),
    }
