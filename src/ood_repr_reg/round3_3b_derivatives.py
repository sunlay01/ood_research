"""Analytic structural paths and risk derivatives for round 3B.

The path used here is affine in the structural parameters and deliberately
does not project covariance matrices back to the SPD cone.  Callers must
choose directions for which the requested path points are valid.  This keeps
the moment map a genuine polynomial, so the derivatives below are analytic
coefficients rather than finite-difference labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Iterable

import numpy as np

from .round3_algebra import model_lift, svec_symmetric

Array = np.ndarray


def _vector(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return result.copy()


def _matrix(value: Array, name: str) -> Array:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.allclose(result, result.T, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    return result.copy()


@dataclass(frozen=True)
class OrderEnvironment:
    """A structural environment without covariance projection or clamping."""

    mu_c: Array
    sigma_c: Array
    gamma: Array
    b: Array
    mu_xi: Array
    sigma_xi: Array
    beta: Array
    noise_variance: float = 0.25

    def __post_init__(self) -> None:
        mu_c = _vector(self.mu_c, "mu_c")
        sigma_c = _matrix(self.sigma_c, "sigma_c")
        gamma = np.asarray(self.gamma, dtype=float)
        b = _vector(self.b, "b")
        mu_xi = _vector(self.mu_xi, "mu_xi")
        sigma_xi = _matrix(self.sigma_xi, "sigma_xi")
        beta = _vector(self.beta, "beta")
        if sigma_c.shape != (mu_c.size, mu_c.size):
            raise ValueError("sigma_c has incompatible shape")
        if gamma.ndim != 2 or gamma.shape[1] != mu_c.size:
            raise ValueError("gamma has incompatible shape")
        if b.size != gamma.shape[0] or mu_xi.size != gamma.shape[0]:
            raise ValueError("nuisance vectors have incompatible shape")
        if sigma_xi.shape != (gamma.shape[0], gamma.shape[0]):
            raise ValueError("sigma_xi has incompatible shape")
        if beta.shape != mu_c.shape:
            raise ValueError("beta has incompatible shape")
        if self.noise_variance < 0.0:
            raise ValueError("noise_variance must be nonnegative")
        object.__setattr__(self, "mu_c", mu_c)
        object.__setattr__(self, "sigma_c", sigma_c)
        object.__setattr__(self, "gamma", gamma.copy())
        object.__setattr__(self, "b", b)
        object.__setattr__(self, "mu_xi", mu_xi)
        object.__setattr__(self, "sigma_xi", sigma_xi)
        object.__setattr__(self, "beta", beta)

    @property
    def c_dim(self) -> int:
        return self.mu_c.size

    @property
    def a_dim(self) -> int:
        return self.b.size

    @property
    def mu_a(self) -> Array:
        return self.gamma @ self.mu_c + self.b + self.mu_xi

    def at(self, direction: "OrderDirection", t: float) -> "OrderEnvironment":
        """Evaluate the affine structural path ``eta_0 + t * direction``."""
        return OrderEnvironment(
            self.mu_c + t * direction.d_mu_c,
            self.sigma_c + t * direction.d_sigma_c,
            self.gamma + t * direction.d_gamma,
            self.b + t * direction.d_b,
            self.mu_xi + t * direction.d_mu_xi,
            self.sigma_xi + t * direction.d_sigma_xi,
            self.beta + t * direction.d_beta,
            self.noise_variance,
        )

    def covariance_eigenvalues(self) -> tuple[Array, Array]:
        return np.linalg.eigvalsh(self.sigma_c), np.linalg.eigvalsh(self.sigma_xi)

    def is_valid_covariance(self, tolerance: float = 1e-10) -> bool:
        core, nuisance = self.covariance_eigenvalues()
        return bool(np.min(core) > tolerance and np.min(nuisance) > tolerance)


@dataclass(frozen=True)
class OrderDirection:
    """A tangent direction in structural parameter space."""

    d_mu_c: Array
    d_sigma_c: Array
    d_gamma: Array
    d_b: Array
    d_mu_xi: Array
    d_sigma_xi: Array
    d_beta: Array
    name: str = "direction"
    family: str = "custom"

    def __post_init__(self) -> None:
        object.__setattr__(self, "d_mu_c", _vector(self.d_mu_c, "d_mu_c"))
        object.__setattr__(self, "d_sigma_c", _matrix(self.d_sigma_c, "d_sigma_c"))
        gamma = np.asarray(self.d_gamma, dtype=float)
        if gamma.ndim != 2:
            raise ValueError("d_gamma must be two-dimensional")
        object.__setattr__(self, "d_gamma", gamma.copy())
        object.__setattr__(self, "d_b", _vector(self.d_b, "d_b"))
        object.__setattr__(self, "d_mu_xi", _vector(self.d_mu_xi, "d_mu_xi"))
        object.__setattr__(self, "d_sigma_xi", _matrix(self.d_sigma_xi, "d_sigma_xi"))
        object.__setattr__(self, "d_beta", _vector(self.d_beta, "d_beta"))

    def scaled(self, scale: float, name: str | None = None) -> "OrderDirection":
        return OrderDirection(
            self.d_mu_c * scale,
            self.d_sigma_c * scale,
            self.d_gamma * scale,
            self.d_b * scale,
            self.d_mu_xi * scale,
            self.d_sigma_xi * scale,
            self.d_beta * scale,
            name=name or self.name,
            family=self.family,
        )


def direction_from_environments(base: OrderEnvironment, target: OrderEnvironment, name: str = "target") -> OrderDirection:
    """Return the structural difference ``target - base``."""
    if (base.c_dim, base.a_dim) != (target.c_dim, target.a_dim):
        raise ValueError("base and target dimensions must match")
    return OrderDirection(
        target.mu_c - base.mu_c,
        target.sigma_c - base.sigma_c,
        target.gamma - base.gamma,
        target.b - base.b,
        target.mu_xi - base.mu_xi,
        target.sigma_xi - base.sigma_xi,
        target.beta - base.beta,
        name=name,
        family="target-path",
    )


def combine_directions(*directions: OrderDirection, name: str = "mixed", family: str = "mixed") -> OrderDirection:
    """Add structural directions componentwise."""
    if not directions:
        raise ValueError("at least one direction is required")
    first = directions[0]
    return OrderDirection(
        sum((d.d_mu_c for d in directions), start=np.zeros_like(first.d_mu_c)),
        sum((d.d_sigma_c for d in directions), start=np.zeros_like(first.d_sigma_c)),
        sum((d.d_gamma for d in directions), start=np.zeros_like(first.d_gamma)),
        sum((d.d_b for d in directions), start=np.zeros_like(first.d_b)),
        sum((d.d_mu_xi for d in directions), start=np.zeros_like(first.d_mu_xi)),
        sum((d.d_sigma_xi for d in directions), start=np.zeros_like(first.d_sigma_xi)),
        sum((d.d_beta for d in directions), start=np.zeros_like(first.d_beta)),
        name=name,
        family=family,
    )


# A polynomial is represented by coefficient arrays in ascending powers of t.
Poly = list[Array]


def _as_array(value: Array | float) -> Array:
    return np.asarray(value, dtype=float)


def _poly_trim(poly: Poly) -> Poly:
    result = [np.asarray(value, dtype=float) for value in poly]
    while len(result) > 1 and np.all(np.abs(result[-1]) <= 0.0):
        result.pop()
    return result


def _poly_pad(poly: Poly, degree: int, shape: tuple[int, ...]) -> Poly:
    result = [np.zeros(shape, dtype=float) for _ in range(degree + 1)]
    for index, value in enumerate(poly[: degree + 1]):
        result[index] = np.asarray(value, dtype=float)
    return result


def _poly_add(*polys: Poly) -> Poly:
    if not polys:
        raise ValueError("at least one polynomial is required")
    degree = max(len(poly) for poly in polys)
    shape = polys[0][0].shape
    result = [np.zeros(shape, dtype=float) for _ in range(degree)]
    for poly in polys:
        if poly[0].shape != shape:
            raise ValueError("polynomial shapes do not match")
        for index, value in enumerate(poly):
            result[index] += value
    return _poly_trim(result)


def _poly_scale(poly: Poly, scale: float) -> Poly:
    return [np.asarray(value, dtype=float) * scale for value in poly]


def _poly_matmul(first: Poly, second: Poly) -> Poly:
    degree = len(first) + len(second) - 2
    shape = np.matmul(first[0], second[0]).shape
    result = [np.zeros(shape, dtype=float) for _ in range(degree + 1)]
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            result[i + j] += np.matmul(left, right)
    return _poly_trim(result)


def _poly_outer(first: Poly, second: Poly) -> Poly:
    degree = len(first) + len(second) - 2
    shape = np.outer(first[0], second[0]).shape
    result = [np.zeros(shape, dtype=float) for _ in range(degree + 1)]
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            result[i + j] += np.outer(left, right)
    return _poly_trim(result)


def _poly_dot(first: Poly, second: Poly) -> Poly:
    degree = len(first) + len(second) - 2
    result = [np.asarray(0.0) for _ in range(degree + 1)]
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            result[i + j] += np.asarray(np.dot(np.ravel(left), np.ravel(right)))
    return _poly_trim(result)


def _poly_transpose(poly: Poly) -> Poly:
    return [np.asarray(value).T for value in poly]


def _poly_concat(first: Poly, second: Poly) -> Poly:
    degree = max(len(first), len(second))
    shape = (first[0].size + second[0].size,)
    left = _poly_pad(first, degree - 1, first[0].shape)
    right = _poly_pad(second, degree - 1, second[0].shape)
    return [np.concatenate((left[index].reshape(-1), right[index].reshape(-1))) for index in range(degree)]


def _poly_block(cc: Poly, ca: Poly, ac: Poly, aa: Poly, c_dim: int, a_dim: int) -> Poly:
    degree = max(len(cc), len(ca), len(ac), len(aa))
    total = c_dim + a_dim
    result = [np.zeros((total, total), dtype=float) for _ in range(degree)]
    for index in range(degree):
        result[index][:c_dim, :c_dim] = cc[index] if index < len(cc) else 0.0
        result[index][:c_dim, c_dim:] = ca[index] if index < len(ca) else 0.0
        result[index][c_dim:, :c_dim] = ac[index] if index < len(ac) else 0.0
        result[index][c_dim:, c_dim:] = aa[index] if index < len(aa) else 0.0
    return _poly_trim(result)


def _poly_linear(base: Array, direction: Array) -> Poly:
    return [np.asarray(base, dtype=float).copy(), np.asarray(direction, dtype=float).copy()]


def _poly_constant(value: Array | float) -> Poly:
    return [np.asarray(value, dtype=float).copy()]


def moment_polynomial(base: OrderEnvironment, direction: OrderDirection) -> tuple[Poly, Poly, Poly]:
    """Return polynomial coefficients for ``M_XX(t), m_XY(t), m_Y2(t)``."""
    if direction.d_gamma.shape != base.gamma.shape:
        raise ValueError("direction and environment dimensions do not match")
    mu_c = _poly_linear(base.mu_c, direction.d_mu_c)
    sigma_c = _poly_linear(base.sigma_c, direction.d_sigma_c)
    gamma = _poly_linear(base.gamma, direction.d_gamma)
    b = _poly_linear(base.b, direction.d_b)
    mu_xi = _poly_linear(base.mu_xi, direction.d_mu_xi)
    sigma_xi = _poly_linear(base.sigma_xi, direction.d_sigma_xi)
    beta = _poly_linear(base.beta, direction.d_beta)

    mu_a = _poly_add(_poly_matmul(gamma, mu_c), b, mu_xi)
    sigma_ca = _poly_matmul(sigma_c, _poly_transpose(gamma))
    sigma_aa = _poly_add(_poly_matmul(_poly_matmul(gamma, sigma_c), _poly_transpose(gamma)), sigma_xi)

    mcc = _poly_add(sigma_c, _poly_outer(mu_c, mu_c))
    mca = _poly_add(sigma_ca, _poly_outer(mu_c, mu_a))
    maa = _poly_add(sigma_aa, _poly_outer(mu_a, mu_a))
    # 3A uses the affine predictor X=(1,C,A), so include its intercept row.
    z = _poly_concat(mu_c, mu_a)
    z_cov = _poly_block(mcc, mca, _poly_transpose(mca), maa, base.c_dim, base.a_dim)
    degree = len(z_cov)
    mxx = [np.zeros((1 + base.c_dim + base.a_dim, 1 + base.c_dim + base.a_dim), dtype=float) for _ in range(degree)]
    for index in range(degree):
        mxx[index][0, 0] = 1.0 if index == 0 else 0.0
        z_index = z[index] if index < len(z) else np.zeros_like(z[0])
        mxx[index][0, 1:] = z_index
        mxx[index][1:, 0] = z_index
        mxx[index][1:, 1:] = z_cov[index]

    mean_y = _poly_dot(beta, mu_c)
    cov_cy = _poly_matmul(sigma_c, beta)
    cov_ay = _poly_matmul(gamma, cov_cy)
    mu_z = _poly_concat(mu_c, mu_a)
    xy_z = _poly_add(_poly_concat(cov_cy, cov_ay), _poly_elementwise_product(mu_z, mean_y))
    mxy = _poly_concat(mean_y, xy_z)
    y2 = _poly_add(
        _poly_dot(beta, _poly_matmul(sigma_c, beta)),
        _poly_elementwise_product(mean_y, mean_y),
        _poly_constant(base.noise_variance),
    )
    return mxx, mxy, y2


def _poly_elementwise_product(first: Poly, second: Poly) -> Poly:
    degree = len(first) + len(second) - 2
    shape = np.broadcast_shapes(first[0].shape, second[0].shape)
    result = [np.zeros(shape, dtype=float) for _ in range(degree + 1)]
    for i, left in enumerate(first):
        for j, right in enumerate(second):
            result[i + j] += left * right
    return _poly_trim(result)


def evaluate_polynomial(poly: Poly, t: float) -> Array:
    result = np.zeros_like(poly[0], dtype=float)
    for coefficient in reversed(poly):
        result = result * t + coefficient
    return result


def moment_statistics(environment: OrderEnvironment) -> tuple[Array, Array, float]:
    """Evaluate the raw second moments at one environment point."""
    zero = OrderDirection(
        np.zeros_like(environment.mu_c),
        np.zeros_like(environment.sigma_c),
        np.zeros_like(environment.gamma),
        np.zeros_like(environment.b),
        np.zeros_like(environment.mu_xi),
        np.zeros_like(environment.sigma_xi),
        np.zeros_like(environment.beta),
    )
    mxx, mxy, y2 = moment_polynomial(environment, zero)
    return mxx[0], mxy[0], float(y2[0])


def risk_at(weights: Array, environment: OrderEnvironment) -> float:
    mxx, mxy, y2 = moment_statistics(environment)
    w = np.asarray(weights, dtype=float)
    return float(w @ mxx @ w - 2.0 * w @ mxy + y2)


def risk_polynomial(weights: Array, base: OrderEnvironment, direction: OrderDirection) -> Array:
    mxx, mxy, y2 = moment_polynomial(base, direction)
    w = np.asarray(weights, dtype=float)
    degree = max(len(mxx), len(mxy), len(y2))
    result = np.zeros(degree, dtype=float)
    for index in range(degree):
        matrix = mxx[index] if index < len(mxx) else np.zeros_like(mxx[0])
        vector = mxy[index] if index < len(mxy) else np.zeros_like(mxy[0])
        scalar = y2[index] if index < len(y2) else np.asarray(0.0)
        result[index] = float(w @ matrix @ w - 2.0 * w @ vector + scalar)
    return np.trim_zeros(result, trim="b") if np.any(result) else np.asarray([0.0])


def risk_derivative(weights: Array, base: OrderEnvironment, direction: OrderDirection, order: int) -> float:
    if order < 0:
        raise ValueError("order must be nonnegative")
    coefficients = risk_polynomial(weights, base, direction)
    if order >= coefficients.size:
        return 0.0
    return float(factorial(order) * coefficients[order])


def moment_derivative(base: OrderEnvironment, direction: OrderDirection, order: int) -> tuple[Array, Array, float]:
    """Return ``D^order(M_XX,m_XY,m_Y2)[direction^order]``."""
    mxx, mxy, y2 = moment_polynomial(base, direction)
    if order < len(mxx):
        d_mxx = factorial(order) * mxx[order]
    else:
        d_mxx = np.zeros_like(mxx[0])
    if order < len(mxy):
        d_mxy = factorial(order) * mxy[order]
    else:
        d_mxy = np.zeros_like(mxy[0])
    d_y2 = factorial(order) * float(y2[order]) if order < len(y2) else 0.0
    return d_mxx, d_mxy, d_y2


def risk_statistic_lift_derivative(base: OrderEnvironment, direction: OrderDirection, order: int) -> Array:
    """Lift the analytic order derivative into the 3A risk-statistic space."""
    d_mxx, d_mxy, d_y2 = moment_derivative(base, direction, order)
    return np.concatenate((svec_symmetric(d_mxx), d_mxy, np.asarray([d_y2])))


def model_response_from_liftings(model_weights: Iterable[Array], liftings: Array) -> Array:
    """Apply the 3A model lifting to an order-lifting matrix."""
    phi = np.asarray([model_lift(weights) for weights in model_weights], dtype=float)
    values = np.asarray(liftings, dtype=float)
    if values.ndim != 2 or values.shape[1] != phi.shape[1]:
        raise ValueError("liftings have incompatible feature dimension")
    return phi @ values.T


def directional_liftings(base: OrderEnvironment, directions: Iterable[OrderDirection], order: int) -> Array:
    directions = tuple(directions)
    x_dimension = 1 + base.c_dim + base.a_dim
    dimension = x_dimension * (x_dimension + 1) // 2 + x_dimension + 1
    if not directions:
        return np.zeros((0, dimension), dtype=float)
    return np.asarray([risk_statistic_lift_derivative(base, direction, order) for direction in directions], dtype=float)


def pure_family_directions(
    base: OrderEnvironment,
    samples_per_family: int = 100,
    seed: int = 20260906,
) -> dict[str, tuple[OrderDirection, ...]]:
    """Return deterministic random directions spanning each pure family.

    The coordinate axes are useful for hand inspection, but a filtration claim
    should not depend on that finite choice.  The returned directions are
    normalized random combinations within one structural block only.  Their
    amplitudes are deliberately small and the runner additionally checks the
    covariance path at its largest finite-difference stencil.
    """
    c, a = base.c_dim, base.a_dim
    if (c, a) != (2, 2):
        raise ValueError("3B main track requires d_C=d_A=2")
    if samples_per_family < 1:
        raise ValueError("samples_per_family must be positive")
    zero_c = np.zeros(c)
    zero_a = np.zeros(a)
    zero_cc = np.zeros((c, c))
    zero_aa = np.zeros((a, a))
    families: dict[str, list[OrderDirection]] = {name: [] for name in ("core_mean", "core_covariance", "relation", "b", "mu_xi", "sigma_xi", "task")}

    rng = np.random.default_rng(seed)

    def random_vector(scale: float, dimension: int) -> Array:
        value = rng.normal(size=dimension)
        return scale * value / max(np.linalg.norm(value), 1e-30)

    def random_symmetric(scale: float, dimension: int) -> Array:
        value = rng.normal(size=(dimension, dimension))
        value = (value + value.T) / 2.0
        return scale * value / max(np.linalg.norm(value, ord="fro"), 1e-30)

    def random_matrix(scale: float, shape: tuple[int, int]) -> Array:
        value = rng.normal(size=shape)
        return scale * value / max(np.linalg.norm(value, ord="fro"), 1e-30)

    for index in range(samples_per_family):
        families["core_mean"].append(OrderDirection(
            random_vector(0.35, c), zero_cc, np.zeros_like(base.gamma), zero_a, zero_a,
            zero_aa, zero_c, f"core_mean[{index}]", "core_mean"))
        families["task"].append(OrderDirection(
            zero_c, zero_cc, np.zeros_like(base.gamma), zero_a, zero_a, zero_aa,
            random_vector(0.25, c), f"task[{index}]", "task"))
        families["core_covariance"].append(OrderDirection(
            zero_c, random_symmetric(0.10, c), np.zeros_like(base.gamma), zero_a,
            zero_a, zero_aa, zero_c, f"core_covariance[{index}]", "core_covariance"))
        families["relation"].append(OrderDirection(
            zero_c, zero_cc, random_matrix(0.25, base.gamma.shape), zero_a, zero_a,
            zero_aa, zero_c, f"relation[{index}]", "relation"))
        for name, field in (("b", "d_b"), ("mu_xi", "d_mu_xi")):
            kwargs = {
                "d_mu_c": zero_c, "d_sigma_c": zero_cc, "d_gamma": np.zeros_like(base.gamma),
                "d_b": zero_a, "d_mu_xi": zero_a, "d_sigma_xi": zero_aa, "d_beta": zero_c,
            }
            kwargs[field] = random_vector(0.35, a)
            families[name].append(OrderDirection(**kwargs, name=f"{name}[{index}]", family=name))
        families["sigma_xi"].append(OrderDirection(
            zero_c, zero_cc, np.zeros_like(base.gamma), zero_a, zero_a,
            random_symmetric(0.10, a), zero_c, f"sigma_xi[{index}]", "sigma_xi"))
    return {name: tuple(values) for name, values in families.items()}
