"""Population training for the round-three empirical retry.

The retry deliberately keeps the model family small and inspectable.  Every
returned predictor is obtained by optimizing a declared population objective;
no predictor coefficients are hand assigned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .round3_shift_ensemble import PopulationEnvironment

Array = np.ndarray


@dataclass(frozen=True)
class TrainedModel:
    model_id: str
    method: str
    seed: int
    lambda_: float
    weights: Array
    source_risk: float
    penalty: float
    optimization_status: str
    representation_dim: int | None = None


def _stats(environment: PopulationEnvironment) -> tuple[Array, Array, float]:
    """Return E[XX'], E[XY], E[Y^2] for X=(1,C,A)."""
    c, a = environment.c_dim, environment.a_dim
    mu_c = environment.mu_c
    mu_a = environment.mu_a
    sigma_c = environment.sigma_c
    sigma_ca = sigma_c @ environment.gamma.T
    sigma_a = environment.gamma @ sigma_c @ environment.gamma.T + environment.sigma_xi
    mu_z = np.concatenate((mu_c, mu_a))
    sigma_z = np.block([[sigma_c, sigma_ca], [sigma_ca.T, sigma_a]])
    raw_z = sigma_z + np.outer(mu_z, mu_z)
    moment_xx = np.zeros((1 + c + a, 1 + c + a), dtype=float)
    moment_xx[0, 0] = 1.0
    moment_xx[0, 1:] = mu_z
    moment_xx[1:, 0] = mu_z
    moment_xx[1:, 1:] = raw_z
    mean_y = float(environment.beta @ mu_c)
    xy_z = sigma_z[:, :c] @ environment.beta + mu_z * mean_y
    moment_xy = np.concatenate(([mean_y], xy_z))
    y2 = float(environment.beta @ sigma_c @ environment.beta + mean_y**2 + environment.noise_variance)
    return moment_xx, moment_xy, y2


def risk(weights: Array, environment: PopulationEnvironment) -> float:
    if environment.family != "linear-gaussian":
        return monte_carlo_risk(weights, environment)
    mxx, mxy, y2 = _stats(environment)
    w = np.asarray(weights, dtype=float)
    return float(w @ mxx @ w - 2.0 * w @ mxy + y2)


def monte_carlo_risk(weights: Array, environment: PopulationEnvironment, samples: int = 30000, seed: int = 911) -> float:
    rng = np.random.default_rng(seed)
    c = rng.multivariate_normal(environment.mu_c, environment.sigma_c, size=samples)
    xi = rng.multivariate_normal(environment.mu_xi, environment.sigma_xi, size=samples)
    a = c @ environment.gamma.T + environment.b + xi
    if environment.quadratic_alpha is not None:
        a = a + (c**2) @ environment.quadratic_alpha.T
    y = c @ environment.beta + rng.normal(scale=np.sqrt(environment.noise_variance), size=samples)
    x = np.concatenate((np.ones((samples, 1)), c, a), axis=1)
    residual = x @ np.asarray(weights, dtype=float) - y
    return float(np.mean(residual**2))


def _source_risk(weights: Array, sources: tuple[PopulationEnvironment, ...]) -> float:
    return float(np.mean([risk(weights, env) for env in sources]))


def _ridge_solution(sources: tuple[PopulationEnvironment, ...], lam: float) -> Array:
    mxx = np.mean([_stats(env)[0] for env in sources], axis=0)
    mxy = np.mean([_stats(env)[1] for env in sources], axis=0)
    penalty = np.eye(mxx.shape[0])
    penalty[0, 0] = 0.0
    return np.linalg.solve(mxx + lam * penalty, mxy)


def _irm_penalty(weights: Array, sources: tuple[PopulationEnvironment, ...]) -> float:
    values = []
    for env in sources:
        mxx, mxy, _ = _stats(env)
        values.append(2.0 * float(weights @ (mxx @ weights - mxy)))
    return float(np.mean(np.square(values)))


def _vrex_penalty(weights: Array, sources: tuple[PopulationEnvironment, ...]) -> float:
    values = np.asarray([risk(weights, env) for env in sources])
    return float(np.var(values))


def _optimize(objective, initial: Array, seed: int, maxiter: int = 300) -> tuple[Array, str]:
    try:
        from scipy.optimize import minimize
    except ImportError:  # pragma: no cover - research extras provide scipy
        return initial, "scipy-unavailable"
    result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12})
    if result.success and np.all(np.isfinite(result.x)):
        return np.asarray(result.x, dtype=float), "success"
    # QR-based gauge reconstruction is piecewise smooth.  A derivative-free
    # retry keeps the model in the declared objective instead of accepting an
    # arbitrary hand-written fallback.
    fallback = minimize(objective, initial, method="Powell", options={"maxiter": maxiter, "xtol": 1e-7, "ftol": 1e-9})
    if np.all(np.isfinite(fallback.x)):
        return np.asarray(fallback.x, dtype=float), "fallback-powell"
    return np.asarray(result.x, dtype=float), f"failed:{result.message}"


def _coral_covariance(environment: PopulationEnvironment) -> Array:
    sigma_ca = environment.sigma_c @ environment.gamma.T
    sigma_a = environment.gamma @ environment.sigma_c @ environment.gamma.T + environment.sigma_xi
    return np.block([[environment.sigma_c, sigma_ca], [sigma_ca.T, sigma_a]])


def _orthonormal_rows(raw: Array, rows: int) -> Array:
    q, _ = np.linalg.qr(np.asarray(raw, dtype=float).T)
    return q[:, :rows].T


def _fit_coral(sources: tuple[PopulationEnvironment, ...], lam: float, seed: int) -> tuple[Array, float, str, int]:
    p = sources[0].c_dim + sources[0].a_dim
    k = min(2, p)
    rng = np.random.default_rng(seed)
    initial_b = _orthonormal_rows(rng.normal(size=(k, p)), k)
    initial = np.concatenate((np.zeros(1), rng.normal(scale=0.1, size=k), initial_b.ravel()))

    covariances = [_coral_covariance(env) for env in sources]

    def unpack(params: Array) -> tuple[float, Array, Array]:
        intercept = float(params[0])
        head = params[1 : 1 + k]
        raw = params[1 + k :].reshape(k, p)
        return intercept, head, _orthonormal_rows(raw, k)

    def objective(params: Array) -> float:
        intercept, head, b = unpack(params)
        weights = np.concatenate(([intercept], b.T @ head))
        fit = _source_risk(weights, sources)
        reference = covariances[0]
        align = sum(float(np.linalg.norm(b @ (cov - reference) @ b.T, ord="fro") ** 2) for cov in covariances[1:])
        return fit + lam * align

    params, status = _optimize(objective, initial, seed)
    intercept, head, b = unpack(params)
    weights = np.concatenate(([intercept], b.T @ head))
    penalty = float(sum(np.linalg.norm(b @ (cov - covariances[0]) @ b.T, ord="fro") ** 2 for cov in covariances[1:]))
    return weights, penalty, status, k


def train_models(
    sources: tuple[PopulationEnvironment, ...],
    seeds: Iterable[int] = range(20),
    lambdas: Iterable[float] | None = None,
) -> tuple[TrainedModel, ...]:
    """Train the complete retry ensemble on source environments only."""
    strengths = tuple(lambdas or np.logspace(-3, 1, 8))
    source_tuple = tuple(sources)
    models: list[TrainedModel] = []
    for seed in tuple(seeds):
        rng = np.random.default_rng(seed)
        erm = _ridge_solution(source_tuple, 0.0)
        models.append(TrainedModel(f"erm-s{seed}", "ERM", seed, 0.0, erm, _source_risk(erm, source_tuple), 0.0, "analytic"))
        for lam in strengths:
            ridge = _ridge_solution(source_tuple, float(lam))
            models.append(TrainedModel(f"l2-s{seed}-l{lam:.4g}", "L2", seed, float(lam), ridge, _source_risk(ridge, source_tuple), float(np.sum(ridge[1:] ** 2)), "analytic"))
            init = erm + rng.normal(scale=0.05, size=erm.shape)
            irm, irm_status = _optimize(lambda w: _source_risk(w, source_tuple) + lam * _irm_penalty(w, source_tuple), init, seed)
            models.append(TrainedModel(f"irm-s{seed}-l{lam:.4g}", "IRMv1", seed, float(lam), irm, _source_risk(irm, source_tuple), _irm_penalty(irm, source_tuple), irm_status))
            vrex, vrex_status = _optimize(lambda w: _source_risk(w, source_tuple) + lam * _vrex_penalty(w, source_tuple), init, seed)
            models.append(TrainedModel(f"vrex-s{seed}-l{lam:.4g}", "V-REx", seed, float(lam), vrex, _source_risk(vrex, source_tuple), _vrex_penalty(vrex, source_tuple), vrex_status))
            coral_w, coral_penalty, coral_status, k = _fit_coral(source_tuple, float(lam), seed)
            models.append(TrainedModel(f"coral-s{seed}-l{lam:.4g}", "CORAL", seed, float(lam), coral_w, _source_risk(coral_w, source_tuple), coral_penalty, coral_status, k))
    return tuple(models)
