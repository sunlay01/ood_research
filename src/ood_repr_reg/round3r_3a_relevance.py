"""Source-optimality-aware OOD relevance quantities."""

from __future__ import annotations

import numpy as np

from .round3r_3a_benchmark import Array
from .round3r_3a_source_geometry import PopulationMoments, hessian, risk, source_excess
from .round3r_3a_residual_coupling import analytic_gradient


def shift_delta_risk(weights: Array, source: PopulationMoments, target: PopulationMoments) -> float:
    return risk(weights, target) - risk(weights, source)


def common_burden(optimum: Array, source: PopulationMoments, target: PopulationMoments) -> float:
    return abs(shift_delta_risk(optimum, source, target))


def leading_relevance(
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> float:
    gradient = analytic_gradient(optimum, source, target)
    inverse_hessian_gradient = np.linalg.solve(hessian(source), gradient)
    return float(np.sqrt(max(gradient @ inverse_hessian_gradient, 0.0)))


def local_vulnerability(
    epsilon: float,
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> float:
    return float(np.sqrt(2.0 * epsilon) * leading_relevance(optimum, source, target))


def second_order_matrix(source: PopulationMoments, target: PopulationMoments) -> Array:
    return target.second - source.second


def curvature_coefficient(source: PopulationMoments, target: PopulationMoments) -> float:
    # K_s is defined on the M_S-unit ellipsoid.  Using H_S here would insert
    # an extra factor of 1/2 into the finite-epsilon bound.
    values, vectors = np.linalg.eigh((source.second + source.second.T) / 2.0)
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    whitened = inverse_root @ second_order_matrix(source, target) @ inverse_root
    return float(np.max(np.abs(np.linalg.eigvalsh((whitened + whitened.T) / 2.0))))


def finite_epsilon_bound(
    epsilon: float,
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> tuple[float, float, float]:
    """Return local support, curvature, and the two-sided error bound."""
    local = local_vulnerability(epsilon, optimum, source, target)
    curvature = curvature_coefficient(source, target)
    return local, curvature, float(epsilon * curvature)


def response_polynomial(
    delta: Array,
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> float:
    """Return ``Delta_s(w*+delta)-Delta_s(w*)`` exactly."""
    d = np.asarray(delta, dtype=float)
    gradient = analytic_gradient(optimum, source, target)
    return float(d @ second_order_matrix(source, target) @ d + gradient @ d)


def source_good(weights: Array, optimum: Array, source: PopulationMoments, epsilon: float, tolerance: float = 1e-10) -> bool:
    return source_excess(weights, optimum, source) <= epsilon + tolerance


def relevance_signature(
    optimum: Array,
    source: PopulationMoments,
    target: PopulationMoments,
) -> Array:
    """Return the whitened gradient ``H_S^-1/2 g_s``."""
    gradient = analytic_gradient(optimum, source, target)
    values, vectors = np.linalg.eigh((hessian(source) + hessian(source).T) / 2.0)
    if np.min(values) <= 0:
        raise ValueError("source Hessian must be positive definite")
    inverse_root = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    return inverse_root @ gradient


def relevance_spectrum(signatures: Array, tolerance: float = 1e-10) -> dict[str, object]:
    values = np.asarray(signatures, dtype=float)
    if values.ndim != 2:
        raise ValueError("signatures must be a matrix")
    singular = np.linalg.svd(values, compute_uv=False)
    energy = singular**2
    total = max(float(np.sum(energy)), 1e-30)
    normalized = energy / total
    cumulative = np.cumsum(normalized)
    return {
        "singular_values": singular,
        "numerical_rank": int(np.sum(singular > tolerance * singular[0])) if singular.size and singular[0] > 0 else 0,
        "stable_rank": float(np.sum(energy) / max(singular[0] ** 2, 1e-30)) if singular.size else 0.0,
        "entropy_effective_rank": float(np.exp(-np.sum(normalized[normalized > 0] * np.log(normalized[normalized > 0])))) if singular.size else 0.0,
        "participation_ratio": float(np.sum(energy) ** 2 / max(np.sum(energy**2), 1e-30)),
        "energy_threshold_indices": {str(level): int(np.searchsorted(cumulative, level) + 1) for level in (0.90, 0.95, 0.99)},
    }


def fit_scaling_exponent(epsilons: Array, vulnerabilities: Array) -> float:
    x = np.asarray(epsilons, dtype=float)
    y = np.asarray(vulnerabilities, dtype=float)
    mask = (x > 0) & (y > 1e-14)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])
