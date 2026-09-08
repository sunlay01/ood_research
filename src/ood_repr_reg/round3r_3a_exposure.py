"""Source exposure diagnostics, deliberately separate from relevance."""

from __future__ import annotations

import numpy as np

from .round3r_3a_benchmark import Array, RelevanceEnvironment
from .round3r_3a_residual_coupling import analytic_gradient
from .round3r_3a_source_geometry import PopulationMoments, mixture_moments


def structural_vector(environment: RelevanceEnvironment) -> Array:
    """A registered coordinate chart used only for exposure diagnostics."""
    shortcut_covariance = (
        np.asarray(environment.shortcut_noise_covariance, dtype=float)
        if environment.shortcut_noise_covariance is not None
        else np.diag(np.square(environment.sigma_shortcut))
        + environment.shortcut_shared_sigma**2 * np.ones((len(environment.shortcut_rhos), len(environment.shortcut_rhos)))
    )
    return np.concatenate((
        np.asarray(environment.shortcut_rhos, dtype=float),
        shortcut_covariance.reshape(-1),
        [environment.u_gamma, environment.z_gamma, environment.y_noise_variance],
        np.asarray(environment.noise_mean, dtype=float),
        np.asarray(environment.noise_variance, dtype=float),
    ))


def _projection_score(vector: Array, columns: Array, tolerance: float = 1e-10) -> tuple[float, float]:
    value = np.asarray(vector, dtype=float)
    matrix = np.asarray(columns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != value.size:
        raise ValueError("projection matrix has incompatible shape")
    if matrix.shape[1] == 0 or np.linalg.norm(value) <= tolerance:
        return 0.0, float(np.linalg.norm(value))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.sum(singular > tolerance * max(singular[0], 1e-30))) if singular.size else 0
    projection = u[:, :rank] @ (u[:, :rank].T @ value) if rank else np.zeros_like(value)
    norm = float(np.linalg.norm(value))
    return (float(np.linalg.norm(projection)) / norm if norm > tolerance else 0.0, float(np.linalg.norm(value - projection)))


def structural_exposure(
    source_environments: tuple[RelevanceEnvironment, ...],
    base: RelevanceEnvironment,
    target: RelevanceEnvironment,
    tolerance: float = 1e-10,
) -> dict[str, float | bool]:
    source_deltas = np.column_stack([structural_vector(env) - structural_vector(base) for env in source_environments])
    target_delta = structural_vector(target) - structural_vector(base)
    score, residual = _projection_score(target_delta, source_deltas, tolerance)
    return {"score": score, "residual": residual, "exposed": bool(residual <= tolerance * max(1.0, np.linalg.norm(target_delta)))}


def risk_response_exposure(
    source_environments: tuple[RelevanceEnvironment, ...],
    source_moments: PopulationMoments,
    optimum: Array,
    target: PopulationMoments,
    tolerance: float = 1e-10,
) -> dict[str, float | bool]:
    source_gradients = np.column_stack([
        analytic_gradient(optimum, source_moments, mixture_moments((env,)))
        for env in source_environments
    ])
    target_gradient = analytic_gradient(optimum, source_moments, target)
    score, residual = _projection_score(target_gradient, source_gradients, tolerance)
    return {"score": score, "residual": residual, "exposed": bool(residual <= tolerance * max(1.0, np.linalg.norm(target_gradient)))}


def classify_relevance_exposure(
    relevance: float,
    exposure: float,
    relevance_threshold: float,
    exposure_threshold: float = 0.8,
) -> str:
    relevant = relevance >= relevance_threshold
    exposed = exposure >= exposure_threshold
    if relevant and exposed:
        return "exposed-relevant"
    if relevant:
        return "unexposed-relevant"
    if exposed:
        return "exposed-low-relevance"
    return "unexposed-low-relevance"
