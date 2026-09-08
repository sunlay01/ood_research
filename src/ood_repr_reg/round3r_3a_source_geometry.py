"""Source population geometry for the relevance benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3a_benchmark import Array, RelevanceEnvironment


@dataclass(frozen=True)
class PopulationMoments:
    second: Array
    xy: Array
    y2: float


def moments(environment: RelevanceEnvironment) -> PopulationMoments:
    """Return ``E[XX']``, ``E[XY]``, and ``E[Y^2]`` for the environment."""
    p = environment.dimension - 1
    coefficients = np.zeros(p)
    coefficients[0] = 1.0  # C = Y + noise.
    shortcut_count = len(environment.shortcut_rhos)
    coefficients[1 : 1 + shortcut_count] = environment.shortcut_rhos
    coefficients[1 + shortcut_count + environment.n_noise] = environment.u_gamma
    coefficients[-1] = environment.z_gamma

    covariance = np.outer(coefficients, coefficients)
    covariance[0, 0] += environment.sigma_c**2
    offset = 1
    shortcut_covariance = (
        np.asarray(environment.shortcut_noise_covariance, dtype=float).copy()
        if environment.shortcut_noise_covariance is not None
        else np.diag(np.square(environment.sigma_shortcut))
    )
    if environment.shortcut_noise_covariance is None and environment.shortcut_shared_sigma:
        shortcut_covariance += environment.shortcut_shared_sigma**2 * np.ones((shortcut_count, shortcut_count))
    covariance[offset : offset + shortcut_count, offset : offset + shortcut_count] += shortcut_covariance
    offset += shortcut_count
    noise_covariance = np.diag(np.asarray(environment.noise_variance, dtype=float))
    if environment.noise_rotation is not None:
        noise_covariance = environment.noise_rotation @ noise_covariance @ environment.noise_rotation.T
    covariance[offset : offset + environment.n_noise, offset : offset + environment.n_noise] += noise_covariance
    offset += environment.n_noise
    covariance[offset, offset] += environment.u_sigma**2
    covariance[offset + 1, offset + 1] += environment.z_sigma**2

    means = np.zeros(p)
    means[1 + shortcut_count : 1 + shortcut_count + environment.n_noise] = environment.noise_mean
    raw = covariance + np.outer(means, means)
    second = np.zeros((p + 1, p + 1))
    second[0, 0] = 1.0
    second[0, 1:] = means
    second[1:, 0] = means
    second[1:, 1:] = raw
    xy = np.concatenate(([0.0], coefficients))
    return PopulationMoments(second, xy, 1.0 + environment.y_noise_variance)


def mixture_moments(environments: tuple[RelevanceEnvironment, ...]) -> PopulationMoments:
    if not environments:
        raise ValueError("at least one source environment is required")
    values = [moments(environment) for environment in environments]
    return PopulationMoments(
        np.mean([value.second for value in values], axis=0),
        np.mean([value.xy for value in values], axis=0),
        float(np.mean([value.y2 for value in values])),
    )


def source_optimum(environments: tuple[RelevanceEnvironment, ...]) -> tuple[Array, PopulationMoments]:
    source = mixture_moments(environments)
    return np.linalg.solve(source.second, source.xy), source


def risk(weights: Array, population: PopulationMoments) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w @ population.second @ w - 2.0 * w @ population.xy + population.y2)


def source_excess(weights: Array, optimum: Array, source: PopulationMoments) -> float:
    delta = np.asarray(weights, dtype=float) - optimum
    return float(delta @ source.second @ delta)


def hessian(source: PopulationMoments) -> Array:
    return 2.0 * source.second


def feature_residual_coupling(weights: Array, population: PopulationMoments) -> Array:
    """Return ``E[X(Y-X'w)]`` in the supplied environment."""
    return population.xy - population.second @ np.asarray(weights, dtype=float)


def risk_difference_gradient(optimum: Array, source: PopulationMoments, target: PopulationMoments) -> Array:
    delta_second = target.second - source.second
    delta_xy = target.xy - source.xy
    return 2.0 * delta_second @ optimum - 2.0 * delta_xy
