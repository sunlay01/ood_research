"""Linear-Gaussian benchmark for source-optimality-aware OOD relevance.

The benchmark is deliberately separate from the older structural SCM.  Its
features are noisy measurements of a fixed outcome, plus shortcut, emergent,
stable, and independent-noise coordinates.  All calculations are population
calculations; target environments are never used to fit the source optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class RelevanceEnvironment:
    """A Gaussian environment generated from ``Y`` and independent noises."""

    shortcut_rhos: tuple[float, ...] = (0.8,)
    sigma_c: float = 0.9
    sigma_shortcut: tuple[float, ...] = (0.7,)
    n_noise: int = 4
    noise_mean: Array | None = None
    noise_variance: Array | None = None
    u_gamma: float = 0.0
    u_sigma: float = 0.8
    z_gamma: float = 0.75
    z_sigma: float = 0.9
    shortcut_shared_sigma: float = 0.0
    shortcut_noise_covariance: Array | None = None
    y_noise_variance: float = 0.25
    noise_rotation: Array | None = None

    def __post_init__(self) -> None:
        rhos = tuple(float(v) for v in self.shortcut_rhos)
        sigmas = tuple(float(v) for v in self.sigma_shortcut)
        if len(rhos) != len(sigmas) or not rhos:
            raise ValueError("shortcut_rhos and sigma_shortcut must be nonempty and aligned")
        if any(v < 0 for v in sigmas) or self.sigma_c < 0 or self.u_sigma < 0 or self.z_sigma < 0:
            raise ValueError("feature noise scales must be nonnegative")
        if self.n_noise < 0 or self.shortcut_shared_sigma < 0 or self.y_noise_variance < 0:
            raise ValueError("noise dimensions and variances must be nonnegative")
        mean = np.zeros(self.n_noise) if self.noise_mean is None else np.asarray(self.noise_mean, dtype=float)
        variance = np.ones(self.n_noise) if self.noise_variance is None else np.asarray(self.noise_variance, dtype=float)
        if mean.shape != (self.n_noise,) or variance.shape != (self.n_noise,):
            raise ValueError("noise_mean and noise_variance must match n_noise")
        if np.any(variance < 0):
            raise ValueError("noise variances must be nonnegative")
        rotation = None if self.noise_rotation is None else np.asarray(self.noise_rotation, dtype=float)
        if rotation is not None:
            if rotation.shape != (self.n_noise, self.n_noise):
                raise ValueError("noise_rotation has incompatible shape")
            if not np.allclose(rotation.T @ rotation, np.eye(self.n_noise), atol=1e-8):
                raise ValueError("noise_rotation must be orthogonal")
        shortcut_covariance = None if self.shortcut_noise_covariance is None else np.asarray(self.shortcut_noise_covariance, dtype=float)
        if shortcut_covariance is not None:
            q = len(rhos)
            if shortcut_covariance.shape != (q, q) or not np.allclose(shortcut_covariance, shortcut_covariance.T, atol=1e-8):
                raise ValueError("shortcut_noise_covariance has incompatible shape")
            if np.min(np.linalg.eigvalsh(shortcut_covariance)) < -1e-10:
                raise ValueError("shortcut_noise_covariance must be positive semidefinite")
        object.__setattr__(self, "shortcut_rhos", rhos)
        object.__setattr__(self, "sigma_shortcut", sigmas)
        object.__setattr__(self, "noise_mean", mean.copy())
        object.__setattr__(self, "noise_variance", variance.copy())
        object.__setattr__(self, "noise_rotation", None if rotation is None else rotation.copy())
        object.__setattr__(self, "shortcut_noise_covariance", None if shortcut_covariance is None else shortcut_covariance.copy())

    @property
    def feature_names(self) -> tuple[str, ...]:
        shortcuts = tuple(f"S{index + 1}" for index in range(len(self.shortcut_rhos)))
        noises = tuple(f"N{index + 1}" for index in range(self.n_noise))
        return ("intercept", "C", *shortcuts, *noises, "U", "Z")

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def updated(self, **changes: object) -> "RelevanceEnvironment":
        return replace(self, **changes)


def base_environment(
    n_noise: int = 4,
    rho: float = 0.8,
    shortcut_count: int = 1,
    redundant: bool = False,
    noise_rotation: Array | None = None,
) -> RelevanceEnvironment:
    """Return the registered benchmark base point."""
    rhos = tuple(float(rho) for _ in range(shortcut_count))
    sigmas = tuple(0.7 for _ in range(shortcut_count))
    shared = 0.45 if redundant else 0.0
    return RelevanceEnvironment(
        shortcut_rhos=rhos,
        sigma_shortcut=sigmas,
        n_noise=n_noise,
        u_gamma=0.0,
        z_gamma=0.75,
        shortcut_shared_sigma=shared,
        noise_rotation=noise_rotation,
    )


def source_environments(
    base: RelevanceEnvironment,
    exposed: bool = True,
    relation_delta: float = 0.12,
    include_stable_burden_probe: bool = True,
) -> tuple[RelevanceEnvironment, ...]:
    """Create a source design with optional shortcut relation exposure.

    Relation-exposed sources vary one shortcut coordinate at a time.  The
    optional label-noise probe gives an exposure-high but model-discrimination-
    low negative control.
    """
    environments = [base]
    if exposed:
        for index in range(len(base.shortcut_rhos)):
            plus = list(base.shortcut_rhos)
            minus = list(base.shortcut_rhos)
            plus[index] += relation_delta
            minus[index] -= relation_delta
            environments.extend((base.updated(shortcut_rhos=tuple(plus)), base.updated(shortcut_rhos=tuple(minus))))
    if include_stable_burden_probe:
        environments.append(base.updated(y_noise_variance=base.y_noise_variance + 0.20))
    return tuple(environments)


def source_reference(environments: tuple[RelevanceEnvironment, ...]) -> RelevanceEnvironment:
    """Construct a single Gaussian reference matching source second moments.

    Averaging source relation environments changes ``E[rho^2]`` even when
    their mean rho is unchanged.  This reference absorbs that variance so
    isolated target probes do not accidentally contain a source-design shift.
    """
    if not environments:
        raise ValueError("at least one source environment is required")
    first = environments[0]
    rho = np.mean([env.shortcut_rhos for env in environments], axis=0)
    rho_values = np.asarray([env.shortcut_rhos for env in environments], dtype=float)
    noise_covariances = np.asarray([
        np.diag(np.square(env.sigma_shortcut))
        + env.shortcut_shared_sigma**2 * np.ones((len(env.shortcut_rhos), len(env.shortcut_rhos)))
        for env in environments
    ])
    residual_covariance = np.mean(noise_covariances + np.einsum("ni,nj->nij", rho_values, rho_values), axis=0) - np.outer(rho, rho)
    sigma = np.sqrt(np.maximum(np.diag(residual_covariance), 0.0))
    y_noise = float(np.mean([env.y_noise_variance for env in environments]))
    return first.updated(
        shortcut_rhos=tuple(rho),
        sigma_shortcut=tuple(sigma),
        shortcut_shared_sigma=0.0,
        shortcut_noise_covariance=residual_covariance,
        y_noise_variance=y_noise,
    )


def target_shortcut_shift(base: RelevanceEnvironment, rho: float, index: int = 0) -> RelevanceEnvironment:
    values = list(base.shortcut_rhos)
    values[index] = float(rho)
    return base.updated(shortcut_rhos=tuple(values))


def target_noise_shift(base: RelevanceEnvironment, index: int = 0, mean: float = 1.0, variance: float | None = None) -> RelevanceEnvironment:
    means = np.asarray(base.noise_mean, dtype=float).copy()
    variances = np.asarray(base.noise_variance, dtype=float).copy()
    means[index] += mean
    if variance is not None:
        variances[index] = float(variance)
    return base.updated(noise_mean=means, noise_variance=variances)


def target_emergent_shift(base: RelevanceEnvironment, gamma: float = 0.7) -> RelevanceEnvironment:
    return base.updated(u_gamma=float(gamma))


def target_stable_burden_shift(base: RelevanceEnvironment, amount: float = 1.0) -> RelevanceEnvironment:
    return base.updated(y_noise_variance=base.y_noise_variance + float(amount))


def target_shortcut_variance_shift(base: RelevanceEnvironment, index: int = 0, amount: float = 0.8) -> RelevanceEnvironment:
    values = list(base.sigma_shortcut)
    values[index] = np.sqrt(values[index] ** 2 + float(amount))
    covariance = None if base.shortcut_noise_covariance is None else np.asarray(base.shortcut_noise_covariance, dtype=float).copy()
    if covariance is not None:
        covariance[index, index] += float(amount)
    return base.updated(sigma_shortcut=tuple(values), shortcut_noise_covariance=covariance)


def target_noise_variance_shift(base: RelevanceEnvironment, index: int = 0, amount: float = 1.0) -> RelevanceEnvironment:
    variances = np.asarray(base.noise_variance, dtype=float).copy()
    variances[index] += float(amount)
    return base.updated(noise_variance=variances)


def rotate_noise_environment(base: RelevanceEnvironment, seed: int) -> RelevanceEnvironment:
    if base.n_noise == 0:
        return base
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(base.n_noise, base.n_noise)))
    q *= np.sign(np.diag(r))[None, :]
    return base.updated(noise_rotation=q)
