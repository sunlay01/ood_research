"""Population Gaussian benchmark for 3B response-module experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MomentState:
    second: Array
    xy: Array
    y2: float = 1.0

    def __add__(self, other: "MomentState") -> "MomentState":
        return MomentState(self.second + other.second, self.xy + other.xy, self.y2 + other.y2)

    def __sub__(self, other: "MomentState") -> "MomentState":
        return MomentState(self.second - other.second, self.xy - other.xy, self.y2 - other.y2)

    def scaled(self, factor: float) -> "MomentState":
        return MomentState(self.second * factor, self.xy * factor, self.y2 * factor)


@dataclass(frozen=True)
class ModuleEnvironment:
    """Y~N(0,1), with noisy C, shortcuts, nuisances, and emergent U."""

    shortcut_rhos: tuple[float, ...] = (0.75, 0.45)
    shortcut_means: tuple[float, ...] = (0.18, -0.12)
    shortcut_variances: tuple[float, ...] = (0.49, 0.64)
    n_noise: int = 4
    noise_means: Array | None = None
    noise_variances: Array | None = None
    c_noise_variance: float = 0.81
    u_gamma: float = 0.0
    u_noise_variance: float = 0.72

    def __post_init__(self) -> None:
        q = len(self.shortcut_rhos)
        if q == 0 or len(self.shortcut_means) != q or len(self.shortcut_variances) != q:
            raise ValueError("shortcut fields must have the same nonzero length")
        if self.n_noise < 0 or self.c_noise_variance < 0 or self.u_noise_variance < 0:
            raise ValueError("invalid noise configuration")
        means = np.zeros(self.n_noise) if self.noise_means is None else np.asarray(self.noise_means, dtype=float)
        variances = np.ones(self.n_noise) if self.noise_variances is None else np.asarray(self.noise_variances, dtype=float)
        if means.shape != (self.n_noise,) or variances.shape != (self.n_noise,):
            raise ValueError("noise arrays have incompatible shape")
        if np.any(variances < 0) or np.any(np.asarray(self.shortcut_variances) < 0):
            raise ValueError("variances must be nonnegative")
        object.__setattr__(self, "noise_means", means.copy())
        object.__setattr__(self, "noise_variances", variances.copy())

    @property
    def dimension(self) -> int:
        return 1 + 1 + len(self.shortcut_rhos) + self.n_noise + 1

    @property
    def feature_names(self) -> tuple[str, ...]:
        return ("intercept", "C", *(f"S{i+1}" for i in range(len(self.shortcut_rhos))),
                *(f"N{i+1}" for i in range(self.n_noise)), "U")

    def updated(self, **changes: object) -> "ModuleEnvironment":
        return replace(self, **changes)


@dataclass(frozen=True)
class ShiftProbe:
    shift_id: str
    target: MomentState
    mechanism: str
    intervention_family: str
    magnitude: float
    pure: bool = True
    pair_id: str | None = None


def environment_state(environment: ModuleEnvironment) -> MomentState:
    q = len(environment.shortcut_rhos)
    p = environment.dimension - 1
    loading = np.zeros(p)
    loading[0] = 1.0
    loading[1 : 1 + q] = environment.shortcut_rhos
    loading[-1] = environment.u_gamma
    means = np.zeros(p)
    means[1 : 1 + q] = environment.shortcut_means
    means[1 + q : 1 + q + environment.n_noise] = environment.noise_means
    covariance = np.outer(loading, loading)
    covariance[0, 0] += environment.c_noise_variance
    covariance[1 : 1 + q, 1 : 1 + q] += np.diag(environment.shortcut_variances)
    offset = 1 + q
    covariance[offset : offset + environment.n_noise, offset : offset + environment.n_noise] += np.diag(environment.noise_variances)
    covariance[-1, -1] += environment.u_noise_variance
    second = np.zeros((p + 1, p + 1))
    second[0, 0] = 1.0
    second[0, 1:] = means
    second[1:, 0] = means
    second[1:, 1:] = covariance + np.outer(means, means)
    return MomentState(second, np.concatenate(([0.0], loading)), 1.0)


def nonlinear_environment_state(environment: ModuleEnvironment,
                                shortcut_alphas: tuple[float, ...] | None = None) -> MomentState:
    """Exact second moments for ``S_j = rho_j Y + alpha_j(Y^2-1) + noise``.

    This is a stress family for the response geometry.  It is deliberately
    represented only through exact moments: the primary 3B claims remain
    linear population-risk claims, while this helper checks whether their
    numerical geometry survives a non-Gaussian generator.
    """
    q = len(environment.shortcut_rhos)
    alphas = tuple(0.08 * (index + 1) for index in range(q)) if shortcut_alphas is None else tuple(shortcut_alphas)
    if len(alphas) != q:
        raise ValueError("shortcut_alphas must match shortcut_rhos")
    p = environment.dimension - 1
    means = np.zeros(p)
    means[1 : 1 + q] = environment.shortcut_means
    means[1 + q : 1 + q + environment.n_noise] = environment.noise_means
    means[-1] = 0.0
    # Build every linear Y loading before forming the common covariance.
    # The quadratic shortcut term is uncorrelated with Y, but it does not
    # remove the rho_j Y contribution to Var(S_j), Cov(C, S_j), or Cov(U, S_j).
    loading = np.zeros(p)
    loading[0] = 1.0
    loading[1 : 1 + q] = environment.shortcut_rhos
    loading[-1] = environment.u_gamma
    covariance = np.outer(loading, loading)
    covariance[0, 0] += environment.c_noise_variance
    # E[(Y^2-1)^2] = 2 and E[Y(Y^2-1)] = 0 for Y~N(0,1).
    for index, (rho, alpha, variance) in enumerate(zip(environment.shortcut_rhos, alphas, environment.shortcut_variances)):
        coordinate = 1 + index
        covariance[coordinate, coordinate] += 2.0 * alpha * alpha + variance
    offset = 1 + q
    covariance[offset : offset + environment.n_noise, offset : offset + environment.n_noise] += np.diag(environment.noise_variances)
    covariance[-1, -1] += environment.u_noise_variance
    for index in range(q):
        for other in range(q):
            if index != other:
                covariance[1 + index, 1 + other] += 2.0 * alphas[index] * alphas[other]
    # The linear C and U loadings retain their covariance with Y; the
    # quadratic term has zero covariance with Y and hence does not enter xy.
    second = np.zeros((p + 1, p + 1))
    second[0, 0] = 1.0
    second[0, 1:] = means
    second[1:, 0] = means
    second[1:, 1:] = covariance + np.outer(means, means)
    return MomentState(second, np.concatenate(([0.0], loading)), 1.0)


def mixture_state(environments: tuple[ModuleEnvironment, ...]) -> MomentState:
    if not environments:
        raise ValueError("at least one source environment is required")
    states = [environment_state(env) for env in environments]
    return MomentState(np.mean([s.second for s in states], axis=0),
                       np.mean([s.xy for s in states], axis=0),
                       float(np.mean([s.y2 for s in states])))


def source_design(base: ModuleEnvironment, relation_exposed: bool = True) -> tuple[ModuleEnvironment, ...]:
    envs = [base]
    if relation_exposed:
        for index in range(len(base.shortcut_rhos)):
            plus = list(base.shortcut_rhos)
            minus = list(base.shortcut_rhos)
            plus[index] += 0.08
            minus[index] -= 0.08
            envs.extend((base.updated(shortcut_rhos=tuple(plus)), base.updated(shortcut_rhos=tuple(minus))))
    return tuple(envs)


def source_optimum(environments: tuple[ModuleEnvironment, ...]) -> tuple[Array, MomentState]:
    state = mixture_state(environments)
    return np.linalg.solve(state.second, state.xy), state


def risk(weights: Array, state: MomentState) -> float:
    w = np.asarray(weights, dtype=float)
    return float(w @ state.second @ w - 2 * w @ state.xy + state.y2)


def risk_gradient(optimum: Array, source: MomentState, target: MomentState) -> Array:
    return 2 * (target.second - source.second) @ optimum - 2 * (target.xy - source.xy)


def make_probe_set(base: ModuleEnvironment, source: MomentState,
                   magnitudes: tuple[float, ...] = (0.25, 0.5, 1.0),
                   include_noise: bool = True,
                   shortcut_groups: tuple[tuple[int, ...], ...] | None = None) -> tuple[ShiftProbe, ...]:
    """Generate pure Gaussian probes; labels are metadata, never discovery input."""
    probes: list[ShiftProbe] = []
    base_state = environment_state(base)

    def target_from_environment(environment: ModuleEnvironment) -> MomentState:
        """Apply only the requested environment perturbation to source moments.

        The source design may itself be a mixture of environments.  Using a
        raw single-environment target would silently add the source-to-base
        change to every probe.  Moment translation keeps each probe's intended
        perturbation separate from that background design.
        """
        delta = environment_state(environment) - base_state
        return source + delta
    groups = shortcut_groups or tuple((index,) for index in range(len(base.shortcut_rhos)))
    for group_index, group in enumerate(groups):
        index = group[0]
        for magnitude in magnitudes:
            for sign in (-1.0, 1.0):
                rhos = list(base.shortcut_rhos)
                for member in group:
                    rhos[member] += sign * 0.20 * magnitude
                target = target_from_environment(base.updated(shortcut_rhos=tuple(rhos)))
                probes.append(ShiftProbe(f"S{group_index+1}-relation-{sign:+.0f}-{magnitude}", target, f"S{group_index+1}", "relation", magnitude, pair_id=f"S{group_index+1}-relation-{magnitude}"))
                means = list(base.shortcut_means)
                for member in group:
                    means[member] += sign * 0.35 * magnitude
                target = target_from_environment(base.updated(shortcut_means=tuple(means)))
                probes.append(ShiftProbe(f"S{group_index+1}-mean-{sign:+.0f}-{magnitude}", target, f"S{group_index+1}", "mean", magnitude, pair_id=f"S{group_index+1}-mean-{magnitude}"))
            variances = list(base.shortcut_variances)
            for member in group:
                variances[member] += 0.30 * magnitude
            target = target_from_environment(base.updated(shortcut_variances=tuple(variances)))
            probes.append(ShiftProbe(f"S{group_index+1}-variance-{magnitude}", target, f"S{group_index+1}", "variance", magnitude))
    for magnitude in magnitudes:
        target = target_from_environment(base.updated(u_gamma=0.75 * magnitude))
        probes.append(ShiftProbe(f"U-emergent-{magnitude}", target, "U", "emergent", magnitude))
    if include_noise and base.n_noise:
        for index in range(min(2, base.n_noise)):
            variances = np.asarray(base.noise_variances).copy()
            variances[index] += 0.5
            probes.append(ShiftProbe(f"N{index+1}-variance", target_from_environment(base.updated(noise_variances=variances)), "noise", "variance", 1.0))
    del base_state, source
    return tuple(probes)


def mixed_probe(probes: tuple[ShiftProbe, ...], names: tuple[str, ...], source: MomentState) -> ShiftProbe:
    selected = [next(p for p in probes if p.shift_id == name) for name in names]
    # Moment-level additive target construction is intentional: it isolates
    # response additivity from nonlinear environment parameterization effects.
    target = source + sum((p.target - source for p in selected), MomentState(np.zeros_like(source.second), np.zeros_like(source.xy), 0.0))
    return ShiftProbe("+".join(names), target, "+".join(p.mechanism for p in selected), "mixed", 1.0, False)
