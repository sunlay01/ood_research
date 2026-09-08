"""Controlled high-dimensional structural environments for retry experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

Array = np.ndarray


def _spd(matrix: Array, floor: float = 0.08) -> Array:
    matrix = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    values, vectors = np.linalg.eigh(matrix)
    return vectors @ np.diag(np.maximum(values, floor)) @ vectors.T


@dataclass(frozen=True)
class PopulationEnvironment:
    mu_c: Array
    sigma_c: Array
    gamma: Array
    b: Array
    mu_xi: Array
    sigma_xi: Array
    beta: Array
    noise_variance: float = 0.25
    family: str = "linear-gaussian"
    quadratic_alpha: Array | None = None

    def __post_init__(self) -> None:
        mu_c = np.asarray(self.mu_c, dtype=float)
        sigma_c = _spd(self.sigma_c)
        gamma = np.asarray(self.gamma, dtype=float)
        b = np.asarray(self.b, dtype=float)
        mu_xi = np.asarray(self.mu_xi, dtype=float)
        sigma_xi = _spd(self.sigma_xi)
        beta = np.asarray(self.beta, dtype=float)
        if mu_c.ndim != 1 or gamma.shape != (b.size, mu_c.size):
            raise ValueError("incompatible core/nuisance dimensions")
        if sigma_c.shape != (mu_c.size, mu_c.size) or sigma_xi.shape != (b.size, b.size):
            raise ValueError("invalid covariance shape")
        if beta.shape != mu_c.shape or mu_xi.shape != b.shape:
            raise ValueError("invalid task or innovation vector")
        object.__setattr__(self, "mu_c", mu_c)
        object.__setattr__(self, "sigma_c", sigma_c)
        object.__setattr__(self, "gamma", gamma)
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

    def with_delta(self, delta: dict[str, Array], scale: float = 1.0) -> "PopulationEnvironment":
        return replace(
            self,
            mu_c=self.mu_c + scale * delta.get("mu_c", np.zeros_like(self.mu_c)),
            sigma_c=_spd(self.sigma_c + scale * delta.get("sigma_c", np.zeros_like(self.sigma_c))),
            gamma=self.gamma + scale * delta.get("gamma", np.zeros_like(self.gamma)),
            b=self.b + scale * delta.get("b", np.zeros_like(self.b)),
            mu_xi=self.mu_xi + scale * delta.get("mu_xi", np.zeros_like(self.mu_xi)),
            sigma_xi=_spd(self.sigma_xi + scale * delta.get("sigma_xi", np.zeros_like(self.sigma_xi))),
            beta=self.beta + scale * delta.get("beta", np.zeros_like(self.beta)),
        )


@dataclass(frozen=True)
class ShiftProbe:
    shift_id: str
    target: PopulationEnvironment
    kind: str
    pair_id: str | None
    sign: int
    magnitude: float
    hidden_metadata: dict[str, object]


def base_environment(c_dim: int = 2, a_dim: int = 2) -> PopulationEnvironment:
    if (c_dim, a_dim) != (2, 2):
        raise ValueError("retry main track is fixed at d_C=d_A=2")
    return PopulationEnvironment(
        mu_c=np.array([0.35, -0.45]),
        sigma_c=np.array([[1.25, 0.28], [0.28, 0.9]]),
        gamma=np.array([[0.75, 0.22], [-0.31, 0.58]]),
        b=np.array([0.20, -0.18]),
        mu_xi=np.array([0.12, -0.09]),
        sigma_xi=np.array([[0.45, 0.08], [0.08, 0.62]]),
        beta=np.array([1.0, -0.72]),
    )


def source_environments(base: PopulationEnvironment | None = None) -> tuple[PopulationEnvironment, ...]:
    base = base or base_environment()
    zc = np.zeros_like(base.mu_c)
    za = np.zeros_like(base.mu_xi)
    zcc = np.zeros_like(base.sigma_c)
    zaa = np.zeros_like(base.sigma_xi)
    zg = np.zeros_like(base.gamma)
    directions = [
        {"gamma": np.array([[0.18, -0.05], [0.06, 0.14]])},
        {"gamma": -np.array([[0.13, 0.04], [-0.08, 0.16]])},
        {"mu_c": np.array([0.20, -0.12])},
        {"mu_c": -np.array([0.16, 0.10])},
        {"mu_xi": np.array([0.15, -0.11])},
        {"sigma_xi": np.array([[0.12, 0.03], [0.03, -0.06]])},
    ]
    return (base,) + tuple(base.with_delta(direction) for direction in directions)


def _random_delta(rng: np.random.Generator, base: PopulationEnvironment, kind: str) -> dict[str, Array]:
    c, a = base.c_dim, base.a_dim
    def symmetric(scale: float) -> Array:
        raw = rng.normal(size=(c, c))
        return scale * (raw + raw.T) / 2.0
    def symmetric_a(scale: float) -> Array:
        raw = rng.normal(size=(a, a))
        return scale * (raw + raw.T) / 2.0
    delta = {
        "mu_c": np.zeros(c), "sigma_c": np.zeros((c, c)), "gamma": np.zeros((a, c)),
        "b": np.zeros(a), "mu_xi": np.zeros(a), "sigma_xi": np.zeros((a, a)),
        "beta": np.zeros(c),
    }
    choices = {
        "core_mean": ("mu_c",),
        "core_covariance": ("sigma_c",),
        "nuisance_mean": ("b", "mu_xi"),
        "nuisance_covariance": ("sigma_xi",),
        "relation": ("gamma",),
        "task": ("beta",),
        "observation": ("b", "mu_xi", "sigma_xi"),
    }
    active = choices[kind]
    for key in active:
        if key == "mu_c": delta[key] = rng.normal(scale=0.20, size=c)
        elif key == "beta": delta[key] = rng.normal(scale=0.14, size=c)
        elif key in {"b", "mu_xi"}: delta[key] = rng.normal(scale=0.18, size=a)
        elif key == "gamma": delta[key] = rng.normal(scale=0.16, size=(a, c))
        elif key == "sigma_c": delta[key] = symmetric(0.16)
        elif key == "sigma_xi": delta[key] = symmetric_a(0.16)
    # Every probe has a small mixed component so discovery is not a label lookup.
    if kind not in {"task", "observation"}:
        delta["gamma"] += rng.normal(scale=0.035, size=(a, c))
        delta["mu_xi"] += rng.normal(scale=0.035, size=a)
    return delta


def shift_ensemble(base: PopulationEnvironment | None = None, count: int = 150, seed: int = 20260906) -> tuple[ShiftProbe, ...]:
    base = base or base_environment()
    rng = np.random.default_rng(seed)
    kinds = ("core_mean", "core_covariance", "nuisance_mean", "nuisance_covariance", "relation", "task", "observation", "compound")
    probes: list[ShiftProbe] = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        primary = "relation" if kind == "compound" else kind
        delta = _random_delta(rng, base, primary)
        if kind == "compound":
            other = _random_delta(rng, base, kinds[rng.integers(0, 5)])
            delta = {key: delta[key] + 0.7 * other[key] for key in delta}
        pair_id = f"pair-{index:04d}"
        for sign in (-1, 1):
            target = base.with_delta(delta, scale=float(sign))
            probes.append(ShiftProbe(f"shift-{index:04d}{'m' if sign < 0 else 'p'}", target, kind, pair_id, sign, 1.0, {"primary_kind": primary, "source_generated": True}))
    return tuple(probes)


def nonlinear_environment(base: PopulationEnvironment | None = None, alpha: float = 0.18) -> PopulationEnvironment:
    base = base or base_environment()
    return replace(base, family="quadratic-observation", quadratic_alpha=np.full((base.a_dim, base.c_dim), alpha))

