"""Joint information--regularization affine regret for 3E-B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .round3r_3e_recovery import minimax_recovery_error, nullspace_basis, operator_norm

Array = np.ndarray


@dataclass(frozen=True)
class TrustRegionResult:
    value: float
    maximizer: Array
    multiplier: float
    status: str


def _symmetric(matrix: Array) -> Array:
    return (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0


def shifted_ball_maximum(offset: Array, linear_map: Array, tolerance: float = 1e-11) -> TrustRegionResult:
    """Maximize ``0.5*||offset + linear_map @ u||^2`` on ``||u|| <= 1``.

    The eigensystem of ``B.T B`` reduces the boundary equation to a secular
    equation.  The repeated-top-eigenvalue hard case is handled explicitly.
    """
    c = np.asarray(offset, dtype=float)
    b = np.asarray(linear_map, dtype=float)
    if c.ndim != 1 or b.ndim != 2 or b.shape[0] != c.size:
        raise ValueError("offset and linear_map have incompatible shapes")
    dimension = b.shape[1]
    if dimension == 0:
        return TrustRegionResult(0.5 * float(c @ c), np.zeros(0), 0.0, "empty_world")
    q = _symmetric(b.T @ b)
    d = b.T @ c
    eigenvalues, eigenvectors = np.linalg.eigh(q)
    qmax = float(eigenvalues[-1])
    d_eigen = eigenvectors.T @ d
    scale = max(1.0, abs(qmax), float(np.linalg.norm(d)))
    top = np.flatnonzero(np.abs(eigenvalues - qmax) <= tolerance * scale)

    def norm_at(multiplier: float) -> float:
        denominators = multiplier - eigenvalues
        values = np.divide(d_eigen, denominators, out=np.zeros_like(d_eigen), where=np.abs(denominators) > tolerance * scale)
        return float(np.linalg.norm(values))

    top_linear = float(np.linalg.norm(d_eigen[top])) if top.size else 0.0
    if top_linear <= tolerance * scale:
        lower = np.ones_like(d_eigen, dtype=bool)
        lower[top] = False
        lower_values = np.divide(d_eigen[lower], qmax - eigenvalues[lower],
                                 out=np.zeros(np.sum(lower)), where=np.abs(qmax - eigenvalues[lower]) > tolerance * scale)
        lower_norm = float(np.linalg.norm(lower_values))
        if lower_norm <= 1.0 + tolerance:
            coords = np.zeros(dimension)
            coords[lower] = lower_values
            remaining = max(0.0, 1.0 - lower_norm ** 2)
            coords[top[0]] = np.sqrt(remaining)
            u = eigenvectors @ coords
            return TrustRegionResult(0.5 * float(np.linalg.norm(c + b @ u) ** 2), u, qmax, "hard_case")

    lower_bound = max(qmax, 0.0) + max(1e-12, 1e-12 * scale)
    high = max(lower_bound * 2.0, 1.0)
    while norm_at(high) > 1.0:
        high *= 2.0
    low = lower_bound
    for _ in range(200):
        mid = 0.5 * (low + high)
        if norm_at(mid) > 1.0:
            low = mid
        else:
            high = mid
    multiplier = high
    coords = np.divide(d_eigen, multiplier - eigenvalues)
    norm = np.linalg.norm(coords)
    if norm > 0.0:
        coords /= norm
    u = eigenvectors @ coords
    return TrustRegionResult(0.5 * float(np.linalg.norm(c + b @ u) ** 2), u, multiplier, "secular")


def independent_shifted_ball_maximum(offset: Array, linear_map: Array,
                                     starts: int = 32, seed: int = 0) -> dict[str, object]:
    """Independent numerical cross-check using constrained multi-start SLSQP."""
    try:
        from scipy.optimize import minimize
    except ImportError:  # pragma: no cover
        return {"available": False, "value": float("nan"), "max_error": float("nan")}
    c, b = np.asarray(offset, dtype=float), np.asarray(linear_map, dtype=float)
    rng = np.random.default_rng(seed)
    points = [np.zeros(b.shape[1])]
    points.extend([rng.normal(size=b.shape[1]) for _ in range(max(0, starts - 1))])
    points = [point / max(1.0, np.linalg.norm(point)) for point in points]
    values = []
    for point in points:
        result = minimize(lambda u: -0.5 * np.linalg.norm(c + b @ u) ** 2, point,
                          method="SLSQP", constraints={"type": "ineq", "fun": lambda u: 1.0 - u @ u},
                          options={"ftol": 1e-12, "maxiter": 2000})
        if result.success or result.x @ result.x <= 1.0 + 1e-7:
            values.append(-float(result.fun))
    return {"available": True, "value": max(values) if values else float("nan"),
            "starts": len(points), "successful": len(values)}


def information_floor(response: Array, observation: Array, tolerance: float = 1e-10) -> float:
    return 0.5 * minimax_recovery_error(response, observation, tolerance) ** 2


def recoverable_and_irreducible(response: Array, observation: Array,
                                 tolerance: float = 1e-10) -> tuple[Array, Array]:
    a, o = np.asarray(response, dtype=float), np.asarray(observation, dtype=float)
    null = nullspace_basis(o, tolerance)
    irreducible = a @ null @ null.T if null.shape[1] else np.zeros_like(a)
    return a - irreducible, irreducible


def affine_regret(offset: Array, response: Array, adaptive: Array) -> TrustRegionResult:
    return shifted_ball_maximum(offset, response + adaptive)


def regret_decomposition(z0: Array, response: Array, adaptive: Array,
                         observation: Array) -> dict[str, object]:
    total = affine_regret(z0, response, adaptive)
    static = affine_regret(z0, response, np.zeros_like(adaptive))
    adaptive_only = affine_regret(np.zeros_like(z0), response, adaptive)
    baseline = affine_regret(np.zeros_like(z0), response, np.zeros_like(adaptive))
    recoverable, irreducible = recoverable_and_irreducible(response, observation)
    residual = recoverable + adaptive
    return {
        "total_regret": total.value,
        "static_only_regret": static.value,
        "adaptive_only_regret": adaptive_only.value,
        "baseline_regret": baseline.value,
        "interaction": total.value - static.value - adaptive_only.value + baseline.value,
        "information_floor": 0.5 * operator_norm(irreducible) ** 2,
        "A_irreducible": irreducible,
        "A_recoverable": recoverable,
        "recoverable_residual": residual,
        "recoverable_residual_operator_norm": operator_norm(residual),
        "total_status": total.status,
        "static_status": static.status,
        "adaptive_status": adaptive_only.status,
    }


def metric_regret_decomposition(z0: Array, response: Array, adaptive: Array,
                                observation: Array, metric: Array | None = None) -> dict[str, object]:
    """Evaluate the same affine regret after transporting a tangent metric.

    ``metric`` declares the norm on world coordinates.  The policy matrices
    are right-multiplied by its inverse square root, so the existing Euclidean
    shifted-ball solver is used without changing its contract.
    """
    if metric is None:
        return regret_decomposition(z0, response, adaptive, observation)
    value = np.asarray(metric, dtype=float)
    if value.ndim != 2 or value.shape != (response.shape[1], response.shape[1]):
        raise ValueError("metric has incompatible world dimension")
    value = (value + value.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if eigenvalues.size and eigenvalues.min() <= 0.0:
        raise ValueError("metric must be positive definite")
    inverse_root = eigenvectors @ np.diag(
        1.0 / np.sqrt(eigenvalues),
    ) @ eigenvectors.T
    return regret_decomposition(
        z0, np.asarray(response) @ inverse_root,
        np.asarray(adaptive) @ inverse_root, np.asarray(observation) @ inverse_root,
    )


__all__ = [
    "TrustRegionResult", "shifted_ball_maximum", "independent_shifted_ball_maximum",
    "information_floor", "recoverable_and_irreducible", "affine_regret",
    "regret_decomposition", "metric_regret_decomposition",
]
