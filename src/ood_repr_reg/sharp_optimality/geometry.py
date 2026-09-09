"""Family-metric transport and sharp affine-policy certificates.

This layer is deliberately a thin metric-aware wrapper around the existing
3E-B shifted-ball solver and 3E-C response decomposition.  In particular,
``E`` always denotes ``A_rec + Pi O``.
"""

from __future__ import annotations

import numpy as np

from ..round3r_3e_c_optimality import full_affine_certificate, response_side_orthogonality
from ..round3r_3e_c_spectral import decompose_response
from ..round3r_3e_joint_regret import shifted_ball_maximum

Array = np.ndarray


def inverse_sqrt_metric(metric: Array | None, dimension: int) -> Array:
    if metric is None:
        return np.eye(dimension)
    value = np.asarray(metric, dtype=float)
    if value.shape != (dimension, dimension):
        raise ValueError("metric has incompatible world dimension")
    values, vectors = np.linalg.eigh((value + value.T) / 2.0)
    if values.min(initial=1.0) <= 0.0:
        raise ValueError("metric must be positive definite")
    return vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T


def metric_whiten(response: Array, observation: Array, adaptive: Array | None = None,
                  metric: Array | None = None) -> tuple[Array, Array, Array | None, Array]:
    """Return maps in Euclidean coordinates for ``u.T G u <= 1``."""
    a, o = np.asarray(response, dtype=float), np.asarray(observation, dtype=float)
    if a.ndim != 2 or o.ndim != 2 or a.shape[1] != o.shape[1]:
        raise ValueError("response and observation must share world dimension")
    root = inverse_sqrt_metric(metric, a.shape[1])
    t = None if adaptive is None else np.asarray(adaptive, dtype=float) @ root
    return a @ root, o @ root, t, root


def response_parts(response: Array, observation: Array, adaptive: Array | None = None,
                   metric: Array | None = None, tolerance: float = 1e-10) -> dict[str, object]:
    """Decompose response in the declared family metric.

    ``adaptive`` is the world map ``Pi O`` before whitening.  The returned
    residual is exactly ``A_rec + Pi O`` in whitened coordinates.
    """
    a, o, pi_o, root = metric_whiten(response, observation, adaptive, metric)
    pieces = decompose_response(a, o, tolerance)
    rec = np.asarray(pieces["A_recoverable"])
    irr = np.asarray(pieces["A_irreducible"])
    e = rec if pi_o is None else rec + pi_o
    return {
        **pieces, "A": a, "O": o, "PiO": pi_o,
        "R": irr, "E": e, "metric_inverse_sqrt": root,
        "decomposition_residual": float(np.linalg.norm(a - irr - rec)),
    }


def affine_policy_audit(z0: Array, response: Array, observation: Array,
                        adaptive: Array, metric: Array | None = None,
                        tolerance: float = 1e-10) -> dict[str, object]:
    """Evaluate the sharp theorem for one concrete affine policy."""
    parts = response_parts(response, observation, adaptive, metric, tolerance)
    z = np.asarray(z0, dtype=float)
    certificate = full_affine_certificate(z, parts["A"], parts["O"], parts["E"], tolerance)
    adaptive_only = shifted_ball_maximum(np.zeros_like(z), np.asarray(parts["R"]) + np.asarray(parts["E"]))
    total = shifted_ball_maximum(z, np.asarray(parts["R"]) + np.asarray(parts["E"]))
    orth = response_side_orthogonality(parts["A"], parts["O"], parts["E"], tolerance)
    return {
        **certificate,
        "adaptive_regret": float(adaptive_only.value),
        "total_regret": float(total.value),
        "adaptive_excess": float(adaptive_only.value - certificate["information_floor"]),
        "static_plus_interaction": float(total.value - adaptive_only.value),
        "R_operator_norm": float(np.linalg.svd(parts["R"], compute_uv=False)[0]) if np.asarray(parts["R"]).size else 0.0,
        "E_operator_norm": float(np.linalg.svd(parts["E"], compute_uv=False)[0]) if np.asarray(parts["E"]).size else 0.0,
        "PiO_operator_norm": float(np.linalg.svd(parts["PiO"], compute_uv=False)[0]) if parts["PiO"] is not None and np.asarray(parts["PiO"]).size else 0.0,
        "decomposition_residual": parts["decomposition_residual"],
        "orthogonality": orth,
        "metric_used": metric is not None,
    }


def transported_coordinate_audit(z0: Array, response: Array, observation: Array,
                                 adaptive: Array, metric: Array, transform: Array,
                                 tolerance: float = 1e-9) -> dict[str, object]:
    """Check invariance under ``u=T u_tilde, G_tilde=T.T G T``."""
    base = affine_policy_audit(z0, response, observation, adaptive, metric, tolerance)
    t = np.asarray(transform, dtype=float)
    transformed = affine_policy_audit(
        z0, np.asarray(response) @ t, np.asarray(observation) @ t,
        np.asarray(adaptive) @ t, t.T @ np.asarray(metric) @ t, tolerance,
    )
    keys = ("information_floor", "adaptive_regret", "total_regret", "slack_ratio")
    errors = {
        key: 0.0 if np.isinf(base[key]) and np.isinf(transformed[key])
        else float(abs(base[key] - transformed[key])) for key in keys
    }
    return {"base": base, "transformed": transformed, "errors": errors,
            "pass": bool(max(errors.values(), default=0.0) <= tolerance * 100)}


__all__ = ["inverse_sqrt_metric", "metric_whiten", "response_parts", "affine_policy_audit", "transported_coordinate_audit"]
