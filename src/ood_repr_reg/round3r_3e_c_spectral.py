"""Spectral geometry for the 3E-C sharp optimality audit.

The functions in this module operate only on the finite-dimensional affine
objects ``A``, ``O`` and ``Pi``.  They deliberately do not attach semantic
names to response or kernel directions.
"""

from __future__ import annotations

import numpy as np

from .round3r_3e_recovery import (
    nullspace_basis,
    operator_norm,
    orthogonal_projector_from_basis,
)

Array = np.ndarray


def _pair(response: Array, observation: Array) -> tuple[Array, Array]:
    a = np.asarray(response, dtype=float)
    o = np.asarray(observation, dtype=float)
    if a.ndim != 2 or o.ndim != 2 or a.shape[1] != o.shape[1]:
        raise ValueError("response and observation must be compatible matrices")
    return a, o


def decompose_response(response: Array, observation: Array, tolerance: float = 1e-10) -> dict[str, object]:
    """Return ``P``, ``Q``, ``A_irr`` and ``A_rec`` for ``ker(O)``."""
    a, o = _pair(response, observation)
    n = nullspace_basis(o, tolerance)
    p = orthogonal_projector_from_basis(n)
    q = np.eye(o.shape[1]) - p
    # The response-side split is determined by source observation geometry,
    # never by a regularizer or target quantity.
    return {
        "P": p,
        "Q": q,
        "null_basis": n,
        "A_irreducible": a @ p,
        "A_recoverable": a @ q,
        "kernel_dimension": int(n.shape[1]),
        "tolerance": float(tolerance),
    }


def spectral_slack(response: Array, observation: Array, tolerance: float = 1e-10) -> dict[str, object]:
    """Compute ``S = alpha^2 I - A_irr A_irr.T`` and its spectrum."""
    parts = decompose_response(response, observation, tolerance)
    irr = np.asarray(parts["A_irreducible"])
    alpha = operator_norm(irr)
    gram = (irr @ irr.T + (irr @ irr.T).T) / 2.0
    slack = (alpha * alpha) * np.eye(irr.shape[0]) - gram
    slack = (slack + slack.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(slack)
    scale = max(1.0, alpha * alpha)
    rank = int(np.sum(eigenvalues > tolerance * scale))
    nullity = int(eigenvalues.size - rank)
    top_value = float(eigenvalues.max()) if eigenvalues.size else 0.0
    top_mask = np.abs(eigenvalues - top_value) <= tolerance * scale
    return {
        **parts,
        "alpha": float(alpha),
        "irreducible_gram": gram,
        "slack": slack,
        "slack_eigenvalues": eigenvalues,
        "slack_eigenvectors": eigenvectors,
        "slack_rank": rank,
        "slack_nullity": nullity,
        "slack_top_eigenspace": eigenvectors[:, top_mask],
    }


def slack_ratio(irreducible: Array, recoverable: Array, tolerance: float = 1e-10) -> dict[str, object]:
    """Measure ``E E*`` against spectral slack.

    The ratio is finite only if the range of ``E E*`` is contained in the
    range of the slack operator.  In finite dimensions its value is
    ``||S^(-1/2) E||_op^2`` on that support.
    """
    irr = np.asarray(irreducible, dtype=float)
    e = np.asarray(recoverable, dtype=float)
    if irr.ndim != 2 or e.ndim != 2 or irr.shape[0] != e.shape[0]:
        raise ValueError("irreducible and recoverable operators must share response dimension")
    alpha = operator_norm(irr)
    gram = (irr @ irr.T + (irr @ irr.T).T) / 2.0
    slack = (alpha * alpha) * np.eye(irr.shape[0]) - gram
    slack = (slack + slack.T) / 2.0
    values, vectors = np.linalg.eigh(slack)
    scale = max(1.0, alpha * alpha, operator_norm(e) ** 2)
    support = values > tolerance * scale
    null = vectors[:, ~support]
    ee = (e @ e.T + (e @ e.T).T) / 2.0
    compatibility_residual = float(np.linalg.norm(null.T @ e)) if null.shape[1] else 0.0
    compatible = compatibility_residual <= tolerance * max(1.0, np.linalg.norm(ee))
    if not compatible:
        ratio = float("inf")
    elif not np.any(support):
        ratio = 0.0 if operator_norm(e) <= tolerance else float("inf")
    else:
        inv_root = vectors[:, support] @ np.diag(1.0 / np.sqrt(values[support])) @ vectors[:, support].T
        ratio = float(operator_norm(inv_root @ e) ** 2)
    return {
        "ratio": ratio,
        "support_compatible": bool(compatible),
        "compatibility_residual": compatibility_residual,
        "tolerance": float(tolerance),
    }


def _psd_min(matrix: Array) -> float:
    value = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    return float(np.linalg.eigvalsh(value).min()) if value.size else 0.0


def adaptive_minimax_certificate(irreducible: Array, recoverable: Array,
                                 tolerance: float = 1e-10) -> dict[str, object]:
    """Certify the exact adaptive-only spectral condition."""
    irr = np.asarray(irreducible, dtype=float)
    e = np.asarray(recoverable, dtype=float)
    if irr.ndim != 2 or e.ndim != 2 or irr.shape[0] != e.shape[0]:
        raise ValueError("operators must share response dimension")
    alpha = operator_norm(irr)
    slack = (alpha * alpha) * np.eye(irr.shape[0]) - irr @ irr.T
    slack = (slack + slack.T) / 2.0
    excess = e @ e.T
    excess = (excess + excess.T) / 2.0
    gap = slack - excess
    ratio = slack_ratio(irr, e, tolerance)
    adaptive_regret = 0.5 * operator_norm(irr + e) ** 2
    floor = 0.5 * alpha * alpha
    scale = max(1.0, alpha * alpha, operator_norm(e) ** 2)
    return {
        "alpha": float(alpha),
        "information_floor": float(floor),
        "adaptive_regret": float(adaptive_regret),
        "slack": slack,
        "slack_minus_EE_star": gap,
        "psd_min_gap": _psd_min(gap),
        "condition_holds": bool(_psd_min(gap) >= -tolerance * scale),
        "recoverable_exact": bool(operator_norm(e) <= tolerance * max(1.0, alpha)),
        "slack_ratio": ratio["ratio"],
        "support_compatible": ratio["support_compatible"],
        "A_irreducible": irr,
        "E": e,
        "tolerance": float(tolerance),
    }


def static_steering_lower_bound(z0: Array, response: Array, observation: Array,
                                tolerance: float = 1e-10) -> dict[str, object]:
    """Return the finite-dimensional static steering tax lower bound."""
    from .round3r_3e_joint_regret import shifted_ball_maximum

    z = np.asarray(z0, dtype=float)
    a, o = _pair(response, observation)
    parts = spectral_slack(a, o, tolerance)
    full = shifted_ball_maximum(z, a)
    floor = 0.5 * float(parts["alpha"]) ** 2
    lower = floor + 0.5 * float(z @ z)
    gap = max(0.0, full.value - floor)
    denominator = 0.5 * float(z @ z)
    return {
        "total_regret": float(full.value),
        "information_floor": floor,
        "static_tax_lower_bound": lower,
        "total_excess": gap,
        "static_tax_holds": bool(full.value + tolerance >= lower),
        "tightness_ratio": None if denominator <= tolerance else float(gap / denominator),
        "maximizer": full.maximizer,
        "status": full.status,
    }


def full_minimax_certificate(z0: Array, irreducible: Array, recoverable: Array,
                             tolerance: float = 1e-10) -> dict[str, object]:
    """Certify the full affine iff condition numerically."""
    z = np.asarray(z0, dtype=float)
    adaptive = adaptive_minimax_certificate(irreducible, recoverable, tolerance)
    zero_static = bool(operator_norm(z.reshape(-1, 1)) <= tolerance)
    optimal = bool(zero_static and adaptive["condition_holds"])
    return {
        **adaptive,
        "z0_norm": float(np.linalg.norm(z)),
        "zero_static": zero_static,
        "full_minimax_condition": optimal,
    }


def complete_information_certificate(response: Array, observation: Array,
                                     tolerance: float = 1e-10) -> dict[str, object]:
    """Audit the ``alpha=0`` complete-information boundary."""
    parts = decompose_response(response, observation, tolerance)
    irr = np.asarray(parts["A_irreducible"])
    alpha = operator_norm(irr)
    return {
        "alpha": float(alpha),
        "A_irreducible_norm": float(alpha),
        "complete_information": bool(alpha <= tolerance * max(1.0, operator_norm(response))),
        "kernel_dimension": parts["kernel_dimension"],
    }


__all__ = [
    "decompose_response", "spectral_slack", "slack_ratio",
    "adaptive_minimax_certificate", "static_steering_lower_bound",
    "full_minimax_certificate", "complete_information_certificate",
]
