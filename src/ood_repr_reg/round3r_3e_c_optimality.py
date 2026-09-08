"""Sharp finite-dimensional optimality certificates for 3E-C."""

from __future__ import annotations

import numpy as np

from .round3r_3e_joint_regret import shifted_ball_maximum
from .round3r_3e_recovery import operator_norm
from .round3r_3e_c_spectral import (
    adaptive_minimax_certificate,
    decompose_response,
    spectral_slack,
)

Array = np.ndarray


def response_side_orthogonality(response: Array, observation: Array,
                                adaptive: Array | None = None,
                                tolerance: float = 1e-10) -> dict[str, object]:
    """Numerically verify the projection identities used by the theorem."""
    parts = decompose_response(response, observation, tolerance)
    a = np.asarray(response, dtype=float)
    o = np.asarray(observation, dtype=float)
    p, q = parts["P"], parts["Q"]
    irr, rec = parts["A_irreducible"], parts["A_recoverable"]
    e = rec if adaptive is None else rec + np.asarray(adaptive, dtype=float)
    return {
        "PO_star_norm": float(operator_norm(p @ o.T)),
        "E_P_norm": float(operator_norm(e @ p)),
        "Airr_Q_norm": float(operator_norm(irr @ q)),
        "Airr_E_star_norm": float(operator_norm(irr @ e.T)),
        "E_Airr_star_norm": float(operator_norm(e @ irr.T)),
        "gram_sum_residual": float(operator_norm((irr + e) @ (irr + e).T - irr @ irr.T - e @ e.T)),
    }


def adaptive_certificate(response: Array, observation: Array, recoverable: Array | None = None,
                         tolerance: float = 1e-10) -> dict[str, object]:
    """Return the exact adaptive-only iff certificate for ``E``."""
    parts = spectral_slack(response, observation, tolerance)
    e = np.asarray(parts["A_recoverable"] if recoverable is None else recoverable, dtype=float)
    result = adaptive_minimax_certificate(np.asarray(parts["A_irreducible"]), e, tolerance)
    result["orthogonality"] = response_side_orthogonality(response, observation, e, tolerance)
    return result


def full_affine_certificate(z0: Array, response: Array, observation: Array,
                            adaptive: Array | None = None,
                            tolerance: float = 1e-10) -> dict[str, object]:
    """Evaluate full regret and the exact static/adaptive iff conditions."""
    a = np.asarray(response, dtype=float)
    o = np.asarray(observation, dtype=float)
    z = np.asarray(z0, dtype=float)
    parts = decompose_response(a, o, tolerance)
    e = np.asarray(parts["A_recoverable"] if adaptive is None else adaptive, dtype=float)
    certificate = adaptive_certificate(a, o, e, tolerance)
    # ``e`` is already ``A_rec + Pi O``.  The full affine map is therefore
    # ``A_irr + e``; adding the original ``A`` here would double-count ``A_rec``.
    actual = shifted_ball_maximum(z, np.asarray(parts["A_irreducible"]) + e)
    floor = float(certificate["information_floor"])
    static_bound = floor + 0.5 * float(z @ z)
    scale = max(1.0, actual.value, floor, float(z @ z))
    denominator = 0.5 * float(z @ z)
    return {
        **certificate,
        "total_regret": float(actual.value),
        "total_status": actual.status,
        "static_tax_lower_bound": float(static_bound),
        "static_tax_holds": bool(actual.value + tolerance * scale >= static_bound),
        "total_excess": float(max(0.0, actual.value - floor)),
        "static_tightness_ratio": None if denominator <= tolerance else float((actual.value - floor) / denominator),
        "z0_norm": float(np.linalg.norm(z)),
        "zero_static": bool(np.linalg.norm(z) <= tolerance),
        "full_minimax_condition": bool(
            np.linalg.norm(z) <= tolerance and certificate["condition_holds"]
        ),
        "maximizer": actual.maximizer,
    }


def complete_information_adaptive_certificate(response: Array, observation: Array,
                                              adaptive: Array,
                                              tolerance: float = 1e-10) -> dict[str, object]:
    """Specialize the adaptive theorem at ``A_irr=0``."""
    certificate = adaptive_certificate(response, observation, adaptive, tolerance)
    irr = np.asarray(decompose_response(response, observation, tolerance)["A_irreducible"])
    complete = operator_norm(irr) <= tolerance * max(1.0, operator_norm(response))
    e = np.asarray(adaptive)
    return {
        **certificate,
        "A_irreducible": irr,
        "E": e,
        "complete_information": bool(complete),
        "recoverable_exact": bool(operator_norm(e) <= tolerance),
        "complete_information_iff": bool(complete and operator_norm(e) <= tolerance),
    }


def static_tax_bound(z0: Array, response: Array, observation: Array,
                     adaptive: Array | None = None,
                     tolerance: float = 1e-10) -> dict[str, object]:
    """Return the ``+1/2 ||z0||^2`` lower bound for the full affine policy."""
    return full_affine_certificate(z0, response, observation, adaptive, tolerance)


__all__ = [
    "response_side_orthogonality", "adaptive_certificate", "full_affine_certificate",
    "complete_information_adaptive_certificate", "static_tax_bound",
]
