"""Method-agnostic local mechanism identities.

The learner-side APIs here take only source-state gradient functions.  Task
response ``A`` is intentionally accepted only by post-hoc attribution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
Gradient = Callable[[Array, Array], Array]
Solver = Callable[[float, Array], Array]


def _jacobian(function: Callable[[Array], Array], point: Array, step: float = 1e-5) -> Array:
    x = np.asarray(point, dtype=float)
    value = np.asarray(function(x), dtype=float)
    result = np.empty((value.size, x.size))
    for index in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        result[:, index] = (function(plus) - function(minus)) / (2.0 * step)
    return result


def _operator_norm(value: Array) -> float:
    singular = np.linalg.svd(np.asarray(value, dtype=float), compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


@dataclass(frozen=True)
class Mechanism:
    weights: Array
    H_R: Array
    B_R: Array
    g: Array
    K: Array
    C: Array
    pi: Array
    reconstructed_pi: Array
    reconstruction_relative_error: float
    total_h_relative_error: float
    total_b_relative_error: float


def decompose_mechanism(weights: Array, source_state: Array, lam: float,
                        risk_gradient: Gradient, penalty_gradient: Gradient,
                        *, derivative_step: float = 1e-5,
                        observed_pi: Array | None = None,
                        observed_total_h: Array | None = None,
                        observed_total_b: Array | None = None) -> Mechanism:
    """Differentiate the exact source objective at its actual method solution.

    ``risk_gradient`` and ``penalty_gradient`` are independently supplied,
    making the distinction between source forcing ``C`` and filtering ``K``
    explicit rather than inferred from the total IFT Jacobian.
    """
    w, y = np.asarray(weights, dtype=float), np.asarray(source_state, dtype=float)
    h_r = _jacobian(lambda x: risk_gradient(x, y), w, derivative_step)
    b_r = _jacobian(lambda state: risk_gradient(w, state), y, derivative_step)
    g = np.asarray(penalty_gradient(w, y), dtype=float)
    k = _jacobian(lambda x: penalty_gradient(x, y), w, derivative_step)
    c = _jacobian(lambda state: penalty_gradient(w, state), y, derivative_step)
    total_h = h_r + lam * k
    total_b = b_r + lam * c
    pi = -np.linalg.solve(total_h, total_b)
    reconstructed = pi.copy()
    if observed_pi is not None:
        observed = np.asarray(observed_pi, dtype=float)
        if observed.shape != pi.shape:
            raise ValueError("observed_pi has incompatible shape")
        pi = observed
    relative = float(np.linalg.norm(pi - reconstructed) / max(np.linalg.norm(pi), 1e-12))
    h_error = 0.0 if observed_total_h is None else float(
        np.linalg.norm(total_h - np.asarray(observed_total_h, dtype=float))
        / max(np.linalg.norm(np.asarray(observed_total_h, dtype=float)), 1e-12)
    )
    b_error = 0.0 if observed_total_b is None else float(
        np.linalg.norm(total_b - np.asarray(observed_total_b, dtype=float))
        / max(np.linalg.norm(np.asarray(observed_total_b, dtype=float)), 1e-12)
    )
    return Mechanism(w, h_r, b_r, g, k, c, pi, reconstructed, relative, h_error, b_error)


def common_base_attribution(weights: Array, source_state: Array, lam: float,
                            risk_gradient: Gradient, penalty_gradient: Gradient,
                            *, derivative_step: float = 1e-5) -> dict[str, Array | float]:
    """Four-response forcing/filtering attribution at one explicitly common base."""
    item = decompose_mechanism(weights, source_state, lam, risk_gradient, penalty_gradient,
                               derivative_step=derivative_step)
    pi00 = -np.linalg.solve(item.H_R, item.B_R)
    pi_c0 = -np.linalg.solve(item.H_R, item.B_R + lam * item.C)
    pi_0k = -np.linalg.solve(item.H_R + lam * item.K, item.B_R)
    pi_ck = item.pi
    sensing = 0.5 * ((pi_c0 - pi00) + (pi_ck - pi_0k))
    filtering = 0.5 * ((pi_0k - pi00) + (pi_ck - pi_c0))
    interaction = pi_ck - pi00 - (pi_c0 - pi00) - (pi_0k - pi00)
    return {
        "Pi00": pi00, "PiC0": pi_c0, "Pi0K": pi_0k, "PiCK": pi_ck,
        "sensing": sensing, "filtering": filtering, "interaction": interaction,
        "symmetric_identity_residual": float(np.linalg.norm(pi_ck - pi00 - sensing - filtering)),
    }


def static_path_audit(lam: float, source_state: Array, solver: Solver,
                      risk_gradient: Gradient, penalty_gradient: Gradient,
                      *, step: float = 1e-4, derivative_step: float = 1e-5) -> dict[str, object]:
    """Audit ``dw/dlambda=-(H_R+lambda K)^-1 g`` at one path point."""
    w = np.asarray(solver(lam, source_state), dtype=float)
    mechanism = decompose_mechanism(w, source_state, lam, risk_gradient, penalty_gradient,
                                    derivative_step=derivative_step)
    predicted = -np.linalg.solve(mechanism.H_R + lam * mechanism.K, mechanism.g)
    plus = np.asarray(solver(lam + step, source_state), dtype=float)
    minus = np.asarray(solver(max(0.0, lam - step), source_state), dtype=float)
    denominator = step + min(step, lam)
    finite_difference = (plus - minus) / denominator
    return {
        "lambda": float(lam), "weights": w, "predicted": predicted,
        "finite_difference": finite_difference,
        "relative_error": float(np.linalg.norm(predicted - finite_difference) / max(np.linalg.norm(finite_difference), 1e-12)),
    }


def mechanism_row(mechanism: Mechanism, observation: Array) -> dict[str, object]:
    o = np.asarray(observation, dtype=float)
    c_o = mechanism.C @ o
    return {
        "pi_operator_norm": _operator_norm(mechanism.pi),
        "reconstructed_pi_operator_norm": _operator_norm(mechanism.reconstructed_pi),
        "pi_reconstruction_relative_error": mechanism.reconstruction_relative_error,
        "total_H_relative_error": mechanism.total_h_relative_error,
        "total_B_relative_error": mechanism.total_b_relative_error,
        "g_norm": float(np.linalg.norm(mechanism.g)),
        "K_operator_norm": _operator_norm(mechanism.K),
        "rank_K": int(np.linalg.matrix_rank(mechanism.K, tol=1e-9)),
        "C_operator_norm": _operator_norm(mechanism.C),
        "C_O_operator_norm": _operator_norm(c_o),
        "rank_C_O": int(np.linalg.matrix_rank(c_o, tol=1e-9)),
        "source_factorization_residual": float(np.linalg.norm(mechanism.pi @ o - mechanism.reconstructed_pi @ o)),
    }


__all__ = ["Mechanism", "decompose_mechanism", "common_base_attribution", "static_path_audit", "mechanism_row"]
