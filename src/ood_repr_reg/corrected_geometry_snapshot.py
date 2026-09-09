"""Audits for the corrected response decomposition frozen by the CMNIST bridge."""

from __future__ import annotations

import numpy as np

from .round3r_3e_c_spectral import decompose_response

Array = np.ndarray


def operator_norm(value: Array) -> float:
    singular = np.linalg.svd(np.asarray(value, dtype=float), compute_uv=False)
    return float(singular[0]) if singular.size else 0.0


def corrected_snapshot_audit(response: Array, observation: Array,
                             pi_o: Array | None = None,
                             tolerance: float = 1e-10) -> dict[str, object]:
    """Audit the frozen ``A=A_irr+A_rec`` and ``E=A_rec+PiO`` convention.

    The full-observation and zero-observation checks are algebraic endpoints,
    so they remain meaningful for every supplied benchmark snapshot.
    """
    a, o = np.asarray(response, dtype=float), np.asarray(observation, dtype=float)
    if a.ndim != 2 or o.ndim != 2 or a.shape[1] != o.shape[1]:
        raise ValueError("response and observation must share world dimension")
    parts = decompose_response(a, o, tolerance)
    irr = np.asarray(parts["A_irreducible"])
    rec = np.asarray(parts["A_recoverable"])
    adaptive = np.zeros_like(a) if pi_o is None else np.asarray(pi_o, dtype=float)
    if adaptive.shape != a.shape:
        raise ValueError("pi_o must have the response shape")
    full = decompose_response(a, np.eye(a.shape[1]), tolerance)
    empty = decompose_response(a, np.zeros((1, a.shape[1])), tolerance)
    e = rec + adaptive
    scale = max(1.0, operator_norm(a), operator_norm(adaptive))
    residual = operator_norm(a - irr - rec)
    return {
        "A_irreducible": irr,
        "A_recoverable": rec,
        "E": e,
        "P": parts["P"],
        "Q": parts["Q"],
        "decomposition_residual": residual,
        "full_observation_recoverable_residual": operator_norm(
            np.asarray(full["A_recoverable"]) - a
        ),
        "full_observation_irreducible_norm": operator_norm(full["A_irreducible"]),
        "zero_observation_recoverable_norm": operator_norm(empty["A_recoverable"]),
        "zero_observation_E_norm": operator_norm(np.asarray(empty["A_recoverable"])),
        "passes": bool(
            residual <= tolerance * scale
            and operator_norm(np.asarray(full["A_recoverable"]) - a) <= tolerance * scale
            and operator_norm(full["A_irreducible"]) <= tolerance * scale
            and operator_norm(empty["A_recoverable"]) <= tolerance * scale
        ),
        "tolerance": float(tolerance),
    }


__all__ = ["corrected_snapshot_audit", "operator_norm"]
