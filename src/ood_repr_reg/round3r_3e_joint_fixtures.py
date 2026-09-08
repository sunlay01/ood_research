"""Small abstract fixtures for the joint information/regularization audit."""

from __future__ import annotations

import numpy as np

from .round3r_3e_joint_regret import regret_decomposition, shifted_ball_maximum


def _record(name, a, o, z, t):
    result = regret_decomposition(np.asarray(z, float), np.asarray(a, float), np.asarray(t, float), np.asarray(o, float))
    return {"setting": name, **{k: v for k, v in result.items() if k not in {"A_irreducible", "A_recoverable", "recoverable_residual"}}}


def fixtures() -> dict[str, object]:
    identity = np.eye(2)
    return {
        "pure_information": _record("pure_information", np.diag([2.0, 1.0]), np.zeros((0, 2)), np.zeros(2), np.zeros((2, 2))),
        "pure_static": _record("pure_static", identity, identity, np.array([0.3, -0.1]), np.zeros((2, 2))),
        "pure_adaptive": _record("pure_adaptive", identity, identity, np.zeros(2), np.array([[0.2, 0.0], [0.0, -0.1]])),
        "mixed_failure": _record("mixed_failure", identity, np.array([[1.0, 0.0]]), np.array([0.2, 0.1]), np.array([[0.0, 0.7], [0.0, 0.0]])),
        "zero_steering": _record("zero_steering", identity, identity, np.zeros(2), np.zeros((2, 2))),
        "recoverable_residual_without_worst_case_increase": _record(
            "recoverable_residual_without_worst_case_increase", np.diag([2.0, 1.0]),
            np.array([[0.0, 1.0]]), np.zeros(2), np.array([[0.0, 0.0], [0.0, 0.25]])),
    }


def same_b_k_different_pi() -> dict[str, object]:
    b = np.array([0.4, -0.2])
    k = np.eye(2)
    b_s_first = np.eye(2)
    b_s_second = np.array([[1.0, 0.0], [0.0, -1.0]])
    pi1 = b_s_first
    pi2 = np.array([[0.0, 1.0], [1.0, 0.0]]) @ b_s_second
    lam = 0.5
    frozen_pi = -np.linalg.solve(np.eye(2) + lam * k, np.eye(2))
    frozen_z0 = -lam * np.linalg.solve(np.eye(2) + lam * k, b)
    return {
        "b": b, "K": k, "lambda": lam,
        "B_S_first": b_s_first, "B_S_second": b_s_second,
        "Pi_first": pi1, "Pi_second": pi2,
        "frozen_Pi": frozen_pi, "frozen_z0": frozen_z0,
        "same_frozen_pair": True,
        "different_source_jacobian": bool(not np.allclose(b_s_first, b_s_second)),
        "different_affine_action": bool(not np.allclose(pi1, pi2)),
    }


__all__ = ["fixtures", "same_b_k_different_pi"]
