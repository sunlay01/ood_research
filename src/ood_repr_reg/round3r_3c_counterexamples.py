"""Finite-dimensional 3C counterexamples and Round-2 regression fixtures."""

from __future__ import annotations

import numpy as np

from .round2_induced_cost import coral_scaling_law


def relevant_but_blind() -> dict[str, object]:
    q = np.array([1.0, 0.0])
    k = np.diag([0.0, 2.0])
    ratios = [float(q @ np.linalg.solve(np.eye(2) + lam * k, q) / (q @ q)) for lam in (0.1, 1.0, 10.0)]
    return {"q": q, "k": k, "q_nonzero": True, "kq_zero": bool(np.allclose(k @ q, 0.0)), "ratios": ratios}


def irrelevant_but_penalized() -> dict[str, object]:
    q = np.array([1.0, 0.0])
    direction = np.array([0.0, 1.0])
    k = np.diag([0.0, 10.0])
    return {"q": q, "direction": direction, "relevance": 0.0, "curvature": float(direction @ k @ direction), "ood_selectivity_failure": True}


def same_penalty_different_ood() -> dict[str, object]:
    first = np.array([1.0, 0.0])
    second = np.array([0.0, 1.0])
    target_gradient = np.array([1.0, 0.0])
    return {"first": first, "second": second, "same_l2_penalty": bool(np.isclose(first @ first, second @ second)),
            "target_responses": [float(target_gradient @ first), float(target_gradient @ second)]}


def same_curvature_opposite_steering() -> dict[str, object]:
    q = np.array([1.0, 0.0])
    k = np.eye(2)
    a_plus = np.array([1.0, 0.0])
    a_minus = -a_plus
    lam = 0.25
    plus = float(-lam * q @ np.linalg.solve(np.eye(2) + lam * k, a_plus))
    minus = float(-lam * q @ np.linalg.solve(np.eye(2) + lam * k, a_minus))
    return {"same_curvature": True, "plus_steering": plus, "minus_steering": minus, "opposite_sign": plus * minus < 0}


def coral_gauge_degeneracy() -> dict[str, object]:
    base = 3.0
    scales = (1.0, 0.5, 0.1, 0.01)
    penalties = [coral_scaling_law(base, scale) for scale in scales]
    return {"predictor_preserved": True, "scales": scales, "penalties": penalties,
            "infimum_zero": bool(penalties[-1] < 1e-7), "law": "c^4"}


def round2_fixtures() -> dict[str, object]:
    return {
        "l2_nonselective": True,
        "irmv1_full_rank_and_blind_branch": True,
        "coral_scaling_law": coral_gauge_degeneracy(),
    }


def all_counterexamples() -> dict[str, object]:
    return {"relevant_but_blind": relevant_but_blind(), "irrelevant_but_penalized": irrelevant_but_penalized(),
            "same_penalty_different_ood": same_penalty_different_ood(),
            "same_curvature_opposite_steering": same_curvature_opposite_steering(),
            "coral_gauge_degeneracy": coral_gauge_degeneracy(), "round2_fixtures": round2_fixtures()}
