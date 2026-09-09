"""Runnable source-only mechanism audits with post-hoc response diagnostics."""

from __future__ import annotations

import numpy as np

from .adapters import MechanismInput
from .core import common_base_attribution, decompose_mechanism, mechanism_row, static_path_audit
from ..sharp_optimality.geometry import affine_policy_audit, response_parts


def evaluate_input(item: MechanismInput, *, derivative_step: float = 2e-5) -> dict[str, object]:
    """Audit one optimizer solution without exposing target quantities to its learner side."""
    mechanism = decompose_mechanism(
        item.weights, item.state, item.lam, item.risk_gradient, item.penalty_gradient,
        derivative_step=derivative_step, observed_pi=item.observed_pi,
        observed_total_h=item.observed_total_h, observed_total_b=item.observed_total_b,
    )
    common = common_base_attribution(
        item.common_base_weights, item.state, item.lam, item.risk_gradient, item.penalty_gradient,
        derivative_step=derivative_step,
    )
    learner = mechanism_row(mechanism, item.observation)
    # Response maps are post-hoc only.  The mechanism solver above receives no A.
    sharp = affine_policy_audit(
        item.response_offset, item.response, item.observation, item.response_adaptive,
    )
    parts = response_parts(item.response, item.observation, item.response_adaptive)
    # Counterfactual Pi maps are constructed entirely from source quantities at
    # the shared ERM base.  Only their conversion to E is post-hoc.
    for key in ("Pi00", "PiC0", "Pi0K", "PiCK"):
        adaptive = np.asarray(item.response_transform) @ np.asarray(common[key]) @ np.asarray(item.observation)
        common[f"{key}_response"] = adaptive
        common[f"E_{key[2:]}"] = np.asarray(parts["A_recoverable"]) + adaptive
    return {
        "setting": item.setting, "method": item.method, "lambda": item.lam,
        "weights": item.weights, "H_R": mechanism.H_R, "B_R": mechanism.B_R,
        "g": mechanism.g, "K": mechanism.K, "C": mechanism.C,
        "Pi": mechanism.pi, "Pi_reconstructed": mechanism.reconstructed_pi,
        "observation": np.asarray(item.observation),
        "PiO": item.response_adaptive, "E": parts["E"], "A_recoverable": parts["A_recoverable"],
        "common": common, "learner": learner, "sharp": sharp,
        "l2_exact": {
            "g_equals_weights": bool(np.allclose(mechanism.g, item.weights, atol=2e-6)) if item.method == "L2" else None,
            "K_equals_identity": bool(np.allclose(mechanism.K, np.eye(item.weights.size), atol=2e-6)) if item.method == "L2" else None,
            "C_is_zero": bool(np.linalg.norm(mechanism.C) <= 2e-6) if item.method == "L2" else None,
        },
        "common_base_kind": "ERM_source_solution",
        "common_base_difference_norm": float(np.linalg.norm(item.weights - item.common_base_weights)),
        "target_used_by_learner": False,
        "semantic_or_cluster_used_by_learner": False,
    }


def static_audit(item: MechanismInput) -> dict[str, object] | None:
    if item.lam <= 0.0:
        return None
    return {
        "setting": item.setting, "method": item.method, "lambda": item.lam,
        **static_path_audit(item.lam, item.state, item.solver, item.risk_gradient, item.penalty_gradient),
    }


def scalar_rows(record: dict[str, object]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    learner = dict(record["learner"])
    sharp = dict(record["sharp"])
    common = dict(record["common"])
    base = {"setting": record["setting"], "method": record["method"], "lambda": record["lambda"]}
    exact = {
        **base, **learner,
        "H_R_norm": float(np.linalg.norm(record["H_R"])), "B_R_norm": float(np.linalg.norm(record["B_R"])),
        "PiO_norm": float(np.linalg.norm(record["PiO"])), "E_norm": float(np.linalg.norm(record["E"])),
        "information_floor": sharp["information_floor"], "total_regret": sharp["total_regret"],
    }
    objects = {
        **base, "g_norm": float(np.linalg.norm(record["g"])),
        "K_operator_norm": learner["K_operator_norm"], "C_operator_norm": learner["C_operator_norm"],
        "Pi_operator_norm": learner["pi_operator_norm"],
        **{f"l2_{key}": value for key, value in record["l2_exact"].items()},
    }
    counter = {
        **base,
        "common_base_kind": record["common_base_kind"],
        "Pi00_norm": float(np.linalg.norm(common["Pi00"])), "PiC0_norm": float(np.linalg.norm(common["PiC0"])),
        "Pi0K_norm": float(np.linalg.norm(common["Pi0K"])), "PiCK_norm": float(np.linalg.norm(common["PiCK"])),
        "sensing_norm": float(np.linalg.norm(common["sensing"])),
        "filtering_norm": float(np.linalg.norm(common["filtering"])),
        "interaction_norm": float(np.linalg.norm(common["interaction"])),
        "E00_operator_norm": float(np.linalg.svd(common["E_00"], compute_uv=False)[0]),
        "EC0_operator_norm": float(np.linalg.svd(common["E_C0"], compute_uv=False)[0]),
        "E0K_operator_norm": float(np.linalg.svd(common["E_0K"], compute_uv=False)[0]),
        "ECK_operator_norm": float(np.linalg.svd(common["E_CK"], compute_uv=False)[0]),
        "symmetric_identity_residual": common["symmetric_identity_residual"],
    }
    return exact, objects, counter


__all__ = ["evaluate_input", "static_audit", "scalar_rows"]
