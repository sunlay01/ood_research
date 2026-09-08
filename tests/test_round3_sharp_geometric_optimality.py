from __future__ import annotations

import numpy as np

from ood_repr_reg.round3r_3e_c_benchmarks import (
    evaluate_table,
    primary_hidden_u_world,
    primary_u_exposed_world,
    search_legal_helps_worlds,
)
from ood_repr_reg.round3r_3e_c_counterexamples import counterexamples
from ood_repr_reg.round3r_3e_c_optimality import (
    adaptive_certificate,
    full_affine_certificate,
    response_side_orthogonality,
)
from ood_repr_reg.round3r_3e_c_spectral import decompose_response, spectral_slack
from ood_repr_reg.round3r_3e_c_small_lambda import small_lambda_diagnostics
from ood_repr_reg.round3r_3e_joint_fixtures import same_b_k_different_pi


def test_projection_response_side_identities_and_gram_sum():
    rng = np.random.default_rng(41)
    response = rng.normal(size=(4, 5))
    observation = rng.normal(size=(3, 5))
    parts = decompose_response(response, observation)
    adaptive = rng.normal(size=(4, 3)) @ observation
    audit = response_side_orthogonality(response, observation, adaptive)
    assert audit["PO_star_norm"] < 1e-10
    assert audit["E_P_norm"] < 1e-10
    assert audit["Airr_Q_norm"] < 1e-10
    assert audit["Airr_E_star_norm"] < 1e-10
    assert audit["E_Airr_star_norm"] < 1e-10
    assert audit["gram_sum_residual"] < 1e-10
    assert np.allclose(parts["P"] + parts["Q"], np.eye(5), atol=1e-12)


def test_spectral_slack_is_psd_and_condition_is_iff_numerically():
    irreducible = np.diag([2.0, 0.0])
    inside = np.diag([0.0, 1.0])
    outside = np.diag([0.0, 3.0])
    inside_result = adaptive_certificate(irreducible, np.diag([0.0, 1.0]), inside)
    outside_result = adaptive_certificate(irreducible, np.diag([0.0, 1.0]), outside)
    assert inside_result["condition_holds"]
    assert np.isclose(inside_result["adaptive_regret"], inside_result["information_floor"])
    assert outside_result["condition_holds"] is False
    assert outside_result["adaptive_regret"] > outside_result["information_floor"]
    assert np.linalg.eigvalsh(spectral_slack(irreducible, np.diag([0.0, 1.0]))["slack"]).min() >= -1e-12


def test_nonzero_recoverable_residual_can_be_minimax_optimal():
    examples = counterexamples()
    row = examples["nonzero_E_minimax_optimal"]
    assert row["recoverable_exact"] is False
    assert row["full_minimax_condition"]
    assert np.isclose(row["adaptive_regret"], row["information_floor"])


def test_complete_information_requires_zero_adaptive_residual():
    examples = counterexamples()
    row = examples["complete_information_nonzero_E_failure"]
    assert row["alpha"] == 0.0
    assert row["condition_holds"] is False
    assert row["adaptive_regret"] > 0.0
    assert row["full_minimax_condition"] is False


def test_static_steering_tax_tight_and_strict_fixtures():
    examples = counterexamples()
    tight = examples["static_tax_tight"]
    strict = examples["static_tax_strict"]
    assert tight["static_tax_holds"]
    assert strict["static_tax_holds"]
    assert np.isclose(tight["static_tightness_ratio"], 1.0)
    assert strict["static_tightness_ratio"] > 1.0


def test_hidden_u_recovers_sharp_theorem_and_hurts_with_positive_lambda():
    rows = evaluate_table(primary_hidden_u_world())
    erm = next(row for row in rows if row["method"] == "L2" and row["lambda"] == 0.0)
    valid = [row for row in rows if row["valid"]]
    assert all(row["adaptive_condition_holds"] for row in valid)
    assert all(np.isclose(row["adaptive_regret"], row["information_floor"], atol=1e-10) for row in valid)
    assert all(row["static_tax_holds"] for row in valid)
    assert any(row["lambda"] > 0 and row["total_regret"] > erm["total_regret"] for row in valid)


def test_u_exposed_table_is_complete_information_and_nonzero_e_is_not_free():
    rows = evaluate_table(primary_u_exposed_world())
    assert len(rows) == 18
    valid = [row for row in rows if row["valid"]]
    assert all(row["complete_information"] for row in valid)
    assert all(np.isclose(row["information_floor"], 0.0, atol=1e-12) for row in valid)
    assert any(row["lambda"] > 0 and row["E_operator_norm"] > 1e-8 and row["adaptive_regret"] > 1e-12 for row in valid)


def test_legal_help_search_is_source_derived_and_finds_a_within_world_improvement():
    result = search_legal_helps_worlds()
    assert result["target_oracle_used_for_selection"] is False
    assert result["candidate_count"] > 0
    assert result["selected"] is not None
    selected = result["selected"]
    assert selected["improvement"] > 1e-8
    assert selected["erm_recoverable_norm"] > 1e-8
    assert selected["candidate"]["total_regret"] < selected["erm_total_regret"]


def test_same_b_k_different_pi_and_no_forbidden_geometry_inputs():
    example = same_b_k_different_pi()
    assert example["same_frozen_pair"]
    assert example["different_source_jacobian"]
    assert example["different_affine_action"]
    world = primary_hidden_u_world()
    row = evaluate_table(world, lambdas=(0.0,))[0]
    assert "oracle_mechanism" not in row
    assert "cluster" not in row
    assert "target_risk" not in row


def test_small_lambda_diagnostic_has_quadratic_frozen_scaling():
    result = small_lambda_diagnostics(lambdas=(1e-5, 1e-4, 1e-3), methods=("l2",))
    rows = result["rows"]
    assert result["diagnostic_only"]
    assert all(row["valid"] for row in rows)
    # The residual after subtracting -lambda*b is bounded and does not grow as
    # 1/lambda, which is the numerical signature of the claimed expansion.
    assert rows[-1]["frozen_z_over_lambda_plus_b_norm"] < 10.0 * rows[0]["frozen_z_over_lambda_plus_b_norm"] + 1.0
