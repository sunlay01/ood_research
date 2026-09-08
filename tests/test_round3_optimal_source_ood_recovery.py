from __future__ import annotations

import numpy as np

from ood_repr_reg.round3r_3e_counterexamples import counterexamples
from ood_repr_reg.round3r_3e_information_ladder import (
    coupled_information_ladder,
    information_ladder,
    monotonicity_audit,
    same_source_count_geometry,
)
from ood_repr_reg.round3r_3e_nonlinear_local import nonlinear_local_stress
from ood_repr_reg.round3r_3e_random_matrix_stress import random_matrix_stress
from ood_repr_reg.round3r_3e_recovery import (
    ambiguity_diameter,
    coordinate_transformed_pair,
    factorization_diagnostic,
    irreducible_response_operator,
    minimax_recovery_error,
    nullspace_basis,
    operator_norm,
    orthogonal_projector_from_basis,
    pseudoinverse_residual,
)
from ood_repr_reg.round3r_3e_round3d import round3d_recovery
from ood_repr_reg.round3r_3e_world_tangent import (
    WorldTangentSpec,
    coupled_primary_geometry,
    finite_difference_stability,
)


def test_nullspace_basis_is_orthonormal_and_annihilated():
    observation = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
    basis = nullspace_basis(observation)
    assert basis.shape == (3, 2)
    assert np.allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    assert np.linalg.norm(observation @ basis) < 1e-12


def test_projector_and_singular_value_formula_agree():
    response = np.array([[2.0, 1.0, -1.0], [0.0, 3.0, 4.0]])
    observation = np.array([[1.0, 0.0, 0.0]])
    basis = nullspace_basis(observation)
    alpha = minimax_recovery_error(response, observation)
    assert np.isclose(alpha, operator_norm(response @ basis))
    assert np.isclose(alpha, operator_norm(response @ orthogonal_projector_from_basis(basis)))
    assert np.allclose(irreducible_response_operator(response, observation), response @ basis @ basis.T)


def test_pseudoinverse_achiever_has_exact_irreducible_residual():
    response = np.diag([2.0, 1.0, 4.0])
    observation = np.array([[1.0, 0.0, 0.0]])
    # The third coordinate is the leading right singular direction of A on
    # ker(O), so this unit world attains the operator-norm residual.
    world = np.array([0.0, 0.0, 1.0])
    residual = pseudoinverse_residual(response, observation, world)
    assert np.isclose(np.linalg.norm(residual), minimax_recovery_error(response, observation))


def test_ambiguity_diameter_and_endpoints():
    response = np.diag([3.0, 2.0])
    assert np.isclose(ambiguity_diameter(response, np.zeros((0, 2))), 2.0 * operator_norm(response))
    assert minimax_recovery_error(response, np.eye(2)) == 0.0
    assert minimax_recovery_error(np.zeros((1, 2)), np.zeros((0, 2))) == 0.0


def test_zero_error_agrees_with_kernel_inclusion_and_factorization():
    response = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]])
    observation = np.eye(3)
    diagnostic = factorization_diagnostic(response, observation)
    assert diagnostic["zero_error"]
    assert diagnostic["kernel_inclusion"]
    assert diagnostic["factorization_exists_numerically"]


def test_information_ladder_is_monotone_and_redundancy_does_not_help():
    rows = information_ladder()
    assert monotonicity_audit(rows)["nonincreasing"]
    assert rows[1]["alpha"] == rows[2]["alpha"]
    assert rows[-1]["alpha"] == 0.0


def test_coupled_primary_is_an_eight_dimensional_source_only_tangent():
    geometry = coupled_primary_geometry()
    spec = WorldTangentSpec()
    assert geometry.observation.shape[1] == spec.dimension == 8
    assert geometry.response.shape[1] == spec.dimension
    metadata = spec.metadata()
    assert metadata["world_metric"] == "Euclidean on standardized environment intervention coordinates"
    assert metadata["magnitude_one_scales"] == {
        "S1_relation": 0.20, "S2_relation": 0.20,
        "S1_mean": 0.35, "S2_mean": 0.35,
        "S1_variance": 0.30, "S2_variance": 0.30,
        "N1_variance": 0.50, "U_emergent": 0.75,
    }
    audit = finite_difference_stability(geometry)
    assert audit["pass"]
    assert audit["target_risk_used"] is False
    assert audit["mechanism_labels_used"] is False
    assert audit["cluster_labels_used"] is False
    assert audit["regularizer_geometry_used"] is False


def test_hidden_emergent_u_is_source_null_but_response_active():
    geometry = coupled_primary_geometry()
    index = geometry.spec.index("U_emergent")
    assert np.linalg.norm(geometry.observation[:, index]) < 1e-12
    assert np.linalg.norm(geometry.response[:, index]) > 1e-10


def test_coupled_information_ladder_is_nested_and_exposes_u():
    geometry = coupled_primary_geometry()
    rows = coupled_information_ladder(geometry)
    assert monotonicity_audit(rows)["nonincreasing"]
    alpha = {row["design"]: float(row["alpha"]) for row in rows}
    assert np.isclose(alpha["base_source"], alpha["duplicate_source"])
    assert np.isclose(alpha["base_source"], alpha["risk_null_noise_source"])
    assert alpha["u_exposed_source"] < alpha["base_source"] - 1e-10
    assert alpha["full_information_mathematical_endpoint"] == 0.0


def test_same_source_count_has_different_observation_geometry_and_alpha():
    control = same_source_count_geometry(coupled_primary_geometry())
    assert control["same_source_count"]
    assert control["different_alpha"]
    assert control["aligned_observation_reduces_alpha"]


def test_coordinate_invariance_under_allowed_changes():
    rng = np.random.default_rng(7)
    response = rng.normal(size=(3, 4))
    observation = rng.normal(size=(2, 4))
    alpha = minimax_recovery_error(response, observation)
    source_change = np.array([[2.0, 1.0], [0.0, 1.0]])
    world = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    response_rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    assert np.isclose(alpha, minimax_recovery_error(*coordinate_transformed_pair(response, observation, source_change=source_change)))
    assert np.isclose(alpha, minimax_recovery_error(*coordinate_transformed_pair(response, observation, world_isometry=world)))
    assert np.isclose(alpha, minimax_recovery_error(*coordinate_transformed_pair(response, observation, response_isometry=response_rotation)))


def test_required_dimension_and_rank_counterexamples():
    examples = counterexamples()
    same = examples["same_structural_dimension_different_alpha"]
    assert same["small"]["structural_ambiguity_dimension"] == same["large"]["structural_ambiguity_dimension"] == 1
    assert same["small"]["alpha"] < same["large"]["alpha"]
    different = examples["different_structural_dimension_same_alpha"]
    assert different["one"]["structural_ambiguity_dimension"] != different["two"]["structural_ambiguity_dimension"]
    assert np.isclose(different["one"]["alpha"], different["two"]["alpha"])
    rank = examples["same_observation_rank_different_alpha"]
    assert rank["observe_sensitive"]["rank_O"] == rank["observe_weak"]["rank_O"]
    assert rank["observe_sensitive"]["alpha"] != rank["observe_weak"]["alpha"]


def test_round3d_families_and_pair_audit():
    result = round3d_recovery()
    assert result["identifiable_linear_world_family"]["alpha"] == 0.0
    assert result["hidden_emergent_world_family"]["alpha"] > 0.0
    pair = result["source_identical_target_different_pair"]
    assert pair["source_states_identical_exactly"]
    assert pair["source_state_equality_checked_between_distinct_world_records"]
    assert pair["source_u_gamma_fixed_to_zero_in_both_worlds"]
    assert pair["q_difference_norm"] > 0.0
    assert np.isclose(pair["pair_ratio_to_2alpha"], 1.0)
    coupled = result["coupled_3a_3d_primary"]
    assert coupled["u_source_null"]
    assert coupled["u_response_active"]
    assert coupled["source_observation_uses_target_risk"] is False


def test_nonlinear_track_is_explicitly_local_and_label_free():
    result = nonlinear_local_stress()
    assert result["status"] == "LOCAL-TANGENT-DIAGNOSTIC-ONLY"
    assert result["global_nonlinear_claim"] is False


def test_random_audit_checks_projector_pseudoinverse_and_sampled_estimators():
    result = random_matrix_stress(repeats=8)
    assert result["audit_pass"]
    assert result["max_projector_formula_error"] < 1e-8
    assert result["max_sampled_estimator_lower_bound_violation"] < 1e-8
