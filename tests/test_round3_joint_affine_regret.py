from __future__ import annotations

import numpy as np

from ood_repr_reg.round3r_3c_affine import (
    central_difference_trained_action,
    exact_ift_affine,
    frozen_quadratic_audit,
    main_benchmark,
    source_task_state_matrix,
)
from ood_repr_reg.round3r_3c_benchmark import make_benchmark
from ood_repr_reg.round3r_3e_joint_fixtures import fixtures, same_b_k_different_pi
from ood_repr_reg.round3r_3e_joint_regret import (
    independent_shifted_ball_maximum,
    information_floor,
    regret_decomposition,
    shifted_ball_maximum,
)
from ood_repr_reg.round3r_3e_recovery import minimax_recovery_error
from ood_repr_reg.round3r_3e_world_tangent import coupled_primary_geometry
from ood_repr_reg.round3r_3e_world_tangent import (
    response_factorization_residual,
    u_exposed_observation,
)


def test_exact_ift_and_trained_central_difference_agree_without_oracle_inputs():
    benchmark = make_benchmark()
    geometry = coupled_primary_geometry()
    for method in ("l2", "irmv1", "vrex"):
        result = exact_ift_affine(benchmark, method, 1e-2, geometry.observation)
        audit = central_difference_trained_action(benchmark, method, 1e-2, geometry.observation)
        assert result.valid
        assert audit["pass"]
        assert audit["max_error"] < 1e-5


def test_ift_state_shapes_and_frozen_audit_are_separate():
    benchmark = make_benchmark()
    geometry = coupled_primary_geometry()
    result = exact_ift_affine(benchmark, "l2", 0.1, geometry.observation)
    assert result.dz_f.shape == (benchmark.optimum.size, benchmark.optimum.size)
    assert result.dy_f.shape[1] == source_task_state_matrix(benchmark).size
    assert result.tangent.shape == (benchmark.optimum.size, geometry.spec.dimension)
    audit = frozen_quadratic_audit(benchmark, "l2", 0.1)
    assert audit["source"] == "frozen_wstar_audit_only"
    assert not np.shares_memory(result.z0, audit["quadratic_z0"])


def test_secular_solver_matches_independent_optimizer_including_hard_case():
    cases = ((np.array([0.2, 0.1]), np.eye(2)), (np.zeros(2), np.diag([2.0, 1.0])))
    for offset, matrix in cases:
        exact = shifted_ball_maximum(offset, matrix)
        independent = independent_shifted_ball_maximum(offset, matrix, starts=24)
        assert abs(exact.value - independent["value"]) < 1e-7
        assert exact.maximizer @ exact.maximizer <= 1.0 + 1e-10


def test_affine_regret_identity_and_information_floor():
    geometry = coupled_primary_geometry()
    z0 = np.zeros(geometry.response.shape[0])
    zero = np.zeros_like(geometry.response)
    decomposition = regret_decomposition(z0, geometry.response, zero, geometry.observation)
    assert np.isclose(decomposition["total_regret"], information_floor(geometry.response, geometry.observation))
    assert np.isclose(decomposition["interaction"], 0.0)
    assert np.isclose(decomposition["information_floor"], 0.5 * minimax_recovery_error(geometry.response, geometry.observation) ** 2)


def test_every_concrete_affine_policy_is_above_information_floor():
    benchmark = make_benchmark()
    geometry = coupled_primary_geometry()
    floor = information_floor(geometry.response, geometry.observation)
    for method in ("l2", "irmv1", "vrex"):
        result = exact_ift_affine(benchmark, method, 1e-2, geometry.observation)
        value = shifted_ball_maximum(result.z0, geometry.response + result.tangent).value
        assert value + 1e-10 >= floor


def test_complete_information_endpoint_has_zero_floor():
    geometry = coupled_primary_geometry()
    assert np.isclose(
        information_floor(geometry.response, u_exposed_observation(geometry)), 0.0,
        atol=1e-12,
    )


def test_factorization_residual_is_numerically_zero():
    assert response_factorization_residual(coupled_primary_geometry()) < 1e-10


def test_hidden_u_is_information_floor_and_exposing_it_reduces_it():
    geometry = coupled_primary_geometry()
    u = geometry.spec.index("U_emergent")
    assert np.linalg.norm(geometry.observation[:, u]) < 1e-12
    assert np.linalg.norm(geometry.response[:, u]) > 1e-10
    hidden = information_floor(geometry.response, geometry.observation)
    exposed = information_floor(geometry.response, u_exposed_observation(geometry))
    assert hidden > 0.0
    assert np.isclose(exposed, 0.0, atol=1e-12)


def test_fixtures_and_same_b_k_different_pi_are_explicit():
    values = fixtures()
    assert {"pure_information", "pure_static", "pure_adaptive", "mixed_failure", "zero_steering"} <= values.keys()
    example = same_b_k_different_pi()
    assert example["same_frozen_pair"]
    assert example["different_source_jacobian"]
    assert example["different_affine_action"]
    fixture = values["recoverable_residual_without_worst_case_increase"]
    assert fixture["recoverable_residual_operator_norm"] > 0.0
    assert np.isclose(fixture["total_regret"], fixture["baseline_regret"])


def test_vrex_invalid_local_path_is_stopped():
    benchmark = make_benchmark()
    geometry = coupled_primary_geometry()
    result = exact_ift_affine(benchmark, "vrex", 1e4, geometry.observation)
    assert not result.valid
    assert result.metric_min_eigenvalue <= 0.0


def test_main_benchmark_has_no_semantic_or_target_selection_dependency():
    result = main_benchmark(lambdas=(0.0, 1e-2))
    assert len(result["rows"]) == 6
    assert result["target_risk_used"] is False
    assert result["semantic_labels_used"] is False
    assert result["cluster_labels_used"] is False
    assert result["regularizer_geometry_used_for_selection"] is False


def test_joint_runner_excess_is_relative_to_information_floor(tmp_path):
    from ood_repr_reg.run_round3r_3e_joint import run

    output = run(output=tmp_path / "joint_regret")
    import json
    summary = json.loads((output / "results" / "summary.json").read_text())
    floor = summary["information_floor"]
    for row in summary["rows"]:
        assert row["total_excess"] >= -1e-12
        assert np.isclose(
            row["total_excess"], row["affine_total_regret"] - floor, atol=1e-10,
        )
