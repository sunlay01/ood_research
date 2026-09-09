from pathlib import Path

import numpy as np

from ood_repr_reg.algorithm_mechanism.adapters import gaussian_input
from ood_repr_reg.algorithm_mechanism.audits import evaluate_input
from ood_repr_reg.corrected_geometry_snapshot import corrected_snapshot_audit
from ood_repr_reg.environment_family.finite_difference import family_directional_derivative
from ood_repr_reg.environment_family.legacy_gaussian import LegacyGaussianFamily
from ood_repr_reg.environment_family.source_induced import EnvironmentParameterization, parameter_jacobian
from ood_repr_reg.round3r_3b_benchmark import ModuleEnvironment
from ood_repr_reg.round3r_3e_c_benchmarks import primary_hidden_u_world
from ood_repr_reg.round3r_3e_c_spectral import slack_ratio
from ood_repr_reg.round3r_3e_joint_regret import shifted_ball_maximum
from ood_repr_reg.run_repair_evidence_gate import (
    ToyEnvironment,
    ToyUnitIntervalFamily,
    r1_slack_audit,
    r2_family_provenance,
    r3_boundary_audit,
    run,
)
from ood_repr_reg.sharp_optimality.geometry import response_parts


def test_r1_zero_slack_spill_scale_sweep_and_compatible_case():
    audit = r1_slack_audit()
    assert audit["status"] == "REPAIR-PASS"
    for row in audit["scale_sweep"]:
        assert not row["support_compatible"]
        assert np.isinf(row["rho_slack"])
    assert audit["compatible_case"]["support_compatible"]
    assert np.isclose(audit["compatible_case"]["rho_slack"], 4.0)


def test_slack_ratio_near_tolerance_does_not_pass_on_exact_threshold():
    irreducible = np.array([[1.0], [0.0]])
    spill = slack_ratio(irreducible, np.array([[1e-10], [0.0]]), tolerance=1e-10)
    assert spill["support_residual"] >= spill["support_threshold"]
    assert not spill["support_compatible"]
    assert np.isinf(spill["rho_slack"])


def test_corrected_decomposition_full_and_zero_observation_endpoints():
    response = np.array([[2.0, 0.5], [-1.0, 3.0]])
    full = response_parts(response, np.eye(2), np.zeros_like(response))
    assert np.allclose(full["R"], 0.0)
    assert np.allclose(full["A_recoverable"], response)
    assert np.allclose(full["E"], response)

    zero = corrected_snapshot_audit(response, np.zeros((1, 2)), np.zeros_like(response))
    assert zero["passes"]
    assert np.allclose(zero["A_recoverable"], 0.0)
    assert np.allclose(zero["E"], 0.0)
    assert zero["decomposition_residual"] < 1e-10


def test_r3_boundary_finite_difference_no_illegal_evaluations():
    audit = r3_boundary_audit()
    assert audit["status"] == "REPAIR-PASS"
    assert {row["scheme"] for row in audit["rows"]} == {
        "central", "forward_second_order", "backward_second_order",
    }
    assert all(0.0 <= value <= 1.0 for value in audit["evaluations"])
    assert audit["legacy_zero_variance"]["pass"]
    assert audit["source_induced_parameter_jacobian"]["pass"]


def test_legacy_gaussian_variance_boundary_allows_forward_difference():
    base = ModuleEnvironment(
        shortcut_rhos=(0.75, 0.57),
        shortcut_means=(0.18, 0.08),
        shortcut_variances=(0.0, 0.61),
        n_noise=4,
    )
    family = LegacyGaussianFamily(base=base)
    negative, positive = family.legal_step_interval(base, "S1_variance", role="source")
    assert negative == 0.0
    assert np.isinf(positive)
    derivative, diagnostic = family_directional_derivative(
        family, base, "S1_variance",
        lambda env: np.array([env.shortcut_variances[0]]),
        1e-3, role="source",
    )
    assert diagnostic.scheme == "forward_second_order"
    assert np.isclose(float(derivative[0]), 0.30)


def test_source_induced_parameter_jacobian_is_boundary_safe_at_zero_variance():
    base = ModuleEnvironment(
        shortcut_rhos=(0.75, 0.57),
        shortcut_means=(0.18, 0.08),
        shortcut_variances=(0.0, 0.61),
        n_noise=4,
    )
    params = EnvironmentParameterization(len(base.shortcut_rhos), base.n_noise)
    jacobian, diagnostics = parameter_jacobian(base, params, 1e-6, return_diagnostics=True)
    variance_1 = next(row for row in diagnostics if row["parameter"] == "variance_1")
    assert np.all(np.isfinite(jacobian))
    assert variance_1["scheme"] == "forward_second_order"
    assert variance_1["legal_plus"]
    assert not variance_1["legal_minus"]


def test_toy_family_derivatives_at_interior_and_boundaries():
    family = ToyUnitIntervalFamily()

    def evaluate(env: ToyEnvironment) -> np.ndarray:
        assert 0.0 <= env.value <= 1.0
        return np.array([env.value ** 2 + 3.0 * env.value])

    cases = [
        (ToyEnvironment(0.5), "central", 4.0),
        (ToyEnvironment(0.0), "forward_second_order", 3.0),
        (ToyEnvironment(1.0), "backward_second_order", 5.0),
    ]
    for reference, scheme, expected in cases:
        derivative, diagnostic = family_directional_derivative(
            family, reference, "x", evaluate, 0.1, role="source",
        )
        assert diagnostic.scheme == scheme
        assert np.isclose(float(derivative[0]), expected)


def test_r2_family_provenance_distinguishes_legacy_and_source_induced():
    audit = r2_family_provenance()
    assert audit["status"] == "REPAIR-PASS"
    records = {row["config_id"]: row for row in audit["records"]}
    legacy = records["legacy_hidden_u_primary"]
    source = records["source_induced_comparison"]
    assert legacy["family_name"] != source["family_name"]
    assert legacy["benchmark_role"] == "hidden-U primary"
    assert source["benchmark_role"] == "source-induced comparison"
    assert legacy["family_dimension"] == 8
    assert legacy["rank_O_S"] == 7
    assert np.isclose(legacy["information_floor"], 0.05118145108608892)
    assert source["retained_modes"]


def test_task1_exact_mechanism_identity_and_no_target_leakage_gaussian():
    item = gaussian_input(primary_hidden_u_world(), "l2", 0.01)
    record = evaluate_input(item)
    assert record["learner"]["pi_reconstruction_relative_error"] < 1e-8
    assert record["learner"]["total_H_relative_error"] < 1e-8
    assert record["l2_exact"]["C_is_zero"]
    assert not record["target_used_by_learner"]
    assert not record["semantic_or_cluster_used_by_learner"]


def test_shifted_ball_solver_matches_closed_form_in_one_dimension():
    result = shifted_ball_maximum(np.array([2.0]), np.array([[3.0]]))
    assert np.isclose(result.value, 0.5 * (2.0 + 3.0) ** 2)
    assert np.isclose(abs(float(result.maximizer[0])), 1.0)


def test_repair_evidence_runner_smoke_without_cmnist(tmp_path: Path):
    summary = run(tmp_path, cmnist=False)
    results = tmp_path / "results"
    assert summary["r1_status"] == "REPAIR-PASS"
    assert summary["r2_status"] == "REPAIR-PASS"
    assert summary["r3_status"] == "REPAIR-PASS"
    assert summary["task2_row_count"] > 0
    assert (results / "repair_status.json").exists()
    assert (results / "family_provenance.json").exists()
    assert (results / "task2_theorem_audit.csv").exists()
    assert (tmp_path / "repair_evidence_report.md").exists()
