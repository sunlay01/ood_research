import numpy as np

from ood_repr_reg.round3r_3a_benchmark import base_environment, source_environments
from ood_repr_reg.round3r_3a_source_geometry import source_optimum
from ood_repr_reg.round3r_3b_geometry import transformed_geometry
from ood_repr_reg.round3r_3c_benchmark import make_benchmark
from ood_repr_reg.round3r_3c_counterexamples import all_counterexamples
from ood_repr_reg.round3r_3c_derivatives import audit_regularizer
from ood_repr_reg.round3r_3c_geometry import (
    blind_fraction,
    curvature_ratio,
    infinitesimal_score,
    local_delta,
    response_space_coverage,
    steering_score,
    whitened_action,
)
from ood_repr_reg.round3r_3c_regularizers import RegularizerState, regularizer_state
from ood_repr_reg.round3r_3c_solver import exact_solution, lambda_grid, local_solution


def test_benchmark_reuses_frozen_relevant_response_geometry():
    benchmark = make_benchmark()
    assert len(benchmark.probes) == 35
    assert len(benchmark.relevant_indices) == 33
    assert np.linalg.matrix_rank(benchmark.q_responses[:, benchmark.relevant_indices], tol=1e-9) == 5


def test_all_derivative_audits_agree_with_autodiff_and_finite_difference():
    benchmark = make_benchmark()
    for method in ("l2", "vrex", "irmv1", "coral"):
        audit = audit_regularizer(method, benchmark.optimum, benchmark)
        assert audit["audit_pass"]
        assert audit["gradient_max_error"] < 1e-5
        assert audit["hessian_max_error"] < 1e-3


def test_local_minimizer_and_whitened_action_identities():
    benchmark = make_benchmark()
    state = regularizer_state("l2", benchmark)
    lam = 0.25
    local = local_solution(benchmark, "l2", lam)
    expected = -lam * np.linalg.solve(benchmark.hessian + lam * state.hessian, state.gradient)
    assert np.allclose(local["delta"], expected)
    b, k, root = whitened_action(benchmark.hessian, state)
    z = root @ benchmark.hessian @ local["delta"]
    assert np.allclose(z, -lam * np.linalg.solve(np.eye(z.size) + lam * k, b))


def test_steering_and_curvature_ratio_identities():
    benchmark = make_benchmark()
    state = regularizer_state("l2", benchmark)
    b, k, _ = whitened_action(benchmark.hessian, state)
    q = benchmark.q_responses[:, benchmark.relevant_indices[0]]
    lam = 0.1
    delta = local_delta(benchmark.hessian, state, lam)[0]
    gradient = benchmark.gradients[:, benchmark.relevant_indices[0]]
    assert np.isclose(gradient @ delta, steering_score(q, b, k, lam))
    assert np.isclose(curvature_ratio(q, k, lam), (q @ np.linalg.solve(np.eye(q.size) + lam * k, q)) / (q @ q))
    assert np.isclose(infinitesimal_score(q, k), q @ k @ q / (q @ q))


def test_exact_blind_direction_and_psd_kernel_limit():
    q = np.array([1.0, 0.0])
    k = np.diag([0.0, 2.0])
    assert np.isclose(curvature_ratio(q, k, 0.1), 1.0)
    assert np.isclose(blind_fraction(q, k), 1.0)
    assert np.isclose(curvature_ratio(q, k, 1e8), blind_fraction(q, k), atol=1e-7)


def test_small_lambda_control_expansion():
    q = np.array([1.0, 2.0])
    k = np.diag([2.0, -0.5])
    c = infinitesimal_score(q, k)
    lam = 1e-7
    error = abs(curvature_ratio(q, k, lam) - (1.0 - lam * c))
    assert error < 1e-10


def test_l2_fixture_and_corral_gauge_degeneracy():
    benchmark = make_benchmark()
    l2 = regularizer_state("l2", benchmark)
    assert np.allclose(l2.gradient, benchmark.optimum)
    assert np.allclose(l2.hessian, np.eye(benchmark.optimum.size))
    counterexamples = all_counterexamples()
    assert counterexamples["coral_gauge_degeneracy"]["infimum_zero"]
    assert counterexamples["coral_gauge_degeneracy"]["law"] == "c^4"


def test_counterexamples_cover_required_failure_modes():
    examples = all_counterexamples()
    assert examples["relevant_but_blind"]["kq_zero"]
    assert examples["irrelevant_but_penalized"]["ood_selectivity_failure"]
    assert examples["same_penalty_different_ood"]["same_l2_penalty"]
    assert examples["same_curvature_opposite_steering"]["opposite_sign"]


def test_coverage_is_response_space_level_and_no_cluster_labels():
    benchmark = make_benchmark()
    state = regularizer_state("vrex", benchmark)
    coverage = response_space_coverage(benchmark.q_responses[:, benchmark.relevant_indices], state, benchmark.hessian)
    assert coverage["response_rank"] == 5
    assert "cluster" not in str(coverage).lower()


def test_intrinsic_relevance_survives_coordinate_transform():
    benchmark = make_benchmark()
    rng = np.random.default_rng(11)
    transform = rng.normal(size=(benchmark.optimum.size, benchmark.optimum.size)) + 2.0 * np.eye(benchmark.optimum.size)
    transformed = transformed_geometry(benchmark.hessian, benchmark.gradients, transform)
    original = benchmark.gradients.T @ np.linalg.solve(benchmark.hessian, benchmark.gradients)
    assert np.max(np.abs(transformed["gram"] - original)) < 1e-9


def test_exact_population_path_and_lambda_grid_are_present():
    benchmark = make_benchmark()
    path = [exact_solution(benchmark, "vrex", float(lam)) for lam in lambda_grid()[:5]]
    assert len(path) == 5
    assert all(item["success"] for item in path)


def test_primary_state_does_not_read_3b_semantic_assignments():
    benchmark = make_benchmark()
    state = regularizer_state("irmv1", benchmark)
    assert state.method == "IRMV1"
    assert not hasattr(state, "cluster_assignments")
    assert not hasattr(state, "oracle_mechanism")
