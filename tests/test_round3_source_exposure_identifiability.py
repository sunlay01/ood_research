import numpy as np

from ood_repr_reg.round3r_3b_benchmark import ModuleEnvironment, environment_state, source_design, source_optimum
from ood_repr_reg.round3r_3c_benchmark import make_benchmark
from ood_repr_reg.round3r_3d_counterexamples import design_unexposed_but_source_visible, source_identical_target_different
from ood_repr_reg.round3r_3d_designs import source_design_ladder
from ood_repr_reg.round3r_3d_exposure import (
    exact_exposure_membership,
    exposed_response_basis,
    quotient_dimensions,
    response_operator,
    source_design_span,
    source_reference_audit,
    target_exposure_fraction,
)
from ood_repr_reg.round3r_3d_identifiability import structural_identifiability
from ood_repr_reg.round3r_3d_state import svec_symmetric, task_state, transform_moment_state
from ood_repr_reg.run_round3r_3d import run


def test_task_state_reconstructs_linear_population_risk():
    env = ModuleEnvironment(n_noise=2)
    state = environment_state(env)
    w = np.arange(env.dimension, dtype=float) / 7.0
    vector = task_state(state)
    matrix = state.second
    assert np.isclose(w @ matrix @ w - 2 * w @ state.xy + state.y2, w @ matrix @ w - 2 * w @ state.xy + vector[-1])
    assert np.isclose(svec_symmetric(matrix) @ svec_symmetric(matrix), np.sum(matrix * matrix))


def test_response_operator_matches_gradient_and_shift_difference():
    base = ModuleEnvironment(n_noise=2)
    environments = source_design(base, relation_exposed=True)
    optimum, source = source_optimum(environments)
    target = environment_state(base.updated(u_gamma=0.4))
    direct = 2 * (target.second - source.second) @ optimum - 2 * (target.xy - source.xy)
    q = response_operator(source, optimum, target)
    h = 2 * source.second
    values, vectors = np.linalg.eigh(h)
    root = vectors @ np.diag(1 / np.sqrt(values)) @ vectors.T
    assert np.allclose(q, root @ direct)


def test_predictor_coordinate_transform_preserves_intrinsic_relevance():
    base = ModuleEnvironment(n_noise=2)
    environments = source_design(base, relation_exposed=True)
    optimum, source = source_optimum(environments)
    target = environment_state(base.updated(u_gamma=0.4))
    direct = 2 * (target.second - source.second) @ optimum - 2 * (target.xy - source.xy)
    h = 2 * source.second
    original = direct @ np.linalg.solve(h, direct)
    rng = np.random.default_rng(3)
    transform = rng.normal(size=(h.shape[0], h.shape[0])) + 2.0 * np.eye(h.shape[0])
    transformed_source = transform_moment_state(source, transform)
    transformed_target = transform_moment_state(target, transform)
    transformed_optimum = np.linalg.solve(transform.T, optimum)
    transformed_gradient = 2 * (transformed_target.second - transformed_source.second) @ transformed_optimum - 2 * (transformed_target.xy - transformed_source.xy)
    transformed_hessian = 2 * transformed_source.second
    assert np.isclose(original, transformed_gradient @ np.linalg.solve(transformed_hessian, transformed_gradient), atol=1e-9)


def test_source_reference_invariance_and_exact_membership():
    base = ModuleEnvironment(n_noise=2)
    environments = source_design(base, relation_exposed=True)
    optimum, source = source_optimum(environments)
    states = tuple(environment_state(env) for env in environments)
    span0 = source_design_span(states, 0)
    span_last = source_design_span(states, len(states) - 1)
    operator = response_operator(source, optimum)
    first = exposed_response_basis(span0["contrasts"], operator)
    last = exposed_response_basis(span_last["contrasts"], operator)
    assert first["rank"] == last["rank"]
    assert source_reference_audit(states, operator)["invariant"]
    target = operator @ span0["contrasts"][:, 1]
    assert exact_exposure_membership(target, first["basis"])
    assert np.isclose(target_exposure_fraction(target, first["basis"]), 1.0)


def test_quotient_dimension_identity_and_duplicate_source_invariance():
    result = run()
    q = result["exposure"]["quotient"]
    assert q["unexposed_quotient_dimension"] == q["total_dimension"] - q["exposed_dimension"]
    assert q["inclusion_residual"] < 1e-8
    rows = {row["name"]: row for row in result["source_designs"]}
    assert rows["duplicate_source"]["exposed_response_rank"] == rows["one_relation_contrast"]["exposed_response_rank"]


def test_risk_null_source_variation_does_not_add_exposure_rank():
    rows = {row["name"]: row for row in run()["source_designs"]}
    assert rows["risk_null_nuisance"]["exposed_response_rank"] == rows["one_relation_contrast"]["exposed_response_rank"]
    assert rows["risk_null_nuisance"]["source_contrast_rank"] > rows["one_relation_contrast"]["source_contrast_rank"]


def test_new_relevant_direction_increases_exposed_rank():
    rows = {row["name"]: row for row in run()["source_designs"]}
    assert rows["new_target_relevant_contrast"]["exposed_response_rank"] > rows["one_relation_contrast"]["exposed_response_rank"]


def test_structural_kernel_factorization_equivalence():
    identified = structural_identifiability(np.eye(3), np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 1.0]]))
    hidden = structural_identifiability(np.zeros((1, 2)), np.array([[1.0, -1.0]]))
    assert identified["kernel_inclusion"] and identified["factorization_exists_numerically"]
    assert hidden["structural_ambiguity_dimension"] == 1
    assert not hidden["kernel_inclusion"]


def test_exact_source_identical_world_pair_has_different_target_response():
    pair = source_identical_target_different(ModuleEnvironment(n_noise=2))
    assert pair["source_states_identical_exactly"]
    assert pair["q_different"]
    assert pair["q_difference_norm"] > 1e-8


def test_source_visible_but_design_unexposed_counterexample():
    counterexample = design_unexposed_but_source_visible(ModuleEnvironment(n_noise=2))
    assert counterexample["target_response_norm"] > 0


def test_primary_exposure_has_no_posthoc_labels_or_regularizer_inputs():
    result = run()
    exposure = result["exposure"]
    assert exposure["target_information_used_in_exposure"] is False
    assert exposure["mechanism_labels_used_in_exposure"] is False
    assert exposure["cluster_labels_used_in_exposure"] is False
    assert exposure["regularizer_used_in_exposure"] is False
    assert all(row["posthoc_only"] for row in result["round3c_intersection"])


def test_tolerance_profile_and_nonlinear_boundary_are_diagnostic():
    result = run()
    assert result["rank_tolerance_profile"]["tested"]
    assert result["verdict"] == "3D-DESIGN-EXPOSURE-PASS-STRUCTURAL-PARTIAL"
