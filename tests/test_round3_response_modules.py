import numpy as np

from ood_repr_reg.round3r_3b_benchmark import (
    ModuleEnvironment,
    environment_state,
    mixture_state,
    nonlinear_environment_state,
    risk_gradient,
    source_design,
    source_optimum,
)
from ood_repr_reg.round3r_3b_counterexamples import (
    identical_image_counterexample,
    partial_overlap_counterexample,
)
from ood_repr_reg.round3r_3b_discovery import bootstrap_stability, discover_subspace_modules
from ood_repr_reg.round3r_3b_geometry import (
    GeometryTolerance,
    filter_relevant_shifts,
    inclusion_residual,
    incremental_rank,
    intersection_dimension,
    mixed_shift_decomposition,
    principal_angles,
    relevance_gram,
    transformed_geometry,
    whitened_responses,
)
from ood_repr_reg.round3r_3b_mixed import mixed_shift_additivity
from ood_repr_reg.run_round3r_3b import _state_for


def test_relevance_gram_matches_whitened_response_inner_products():
    _, source, optimum, probes, gradients, responses, gram = _state_for(4, 2)
    assert np.allclose(responses.T @ responses, gram, atol=1e-12)
    assert np.allclose(responses, whitened_responses(2 * source.second, gradients))
    assert len(probes) == gram.shape[0]


def test_additivity_is_exact_for_moment_additive_mixed_response():
    _, source, optimum, probes, gradients, _, _ = _state_for(4, 2)
    first, second = gradients[:, 0], gradients[:, 15]
    target = source + (probes[0].target - source) + (probes[15].target - source)
    mixed = risk_gradient(optimum, source, target)
    check = mixed_shift_additivity(first, second, mixed)
    assert check["exact"]
    assert check["relative_residual"] < 1e-12


def test_direct_sum_and_overlap_have_distinct_identifiability_statuses():
    direct = {"a": np.array([[1.0], [0.0]]), "b": np.array([[0.0], [1.0]])}
    recovered = mixed_shift_decomposition(np.array([2.0, -3.0]), direct)
    assert recovered["unique_if_direct_sum"]
    assert recovered["residual_norm"] < 1e-12
    assert intersection_dimension(direct["a"], direct["b"]) == 0
    assert identical_image_counterexample()["response_only_identifiable"] is False
    assert partial_overlap_counterexample()["response_only_unique"] is False


def test_gram_is_invariant_under_general_predictor_reparameterization():
    _, source, _, _, gradients, _, _ = _state_for(4, 2)
    hessian = 2 * source.second
    rng = np.random.default_rng(4)
    transform = rng.normal(size=(hessian.shape[0], hessian.shape[0]))
    transform += 2 * np.eye(hessian.shape[0])
    result = transformed_geometry(hessian, gradients, transform)
    assert np.max(np.abs(result["gram"] - relevance_gram(hessian, gradients))) < 1e-10


def test_null_filter_removes_independent_nuisance_variance_probes():
    _, _, _, probes, gradients, _, gram = _state_for(4, 2)
    filtered = filter_relevant_shifts(gram)
    null_indices = [i for i, probe in enumerate(probes) if probe.mechanism == "noise"]
    assert not filtered["mask"][null_indices[0]]
    assert not filtered["mask"][null_indices[1]]
    assert np.allclose(gradients[:, null_indices], 0.0)


def test_nuisance_sweep_does_not_change_response_rank():
    ranks = []
    for dimension in (0, 4, 16, 64):
        ranks.append(np.linalg.matrix_rank(_state_for(dimension, 2)[5], tol=1e-9))
    assert ranks == [5, 5, 5, 5]


def test_redundant_shortcut_copies_do_not_create_proportional_response_rank():
    ranks = [np.linalg.matrix_rank(_state_for(4, 1, copies)[5], tol=1e-9) for copies in (1, 4, 16)]
    assert max(ranks) - min(ranks) <= 1


def test_bootstrap_discovery_is_label_free_and_has_declared_metadata():
    responses = _state_for(4, 2)[5]
    result = bootstrap_stability(responses, discover_subspace_modules, repeats=8, seed=3)
    assert result["uses_mechanism_labels"] is False
    assert 0.0 <= result["mean_pairwise_stability"] <= 1.0


def test_nonlinear_stress_state_is_psd_and_has_exact_dimensions():
    env = ModuleEnvironment(n_noise=4, shortcut_rhos=(0.7, 0.4), shortcut_means=(0.1, -0.2), shortcut_variances=(0.5, 0.6))
    state = nonlinear_environment_state(env, (0.1, -0.05))
    assert state.second.shape == (env.dimension, env.dimension)
    assert np.min(np.linalg.eigvalsh(state.second)) > -1e-10


def test_subspace_diagnostics_obey_rank_identities():
    first = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    second = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    assert intersection_dimension(first, second) == 1
    assert np.allclose(principal_angles(first, first), 0.0)
    assert inclusion_residual(first[:, :1], first) < 1e-12
    assert incremental_rank(first, second) == 1
