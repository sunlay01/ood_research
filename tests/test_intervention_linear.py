import numpy as np

from ood_repr_reg.intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    augmented_moment,
    affine_erm,
    causal_oracle_risk,
    correlation_robust_risk,
    effective_nuisance_sensitivity,
    effective_predictor,
    erm_stationarity_residual,
    gradient_alignment_penalty,
    irmv1_radial_response,
    irmv1_scale_penalty,
    moment_robust_risk,
    nuisance_transport_bound,
    nuisance_observability,
    observational_oracle_risk,
    projected_transport_components,
    reparameterize_linear_representation,
    representation_irmv1_scale_penalty,
    representation_risk,
    residual_vector,
    risk,
    risk_transport_components,
    scalar_irmv1_nuisance_squared_bound,
    scalar_relation_response,
    scalar_irmv1_zero_candidates,
    source_risk,
)


def _scalar_scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[0.25]]),
        sigma_y=0.1,
    )


def _scalar_environment(relation: float, *, mean: float = 0.0, variance: float = 0.5) -> GaussianEnvironment:
    return GaussianEnvironment(
        relation=np.array([[relation]]),
        mean=np.array([mean]),
        sigma_a=np.array([[variance]]),
    )


def test_exact_risk_transport_identity() -> None:
    scm = _scalar_scm()
    source = _scalar_environment(0.7, mean=0.2)
    target = _scalar_environment(-0.4, mean=-0.1, variance=0.8)
    w = np.array([0.3, 0.4, 0.6])

    b = residual_vector(scm, source, w)
    expected = b @ (augmented_moment(scm, target) - augmented_moment(scm, source)) @ b

    assert np.isclose(risk(scm, target, w) - risk(scm, source, w), expected)


def test_correlation_ball_formula_matches_scalar_endpoints() -> None:
    scm = _scalar_scm()
    w = np.array([0.0, 0.35, 0.7])
    center = np.array([[0.4]])
    radius = 0.5
    robust = correlation_robust_risk(scm, w, center, radius, np.array([[0.6]]))
    endpoints = [
        risk(scm, _scalar_environment(value, variance=0.6), w)
        for value in (center.item() - radius, center.item() + radius)
    ]

    assert np.isclose(robust, max(endpoints))


def test_moment_ball_formula_matches_scalar_extrema() -> None:
    scm = _scalar_scm()
    w = np.array([0.2, 0.4, -0.5])
    robust = moment_robust_risk(
        scm,
        w,
        relation=np.array([[0.35]]),
        mean_center=np.array([0.1]),
        mean_radius=0.4,
        sigma_a_center=np.array([[0.6]]),
        covariance_radius=0.3,
    )
    values = [
        risk(scm, _scalar_environment(0.35, mean=mean, variance=variance), w)
        for mean in (-0.3, 0.5)
        for variance in (0.3, 0.9)
    ]

    assert np.isclose(robust, max(values))


def test_partial_observation_makes_source_erm_use_nuisance() -> None:
    scm = _scalar_scm()
    sources = (_scalar_environment(0.8), _scalar_environment(0.3))
    erm = affine_erm(scm, sources)

    assert abs(erm[-1]) > 1e-6
    assert observational_oracle_risk(scm, sources) < risk(scm, sources[0], np.zeros(3))


def test_constant_predictor_has_zero_correlation_degradation_but_high_causal_excess() -> None:
    scm = _scalar_scm()
    source = _scalar_environment(0.6)
    w = np.zeros(3)
    robust = correlation_robust_risk(
        scm, w, np.array([[0.0]]), 1.0, np.array([[0.5]])
    )

    assert np.isclose(robust, risk(scm, source, w))
    assert robust - causal_oracle_risk(scm) > 0.5


def test_gradient_alignment_zero_set_contains_nuisance_free_invariant_predictor() -> None:
    scm = _scalar_scm()
    sources = (_scalar_environment(0.8), _scalar_environment(0.2))
    w = np.array([0.0, 1.0, 0.0])

    assert np.isclose(gradient_alignment_penalty(scm, sources, w), 0.0, atol=1e-12)
    assert source_risk(scm, sources, w) > observational_oracle_risk(scm, sources)


def test_irmv1_has_zero_penalty_at_u_only_source_optimum() -> None:
    scm = _scalar_scm()
    sources = (_scalar_environment(0.8), _scalar_environment(0.2))
    u_only = np.array([0.0, 1.0 / 1.25, 0.0])

    assert np.isclose(irmv1_scale_penalty(scm, sources, u_only), 0.0, atol=1e-12)


def test_irmv1_has_source_optimal_harmful_nuisance_blind_direction() -> None:
    """A strict C002-IRM counterexample with exact rational parameters.

    Sources use nuisance correlations 7/10 and -1/10.  The population source
    ERM has (w_U, w_A) = (1/4, 5/6), and the standard scalar-scale IRMv1
    penalty vanishes in both environments.  A task-preserving target with
    relation -1 is inside the unit correlation ball around zero and has a
    strictly larger causal excess risk.
    """
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[2.0]]),
        sigma_y=0.1,
    )
    sources = (
        _scalar_environment(0.7, variance=0.02),
        _scalar_environment(-0.1, variance=0.02),
    )
    w = np.array([0.0, 0.25, 5.0 / 6.0])
    target = _scalar_environment(-1.0, variance=0.02)

    assert np.allclose(affine_erm(scm, sources), w)
    assert np.isclose(source_risk(scm, sources, w), observational_oracle_risk(scm, sources))
    assert np.isclose(irmv1_scale_penalty(scm, sources, w), 0.0, atol=1e-12)
    assert risk(scm, target, w) - causal_oracle_risk(scm) > 2.6
    assert any(np.allclose(candidate, w) for candidate in scalar_irmv1_zero_candidates(scm, sources))


def test_vector_source_unobservable_direction_has_zero_restricted_singular_value() -> None:
    scm = LinearGaussianSCM(
        loading=np.eye(2),
        beta=np.array([1.0, 1.0]),
        sigma_xi=np.eye(2) * 0.2,
        sigma_y=0.1,
    )
    sources = (
        GaussianEnvironment(np.diag([0.8, 0.4]), np.zeros(2), np.eye(2) * 0.4),
        GaussianEnvironment(np.diag([0.2, 0.4]), np.zeros(2), np.eye(2) * 0.4),
    )
    direct, adjusted = nuisance_observability(scm, sources)

    assert np.isclose(direct, 0.0, atol=1e-12)
    assert np.isclose(adjusted, 0.0, atol=1e-12)


def test_transport_components_telescope_relation_mean_and_covariance_changes() -> None:
    scm = _scalar_scm()
    source = _scalar_environment(0.7, mean=0.2, variance=0.5)
    target = _scalar_environment(-0.3, mean=-0.4, variance=0.9)
    w = np.array([0.25, 0.4, 0.6])

    components = risk_transport_components(scm, source, target, w)

    assert np.isclose(components.total, risk(scm, target, w) - risk(scm, source, w))
    assert not np.isclose(components.relation, 0.0)
    assert not np.isclose(components.mean, 0.0)
    assert not np.isclose(components.covariance, 0.0)


def test_projected_transport_components_recover_total_and_separate_blind_part() -> None:
    scm = LinearGaussianSCM(
        loading=np.eye(2),
        beta=np.array([1.0, -0.5]),
        sigma_xi=np.eye(2) * 0.2,
        sigma_y=0.1,
    )
    source = GaussianEnvironment(np.diag([0.7, 0.2]), np.zeros(2), np.eye(2) * 0.4)
    target = GaussianEnvironment(np.diag([-0.2, -0.6]), np.zeros(2), np.eye(2) * 0.6)
    w = np.array([0.0, 0.4, -0.2, 0.5, 0.3])
    projector = np.diag([1.0, 0.0])

    components = projected_transport_components(scm, source, target, w, projector)

    assert np.isclose(components.total, risk(scm, target, w) - risk(scm, source, w))
    assert not np.isclose(components.controlled, 0.0)
    assert not np.isclose(components.blind, 0.0)


def test_erm_only_imposes_source_mixture_stationarity_not_environment_radial_optimality() -> None:
    scm = _scalar_scm()
    sources = (_scalar_environment(0.8), _scalar_environment(0.3))
    erm = affine_erm(scm, sources)
    response = irmv1_radial_response(scm, sources, erm)

    assert erm_stationarity_residual(scm, sources, erm) < 1e-12
    assert response.penalty > 1e-6
    assert np.linalg.norm(response.tangential_gradient_norms) > 1e-6


def test_symmetric_relation_sources_are_a_restricted_erm_generalization_case() -> None:
    scm = _scalar_scm()
    sources = (_scalar_environment(-0.8), _scalar_environment(0.8))
    target = _scalar_environment(-1.4, mean=0.7, variance=1.2)
    erm = affine_erm(scm, sources)

    assert np.isclose(erm[-1], 0.0, atol=1e-12)
    assert np.isclose(risk(scm, target, erm), source_risk(scm, sources, erm))


def test_three_relation_design_controls_scalar_nuisance_sensitivity() -> None:
    scm = _scalar_scm()
    sources = tuple(_scalar_environment(relation) for relation in (-0.6, 0.1, 0.8))
    w = np.array([0.0, 0.45, 0.35])
    response = scalar_relation_response(scm, sources, w)
    bound = scalar_irmv1_nuisance_squared_bound(scm, sources, w)

    assert response.singular_value > 1e-6
    assert np.isclose(response.coefficients[-1], w[-1] ** 2)
    assert np.allclose(2.0 * response.values, irmv1_radial_response(scm, sources, w).derivatives)
    assert w[-1] ** 2 <= bound + 1e-12


def test_full_rank_scalar_irmv1_bridge_enters_a_valid_target_transport_bound() -> None:
    scm = _scalar_scm()
    sources = tuple(_scalar_environment(relation) for relation in (-0.6, 0.1, 0.8))
    source = sources[0]
    target = _scalar_environment(-1.1, mean=0.5, variance=1.0)
    w = np.array([0.0, 0.45, 0.35])
    nuisance_bound = np.sqrt(scalar_irmv1_nuisance_squared_bound(scm, sources, w))

    bound = nuisance_transport_bound(scm, source, target, w, nuisance_bound)

    assert abs(risk(scm, target, w) - risk(scm, source, w)) <= bound + 1e-12


def test_three_relation_zero_penalty_nontrivial_solution_is_nuisance_free_and_shift_stable() -> None:
    scm = _scalar_scm()
    sources = tuple(_scalar_environment(relation) for relation in (-0.6, 0.1, 0.8))
    u_only = np.array([0.0, 1.0 / 1.25, 0.0])
    target = _scalar_environment(-1.4, mean=0.7, variance=1.2)

    assert np.isclose(irmv1_scale_penalty(scm, sources, u_only), 0.0, atol=1e-12)
    assert np.isclose(scalar_irmv1_nuisance_squared_bound(scm, sources, u_only), 0.0, atol=1e-12)
    assert source_risk(scm, sources, u_only) < source_risk(scm, sources, np.zeros(3))
    assert np.isclose(risk(scm, target, u_only), risk(scm, sources[0], u_only))


def test_two_relation_design_is_insufficient_for_scalar_irmv1_nuisance_bound() -> None:
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]),
        beta=np.array([1.0]),
        sigma_xi=np.array([[2.0]]),
        sigma_y=0.1,
    )
    sources = (_scalar_environment(0.7, variance=0.02), _scalar_environment(-0.1, variance=0.02))
    w = np.array([0.0, 0.25, 5.0 / 6.0])

    assert irmv1_scale_penalty(scm, sources, w) < 1e-12
    with np.testing.assert_raises(ValueError):
        scalar_irmv1_nuisance_squared_bound(scm, sources, w)


def test_effective_predictor_and_nuisance_sensitivity_are_reparameterization_invariant() -> None:
    representation = np.array([[1.0, 0.2, 0.7], [-0.3, 0.5, 0.4]])
    head = np.array([0.6, -0.8])
    transform = np.array([[2.0, -0.5], [0.25, 1.5]])
    scm = _scalar_scm()
    environments = (_scalar_environment(0.8), _scalar_environment(0.3))
    transformed_representation, transformed_head = reparameterize_linear_representation(
        representation, head, transform
    )

    assert np.allclose(
        effective_predictor(representation, head),
        effective_predictor(transformed_representation, transformed_head),
    )
    assert np.allclose(
        effective_nuisance_sensitivity(representation, head, 2),
        effective_nuisance_sensitivity(transformed_representation, transformed_head, 2),
    )
    assert np.isclose(
        representation_risk(scm, environments[0], representation, head),
        representation_risk(scm, environments[0], transformed_representation, transformed_head),
    )
    assert np.isclose(
        representation_irmv1_scale_penalty(scm, environments, representation, head),
        representation_irmv1_scale_penalty(scm, environments, transformed_representation, transformed_head),
    )


def test_rank_deficient_representation_can_retain_effective_nuisance_sensitivity() -> None:
    representation = np.array([[1.0, 0.4, 0.9]])
    head = np.array([2.0])

    assert np.linalg.matrix_rank(representation) == 1
    assert np.allclose(effective_nuisance_sensitivity(representation, head, 2), np.array([1.8]))
