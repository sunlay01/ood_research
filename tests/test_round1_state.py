import numpy as np

from ood_repr_reg.intervention_linear import GaussianEnvironment, LinearGaussianSCM, risk
from ood_repr_reg.round1_state import (
    component_transport,
    componentwise_bound,
    conditional_loss_risk,
    environment_state,
    exposure_identifiability,
    project_quadratic_state,
    risk_pairing,
    risk_quotient_basis,
    risk_state,
    recover_target_moment_from_sources,
    source_observation_operator,
    source_environment_moments,
)


def _scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(np.array([[1.0]]), np.array([1.0]), np.array([[0.25]]), 0.1)


def _env(r: float, mean: float = 0.0, variance: float = 0.5) -> GaussianEnvironment:
    return GaussianEnvironment(np.array([[r]]), np.array([mean]), np.array([[variance]]))


def test_risk_state_pairing_and_quotient_are_exact():
    scm = _scm()
    source = _env(0.2)
    target = _env(-0.7, mean=0.3, variance=0.9)
    w = np.array([0.1, 0.6, 0.4])
    state = risk_state(scm, w, source)
    source_moment = environment_state(scm, source)
    target_moment = environment_state(scm, target)
    assert np.isclose(risk_pairing(state, target_moment) + scm.sigma_y**2, risk(scm, target, w))
    basis = risk_quotient_basis((source_moment, target_moment))
    projected = project_quadratic_state(state.quadratic, basis)
    assert np.isclose(risk_pairing(projected, target_moment), risk_pairing(state, target_moment))
    assert np.isclose(risk_pairing(projected, source_moment), risk_pairing(state, source_moment))


def test_conditional_loss_identity_is_weighted_integration():
    losses = np.array([0.5, 2.0, 4.0])
    weights = np.array([0.2, 0.3, 0.5])
    assert np.isclose(conditional_loss_risk(losses, weights), 2.7)


def test_target_in_source_span_is_recoverable():
    scm = _scm()
    sources = (_env(-0.6), _env(0.1), _env(0.8))
    target = _env(0.25)
    recovery = recover_target_moment_from_sources(
        source_environment_moments(scm, sources), environment_state(scm, target)
    )
    assert recovery.exact
    assert recovery.residual_norm < 1e-10


def test_exposure_is_not_identifiability_for_a_low_rank_source():
    scm = _scm()
    sources = (_env(0.0), _env(0.0, mean=1.0))
    target = (_env(0.8),)
    operator = source_observation_operator(source_environment_moments(scm, sources))
    report = exposure_identifiability(operator, source_environment_moments(scm, target))
    assert report.source_kernel_dimension > 0
    assert report.target_blind_norm.shape == (1,)


def test_component_transport_and_bound_close():
    vector = np.array([1.0, -0.5])
    delta = np.array([[0.2, 0.1], [0.1, -0.3]])
    projectors = (np.diag([1.0, 0.0]), np.diag([0.0, 1.0]))
    diagonal, interaction = component_transport(vector, delta, projectors)
    exact = vector @ delta @ vector
    assert np.isclose(diagonal.sum() + interaction.sum(), exact)
    assert abs(exact) <= componentwise_bound(vector, delta, projectors) + 1e-12
