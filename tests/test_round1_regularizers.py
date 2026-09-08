import numpy as np

from ood_repr_reg.intervention_linear import GaussianEnvironment, LinearGaussianSCM, risk
from ood_repr_reg.round1_regularizers import (
    coral_state_response,
    irmv1_state_response,
    l2_state_response,
    regularizer_operator,
)


def _scm() -> LinearGaussianSCM:
    return LinearGaussianSCM(np.array([[1.0]]), np.array([1.0]), np.array([[0.25]]), 0.1)


def _env(r: float, mean: float = 0.0, variance: float = 0.5) -> GaussianEnvironment:
    return GaussianEnvironment(np.array([[r]]), np.array([mean]), np.array([[variance]]))


def test_irmv1_exposes_radial_response_and_relation_design():
    sources = tuple(_env(value) for value in (-0.6, 0.1, 0.8))
    result = irmv1_state_response(_scm(), sources, np.array([0.0, 0.45, 0.35]))
    assert result["observed_object"] == "per-environment radial risk response"
    assert result["relation_design"].shape == (3, 3)
    assert regularizer_operator("irmv1", _scm(), sources).rank == 3


def test_coral_only_observes_centered_covariance_differences():
    sources = (_env(0.0), _env(0.0, mean=1.0), _env(0.8))
    result = coral_state_response(_scm(), sources)
    assert result["observed_object"] == "representation covariance differences"
    assert result["blind_candidate"] == "mean shift and conditional label response"
    assert result["penalty"] >= 0.0


def test_coral_zero_mean_shift_observation_does_not_imply_zero_risk_change():
    scm = _scm()
    source = _env(0.0, mean=0.0)
    target = _env(0.0, mean=1.0)
    coral = coral_state_response(scm, (source, target))
    assert np.isclose(coral["penalty"], 0.0)
    assert not np.isclose(risk(scm, source, np.array([0.0, 0.0, 1.0])), risk(scm, target, np.array([0.0, 0.0, 1.0])))


def test_l2_is_global_and_nonselective():
    result = l2_state_response(np.array([0.0, 0.7, 0.4]))
    assert np.isclose(result["penalty"], 0.65)
    assert result["observed_object"] == "global coefficient norm"
    assert result["blind_candidate"] == "task/nuisance distinction and shift direction"
