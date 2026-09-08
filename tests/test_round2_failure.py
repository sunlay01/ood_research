import numpy as np

from ood_repr_reg.intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    causal_oracle_risk,
    irmv1_scale_penalty,
    risk,
    scalar_irmv1_zero_candidates,
    source_risk,
)
from ood_repr_reg.round2_induced_cost import classify_failure
from ood_repr_reg.round2_state import source_observation_map, source_target_ambiguity
from ood_repr_reg.intervention_linear import augmented_moment


def test_irm_branch_is_target_visible_and_zero_penalty():
    scm = LinearGaussianSCM(np.array([[1.0]]), np.array([1.0]), np.array([[2.0]]), 0.1)
    sources = (
        GaussianEnvironment(np.array([[0.7]]), np.array([0.0]), np.array([[0.02]])),
        GaussianEnvironment(np.array([[-0.1]]), np.array([0.0]), np.array([[0.02]])),
    )
    target = GaussianEnvironment(np.array([[-1.0]]), np.array([0.0]), np.array([[0.02]]))
    candidate = scalar_irmv1_zero_candidates(scm, sources)[-1]
    assert irmv1_scale_penalty(scm, sources, candidate) < 1e-10
    assert source_risk(scm, sources, candidate) < 0.6
    assert risk(scm, target, candidate) - causal_oracle_risk(scm) > 1.0


def test_failure_classifier_preserves_parameterization_degeneracy():
    assert classify_failure(parameterization_degenerate=True, source_unidentified=False, regularizer_blind=False, destructive=False) == "quotient-external/parameterization-degeneracy"
    assert classify_failure(parameterization_degenerate=False, source_unidentified=True, regularizer_blind=False, destructive=False) == "source-unidentified"
    assert classify_failure(parameterization_degenerate=False, source_unidentified=False, regularizer_blind=True, destructive=False) == "regularizer-blind"

