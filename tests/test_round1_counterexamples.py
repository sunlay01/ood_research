import numpy as np

from ood_repr_reg.intervention_linear import (
    GaussianEnvironment,
    LinearGaussianSCM,
    affine_erm,
    causal_oracle_risk,
    irmv1_scale_penalty,
    risk,
    scalar_irmv1_zero_candidates,
    source_risk,
)


def test_irmv1_source_equivalence_does_not_imply_target_equivalence():
    scm = LinearGaussianSCM(
        loading=np.array([[1.0]]), beta=np.array([1.0]), sigma_xi=np.array([[2.0]]), sigma_y=0.1
    )
    sources = (
        GaussianEnvironment(np.array([[0.7]]), np.array([0.0]), np.array([[0.02]])),
        GaussianEnvironment(np.array([[-0.1]]), np.array([0.0]), np.array([[0.02]])),
    )
    target = GaussianEnvironment(np.array([[-1.0]]), np.array([0.0]), np.array([[0.02]]))
    candidate = scalar_irmv1_zero_candidates(scm, sources)[-1]
    erm = affine_erm(scm, sources)
    assert np.isclose(source_risk(scm, sources, candidate), source_risk(scm, sources, erm))
    assert irmv1_scale_penalty(scm, sources, candidate) < 1e-10
    assert risk(scm, target, candidate) - causal_oracle_risk(scm) > 1.0
