import numpy as np

from ood_repr_reg.intervention_linear import GaussianEnvironment, LinearGaussianSCM, irmv1_scale_penalty
from ood_repr_reg.round2_induced_cost import (
    coral_induced_cost,
    coral_scaling_law,
    irmv1_direct_induced_cost,
    irm_relation_operator,
    irmv1_induced_cost,
    l2_induced_cost,
)


def _scm():
    return LinearGaussianSCM(np.array([[1.0]]), np.array([1.0]), np.array([[0.25]]), 0.1)


def _env(r, variance=0.5):
    return GaussianEnvironment(np.array([[r]]), np.array([0.0]), np.array([[variance]]))


def test_l2_rank_one_induced_cost_closed_form():
    q = np.array([-0.3, 0.3])
    beta = np.array([1.0, 0.0])
    result = l2_induced_cost(q, beta)
    expected = q @ q + beta @ beta - 2.0 * abs(q[0] * beta[0])
    assert result.status == "exact"
    assert np.isclose(result.value, expected)


def test_coral_scaling_is_c_four_and_unconstrained_cost_degenerates():
    covariances = (np.eye(2), np.diag([1.0, 2.0]))
    base = coral_scaling_law(1.5, 1.0)
    assert np.isclose(coral_scaling_law(base, 2.0), 16.0 * base)
    result = coral_induced_cost(np.array([0.0, 1.0]), covariances)
    assert result.status == "degenerate_infimum"
    assert np.isclose(result.value, 0.0)
    fixed = coral_induced_cost(np.array([0.0, 1.0]), covariances, gauge=1.0)
    assert fixed.status == "conditional"


def test_irmv1_induced_cost_matches_original_population_penalty():
    sources = (_env(-0.6), _env(0.1), _env(0.8))
    w = np.array([0.0, 0.7, 0.3])
    result = irmv1_induced_cost(_scm(), sources, w)
    assert result.status == "conditional"
    assert np.isclose(result.value, irmv1_scale_penalty(_scm(), sources, w))
    assert np.linalg.matrix_rank(irm_relation_operator(sources)) == 3


def test_irmv1_direct_induced_cost_minimizes_sign_fiber():
    relations = np.array([-0.6, 0.1, 0.8])
    q = np.array([-0.7, 0.3])
    beta = 1.0
    result = irmv1_direct_induced_cost(q, beta, relations, nuisance_variance=0.5)
    values = []
    for sign in (-1.0, 1.0):
        w = np.array([beta, 0.0]) + sign * q
        derivatives = []
        for relation in relations:
            sigma = np.array([[1.0, relation], [relation, relation * relation + 0.5]])
            cross = np.array([beta, relation * beta])
            derivatives.append(2.0 * (w @ sigma @ w - w @ cross))
        values.append(float(np.mean(np.square(derivatives))))
    assert result.status == "exact_fiber_infimum"
    assert np.isclose(result.value, min(values))
