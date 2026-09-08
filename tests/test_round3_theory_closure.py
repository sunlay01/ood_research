from __future__ import annotations

import numpy as np

from ood_repr_reg.round3r_3a_exact_vulnerability import exact_vulnerability
from ood_repr_reg.round3r_3a_relevance import (
    curvature_coefficient,
    leading_relevance,
    local_vulnerability,
    response_polynomial,
)
from ood_repr_reg.round3r_3a_source_geometry import (
    PopulationMoments,
    hessian,
    moments,
    source_excess,
    source_optimum,
)
from ood_repr_reg.round3r_3a_benchmark import (
    base_environment,
    source_environments,
    source_reference,
    target_noise_variance_shift,
    target_shortcut_shift,
)


def _setting():
    base = base_environment(n_noise=2, rho=0.8)
    sources = source_environments(base, exposed=True)
    optimum, source = source_optimum(sources)
    return source_reference(sources), optimum, source


def test_epsilon_zero_and_exact_source_ellipsoid():
    reference, optimum, source = _setting()
    delta = np.array([0.1] * optimum.size)
    assert source_excess(optimum, optimum, source) == 0.0
    assert np.isclose(source_excess(optimum + delta, optimum, source), delta @ source.second @ delta)
    target = moments(target_shortcut_shift(reference, -0.8))
    assert exact_vulnerability(0.0, optimum, source, target) == 0.0


def test_shift_polynomial_is_exact_and_quadratic_matrix_is_symmetric():
    reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    delta = np.linspace(-0.02, 0.03, optimum.size)
    lhs = response_polynomial(delta, optimum, source, target)
    dM = target.second - source.second
    g = 2.0 * dM @ optimum - 2.0 * (target.xy - source.xy)
    rhs = g @ delta + delta @ dM @ delta
    assert np.isclose(lhs, rhs)
    assert np.allclose(dM, dM.T)


def test_first_order_null_is_not_finite_vulnerability_null():
    reference, optimum, source = _setting()
    target = moments(target_noise_variance_shift(reference))
    assert leading_relevance(optimum, source, target) < 1e-12
    curvature = curvature_coefficient(source, target)
    assert curvature > 0.0
    epsilon = 1e-4
    assert np.isclose(
        exact_vulnerability(epsilon, optimum, source, target), epsilon * curvature, rtol=1e-8, atol=1e-11
    )


def test_nuisance_augmentation_preserves_leading_relevance():
    rng = np.random.default_rng(3)
    e = rng.normal(size=(3, 3))
    n = rng.normal(size=(4, 4))
    me = e.T @ e + np.eye(3)
    mn = n.T @ n + np.eye(4)
    ge = rng.normal(size=3)
    ma = np.block([[me, np.zeros((3, 4))], [np.zeros((4, 3)), mn]])
    ga = np.concatenate([ge, np.zeros(4)])
    assert np.isclose(ga @ np.linalg.solve(ma, ga), ge @ np.linalg.solve(me, ge))


def test_metric_coordinate_invariance():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(4, 4))
    m = a.T @ a + np.eye(4)
    g = rng.normal(size=4)
    t = rng.normal(size=(4, 4))
    while abs(np.linalg.det(t)) < 0.1:
        t = rng.normal(size=(4, 4))
    mp = np.linalg.inv(t).T @ m @ np.linalg.inv(t)
    gp = np.linalg.inv(t).T @ g
    assert np.isclose(g @ np.linalg.solve(m, g), gp @ np.linalg.solve(mp, gp))


def test_common_burden_is_not_model_discrimination():
    m = np.eye(2)
    xy = np.array([0.25, -0.5])
    source = PopulationMoments(m, xy, 1.0)
    target = PopulationMoments(m, xy, 4.0)
    optimum = xy.copy()
    assert np.isclose(abs((optimum @ m @ optimum - 2 * optimum @ xy + target.y2) -
                         (optimum @ m @ optimum - 2 * optimum @ xy + source.y2)), 3.0)
    assert leading_relevance(optimum, source, target) == 0.0
    assert exact_vulnerability(0.1, optimum, source, target) == 0.0


def test_counterexample_zero_source_usage_nonzero_target_gradient_is_realizable():
    source = PopulationMoments(np.diag([2.0, 1.0]), np.array([1.0, 0.0]), 1.0)
    target = PopulationMoments(np.array([[2.0, 1.0], [1.0, 2.0]]), np.array([1.0, 1.0]), 1.0)
    optimum = np.linalg.solve(source.second, source.xy)
    gradient = 2.0 * (target.second - source.second) @ optimum - 2.0 * (target.xy - source.xy)
    explained_target = target.xy @ np.linalg.solve(target.second, target.xy)
    assert np.isclose(optimum[1], 0.0)
    assert np.isclose(gradient[1], -1.0)
    assert explained_target < target.y2


def test_hessian_normalization_matches_leading_relevance():
    reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    dM = target.second - source.second
    g = 2.0 * dM @ optimum - 2.0 * (target.xy - source.xy)
    assert np.isclose(
        np.sqrt(g @ np.linalg.solve(source.second, g)),
        np.sqrt(2.0) * leading_relevance(optimum, source, target),
    )
    assert np.isclose(
        local_vulnerability(1e-4, optimum, source, target),
        np.sqrt(2e-4) * leading_relevance(optimum, source, target),
    )
