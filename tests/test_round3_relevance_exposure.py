import numpy as np

from ood_repr_reg.round3r_3a_benchmark import (
    base_environment,
    source_environments,
    source_reference,
    target_emergent_shift,
    target_noise_variance_shift,
    target_shortcut_shift,
    target_stable_burden_shift,
)
from ood_repr_reg.round3r_3a_exact_vulnerability import exact_vulnerability, numerical_vulnerability_check
from ood_repr_reg.round3r_3a_exposure import structural_exposure
from ood_repr_reg.round3r_3a_relevance import fit_scaling_exponent, leading_relevance, local_vulnerability
from ood_repr_reg.round3r_3a_relevance import curvature_coefficient
from ood_repr_reg.round3r_3a_residual_coupling import residual_identity_error
from ood_repr_reg.round3r_3a_source_geometry import hessian, moments, source_excess, source_optimum
from ood_repr_reg.run_round3r_3a import run


def _setting(rho=0.8, n_noise=4, exposed=True):
    base = base_environment(n_noise=n_noise, rho=rho)
    sources = source_environments(base, exposed=exposed)
    optimum, source = source_optimum(sources)
    reference = source_reference(sources)
    return base, sources, reference, optimum, source


def test_source_normal_equation_and_positive_hessian():
    _, _, _, optimum, source = _setting()
    assert np.allclose(source.second @ optimum, source.xy)
    assert np.min(np.linalg.eigvalsh(hessian(source))) > 0


def test_quadratic_source_excess_is_exact_ellipsoid():
    _, _, _, optimum, source = _setting()
    delta = np.arange(optimum.size, dtype=float) / 100.0
    assert np.isclose(source_excess(optimum + delta, optimum, source), delta @ source.second @ delta)


def test_residual_coupling_identity_is_exact():
    base, _, reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    assert residual_identity_error(optimum, source, target) < 1e-12


def test_source_unused_target_emergent_is_relevant():
    base, _, reference, optimum, source = _setting(rho=0.0)
    assert abs(optimum[2]) < 1e-12
    target = moments(target_emergent_shift(reference, 0.7))
    assert leading_relevance(optimum, source, target) > 1e-6


def test_mechanism_off_shortcut_shift_is_first_order_null():
    _, _, reference, optimum, source = _setting(rho=0.0)
    target = moments(target_shortcut_shift(reference, 0.0))
    assert leading_relevance(optimum, source, target) < 1e-12
    values = np.asarray([exact_vulnerability(e, optimum, source, target) for e in (1e-6, 1e-5, 1e-4)])
    assert np.allclose(values, 0.0)


def test_independent_noise_variance_shift_has_zero_gradient():
    _, _, reference, optimum, source = _setting()
    target = moments(target_noise_variance_shift(reference))
    assert np.linalg.norm((target.xy - target.second @ optimum)) < 1e-12
    assert leading_relevance(optimum, source, target) < 1e-12


def test_exact_vulnerability_matches_local_leading_term_at_small_epsilon():
    _, _, reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    exact = exact_vulnerability(1e-8, optimum, source, target)
    local = local_vulnerability(1e-8, optimum, source, target)
    assert abs(exact - local) / max(exact, 1e-30) < 1e-3


def test_exact_vulnerability_is_covered_by_quadratic_two_sided_bound():
    _, _, reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    epsilon = 1e-3
    exact = exact_vulnerability(epsilon, optimum, source, target)
    local = local_vulnerability(epsilon, optimum, source, target)
    curvature = curvature_coefficient(source, target)
    assert abs(exact - local) <= epsilon * curvature + 1e-10


def test_trust_region_value_matches_independent_numerical_check():
    _, _, reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    exact = exact_vulnerability(1e-3, optimum, source, target)
    checked = numerical_vulnerability_check(1e-3, optimum, source, target, starts=12)
    assert abs(exact - checked) / exact < 1e-6


def test_first_order_null_has_exact_linear_epsilon_scaling_when_curvature_is_nonzero():
    _, _, reference, optimum, source = _setting()
    target = moments(target_noise_variance_shift(reference))
    curvature = curvature_coefficient(source, target)
    assert curvature > 0
    epsilon = 1e-3
    assert np.isclose(exact_vulnerability(epsilon, optimum, source, target), epsilon * curvature, rtol=1e-8, atol=1e-10)


def test_zero_source_usage_can_have_nonzero_gradient_in_explicit_moment_counterexample():
    from ood_repr_reg.round3r_3a_source_geometry import PopulationMoments, risk_difference_gradient

    source = PopulationMoments(np.diag([1.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0]), 1.0)
    target = PopulationMoments(np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 1.0], [0.0, 1.0, 2.0]]), np.array([0.0, 1.0, 1.0]), 1.0)
    optimum = np.linalg.solve(source.second, source.xy)
    gradient = risk_difference_gradient(optimum, source, target)
    assert np.isclose(optimum[2], 0.0)
    assert np.isclose(gradient[2], -1.0)


def test_scaling_exponent_distinguishes_first_order_relevance():
    _, _, reference, optimum, source = _setting()
    target = moments(target_shortcut_shift(reference, -0.8))
    eps = np.asarray((1e-6, 1e-5, 1e-4, 1e-3))
    values = np.asarray([exact_vulnerability(e, optimum, source, target) for e in eps])
    assert abs(fit_scaling_exponent(eps, values) - 0.5) < 0.05


def test_target_stable_burden_has_large_common_burden_but_low_discrimination():
    _, _, reference, optimum, source = _setting()
    target = moments(target_stable_burden_shift(reference, 1.0))
    assert leading_relevance(optimum, source, target) < 1e-12
    assert target.y2 > source.y2


def test_exposure_is_separate_from_relevance():
    base, sources, reference, optimum, source = _setting()
    emergent = target_emergent_shift(reference)
    shortcut = target_shortcut_shift(reference, -0.8)
    assert structural_exposure(sources, reference, emergent)["score"] < 1e-12
    assert structural_exposure(sources, reference, shortcut)["score"] > 0.99


def test_noise_dimension_and_redundant_shortcut_benchmarks_are_constructible():
    assert base_environment(n_noise=256).dimension == 261
    assert base_environment(shortcut_count=2).dimension == 10
    assert base_environment(shortcut_count=2, redundant=True).shortcut_shared_sigma > 0


def test_runner_records_relevance_and_exposure_artifacts_without_target_selection():
    result = run(seed_count=2)
    assert result["scope"]["regularizer_control"] is False
    assert result["acceptance"]["identity_max_error"] < 1e-12
    assert result["acceptance"]["second_shortcut_adds_rank"] is True
    assert result["noise_explosion"][-1]["relevance_effective_rank"] == 2
