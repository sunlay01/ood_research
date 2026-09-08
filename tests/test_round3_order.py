import numpy as np

from ood_repr_reg.round3_3b_derivatives import (
    OrderDirection,
    OrderEnvironment,
    moment_polynomial,
    pure_family_directions,
    risk_derivative,
)
from ood_repr_reg.round3_3b_finite_difference import (
    analytic_response_matrix,
    central_difference_samples,
    finite_difference_response_matrix,
    validation_error,
)
from ood_repr_reg.round3_3b_filtration import order_overlap, order_ranks, project_residual
from ood_repr_reg.round3_shift_ensemble import base_environment


def _base() -> OrderEnvironment:
    value = base_environment()
    return OrderEnvironment(value.mu_c, value.sigma_c, value.gamma, value.b, value.mu_xi, value.sigma_xi, value.beta, value.noise_variance)


def test_third_order_stencil_has_correct_sign_and_coefficients():
    samples = {index: np.asarray(float(index) ** 3) for index in (-2, -1, 1, 2)}
    assert np.isclose(central_difference_samples(samples, 3, 1.0), 6.0)


def test_analytic_risk_derivatives_match_finite_difference():
    base = _base()
    direction = pure_family_directions(base)["relation"][0]
    weights = (np.array([0.1, 1.0, -0.7, 0.2, -0.3]),)
    for order in (1, 2, 3):
        analytic = analytic_response_matrix(weights, base, (direction,), order)
        estimate = finite_difference_response_matrix(weights, base, (direction,), 0.05, order)
        assert validation_error(analytic, estimate) < (1e-7 if order < 3 else 1e-5)


def test_structural_path_is_polynomial_and_third_order_can_be_nonzero():
    base = _base()
    direction = pure_family_directions(base)["core_mean"][0]
    degree = len(moment_polynomial(base, direction)[0]) - 1
    assert degree >= 2
    assert risk_derivative(np.array([0.1, 1.0, -0.7, 0.2, -0.3]), base, direction, 4) == 0.0


def test_pure_family_direction_sampler_has_registered_coverage():
    families = pure_family_directions(_base(), samples_per_family=100)
    assert {name: len(directions) for name, directions in families.items()} == {
        "core_mean": 100,
        "core_covariance": 100,
        "relation": 100,
        "b": 100,
        "mu_xi": 100,
        "sigma_xi": 100,
        "task": 100,
    }


def test_order_filtration_and_overlap_are_explicit():
    first = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    second = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    result = order_ranks({1: first, 2: second})
    assert result["cumulative_ranks"] == {"1": 2, "2": 3}
    assert result["incremental_ranks"] == {"1": 2, "2": 1}
    assert order_overlap(first, second)["intersection_dimension"] == 1
    assert project_residual(first, np.column_stack((first, second))) < 1e-12


def test_order_environment_matches_existing_population_risk_at_valid_points():
    value = base_environment()
    base = _base()
    direction = pure_family_directions(base)["sigma_xi"][0]
    assert base.at(direction, 0.2).is_valid_covariance()
    assert np.isfinite(base.at(direction, 0.2).sigma_xi).all()
