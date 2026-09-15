import numpy as np

from ood_repr_reg.counterfactual_repairability import (
    ResponseControllerClass,
    adaptive_repair_value,
    optimize_controller,
    standard_response_basis,
    toy_head_full,
    toy_information_limited,
    toy_spectral_slack,
)


def test_head_full_toy_separates_control_and_realization_layers():
    item = toy_head_full()
    assert np.isclose(item["head"].R_info, 0.0)
    assert np.isclose(item["head"].R_class_star, 0.5)
    assert np.isclose(item["full"].R_class_star, 0.0, atol=1e-8)
    assert np.isclose(item["R_actual"], 0.25)
    assert np.isclose(item["V_repr"], 0.5)
    assert np.isclose(item["Delta_real"], 0.25)
    assert item["actual_membership"]["contains"]


def test_information_limited_toy_is_not_reclassified_as_control_failure():
    item = toy_information_limited()
    assert np.isclose(item["full"].R_info, 1.0)
    assert np.isclose(item["full"].R_class_star, item["full"].R_info)
    assert np.isclose(item["full"].control_gap, 0.0)


def test_nonzero_recoverable_residual_can_attain_information_floor():
    item = toy_spectral_slack()
    assert item["summary"]["E_norm"] > 0.0
    assert np.isclose(item["zero"].R_class_star, item["zero"].R_info)
    assert item["summary"]["condition_holds"]
    assert item["summary"]["support_compatible"]


def test_nested_class_monotonicity_on_random_small_instances():
    rng = np.random.default_rng(20260915)
    from ood_repr_reg.counterfactual_repairability.diagnostics import random_nested_class

    for seed in rng.integers(0, 2**31 - 1, 100):
        item = random_nested_class(int(seed))
        assert item["small_class"].subset_of(item["large_class"])
        assert item["large"].R_class_star <= item["small"].R_class_star + 1e-6


def test_zero_class_and_full_class_shapes_are_explicit():
    zero = ResponseControllerClass([], response_shape=(2, 1), name="zero")
    assert np.allclose(zero.response_operator([]), np.zeros((2, 1)))
    full = ResponseControllerClass(standard_response_basis(2, 1), name="full")
    pi = np.array([[1.0], [-2.0]])
    assert full.contains(pi)
    assert np.isclose(adaptive_repair_value(np.zeros((2, 2)), np.ones((1, 2)), pi), 5.0)


def test_response_space_budget_is_parameterization_independent_for_same_operator():
    basis = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]
    bounded = ResponseControllerClass(basis, budget_type="operator", budget_radius=1.0)
    result = optimize_controller(np.array([[1.0, 1.0]]), np.eye(2), bounded)
    assert result.solver_status == "SUCCESS"
    assert result.Pi_star.shape == (1, 2)
    assert np.linalg.norm(result.Pi_star, ord=2) <= 1.0 + 1e-7
