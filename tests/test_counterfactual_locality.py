import numpy as np

from ood_repr_reg.counterfactual_repairability import (
    ResponseControllerClass,
    induced_response_controller,
)


def test_ift_response_is_not_double_multiplied_by_a_parameter_jacobian():
    """The bridge records an already induced Pi as Y -> Z, not theta -> Z."""
    jacobian = np.array([[2.0, 0.0]])
    parameter_controller = np.array([[0.5], [0.0]])
    induced = induced_response_controller(jacobian, parameter_controller)
    controller = ResponseControllerClass([induced])
    actual_ift = induced.copy()
    assert controller.contains(actual_ift)
    assert actual_ift.shape == (1, 1)
    assert jacobian.shape != actual_ift.shape


def test_response_budget_is_checked_after_induction():
    jacobian = np.diag([3.0, 1.0])
    parameter_controller = np.eye(2)
    induced = induced_response_controller(jacobian, parameter_controller)
    controller = ResponseControllerClass([induced], budget_type="operator", budget_radius=2.0)
    assert not controller.contains(induced)
    assert controller.contains(0.5 * induced)
