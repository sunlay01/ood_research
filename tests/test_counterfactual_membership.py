import numpy as np

from ood_repr_reg.counterfactual_repairability import (
    ResponseControllerClass,
    audit_actual_controller,
    induced_response_controller,
    realization_gap,
    standard_response_basis,
    toy_head_full,
)


def test_actual_response_membership_is_required_before_realization_gap():
    item = toy_head_full()
    full = item["full_class"]
    actual = item["actual"]
    bridge = audit_actual_controller(actual, full)
    assert bridge.valid
    gap = realization_gap(item["response"], item["observation"], actual, item["full"], full)
    assert gap["valid"]
    assert np.isclose(gap["Delta_real"], 0.25)

    head = item["head_class"]
    invalid = realization_gap(item["response"], item["observation"], actual, item["head"], head)
    assert not invalid["valid"]
    assert invalid["Delta_real"] is None


def test_jacobian_bridge_uses_y_to_z_response_coordinates():
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0]])
    parameter_controller = np.array([[0.0], [1.0]])
    induced = induced_response_controller(jacobian, parameter_controller)
    assert np.allclose(induced, np.array([[0.0], [2.0]]))
    controller = ResponseControllerClass([induced], name="jacobian")
    actual = 0.5 * induced
    audit = audit_actual_controller(actual, controller, jacobian=jacobian,
                                    parameter_controller=0.5 * parameter_controller)
    assert audit.valid
    assert audit.audit["induced_response_matches"]


def test_full_standard_basis_contains_every_response_map():
    controller = ResponseControllerClass(standard_response_basis(2, 3))
    value = np.arange(6.0).reshape(2, 3)
    audit = controller.membership_audit(value)
    assert audit["contains"]
    assert audit["membership_residual"] < 1e-12
