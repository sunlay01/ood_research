import numpy as np

from ood_repr_reg.counterfactual_repairability import (
    ResponseControllerClass,
    coordinate_invariance_audit,
    optimize_controller,
)


def test_source_and_response_transport_preserve_class_optimum():
    audit = coordinate_invariance_audit(seed=20260915)
    assert audit["pass"]
    assert max(audit["errors"].values()) < 1e-5


def test_source_transport_preserves_induced_response_exactly():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2, 3))
    o = rng.normal(size=(2, 3))
    basis = [rng.normal(size=(2, 2)) for _ in range(2)]
    controller = ResponseControllerClass(basis)
    source = np.array([[2.0, 0.3], [-0.2, 1.4]])
    transformed = controller.transported(source_transform=source)
    base = optimize_controller(a, o, controller)
    moved = optimize_controller(a, source @ o, transformed)
    assert np.isclose(base.R_class_star, moved.R_class_star, atol=1e-6)
    assert np.allclose(moved.Pi_star @ (source @ o), base.Pi_star @ o, atol=1e-5)


def test_nonorthogonal_response_relabeling_is_rejected_for_euclidean_regret():
    controller = ResponseControllerClass([np.eye(2)])
    with np.testing.assert_raises(ValueError):
        controller.transported(response_transform=np.diag([2.0, 1.0]))
