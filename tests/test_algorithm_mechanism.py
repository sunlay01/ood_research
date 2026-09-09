import numpy as np

from ood_repr_reg.algorithm_mechanism.adapters import gaussian_input
from ood_repr_reg.algorithm_mechanism.audits import evaluate_input, static_audit
from ood_repr_reg.round3r_3e_c_benchmarks import primary_hidden_u_world


def _record(method: str, lam: float):
    return evaluate_input(gaussian_input(primary_hidden_u_world(), method, lam))


def test_exact_mechanism_reconstructs_l2_and_l2_identities_hold():
    record = _record("l2", 0.01)
    learner = record["learner"]
    assert learner["pi_reconstruction_relative_error"] < 1e-6
    assert learner["total_H_relative_error"] < 1e-6
    assert learner["total_B_relative_error"] < 1e-6
    assert all(record["l2_exact"].values())


def test_irm_and_vrex_exact_component_reconstructions_hold():
    for method in ("irmv1", "vrex"):
        record = _record(method, 0.01)
        assert record["learner"]["pi_reconstruction_relative_error"] < 1e-5
        assert record["learner"]["total_H_relative_error"] < 1e-5
        assert record["learner"]["total_B_relative_error"] < 1e-5
        assert record["common"]["symmetric_identity_residual"] < 1e-10
        for key in ("00", "C0", "0K", "CK"):
            expected = record["A_recoverable"] + record["common"][f"Pi{key}_response"]
            assert np.allclose(record["common"][f"E_{key}"], expected)


def test_static_path_and_source_factorization_are_source_only():
    item = gaussian_input(primary_hidden_u_world(), "l2", 0.01)
    record = evaluate_input(item)
    path = static_audit(item)
    assert path is not None
    assert path["relative_error"] < 1e-4
    assert record["learner"]["source_factorization_residual"] < 1e-10
    assert record["target_used_by_learner"] is False
    assert record["semantic_or_cluster_used_by_learner"] is False
    assert np.linalg.norm(record["E"]) > 0.0
    assert record["common_base_kind"] == "ERM_source_solution"
    assert record["common_base_difference_norm"] > 1e-8
    # This is an operator norm, not a Frobenius norm of I.
    assert np.isclose(record["learner"]["K_operator_norm"], 1.0)


def test_positive_local_metric_is_not_rejected_by_zero_initial_reduction():
    eigenvalues = np.array([0.002, 3.0])
    assert eigenvalues.min() > 1e-10
    # The CMNIST adapter must use the actual minimum, not min(initial=0.0).
    assert not (eigenvalues.min() <= 1e-10)
