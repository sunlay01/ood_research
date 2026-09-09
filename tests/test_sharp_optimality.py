import numpy as np

from ood_repr_reg.corrected_geometry_snapshot import corrected_snapshot_audit
from ood_repr_reg.sharp_optimality.counterexamples import counterexamples
from ood_repr_reg.sharp_optimality.geometry import (
    affine_policy_audit,
    response_parts,
    transported_coordinate_audit,
)
from ood_repr_reg.round3r_3e_c_spectral import slack_ratio


def test_corrected_residual_uses_recoverable_response_at_full_observation():
    response = np.array([[2.0, 0.5], [-1.0, 3.0]])
    observation = np.eye(2)
    pieces = response_parts(response, observation, np.zeros_like(response))
    assert np.allclose(pieces["R"], 0.0)
    assert np.allclose(pieces["E"], response)
    assert not np.allclose(pieces["E"], pieces["PiO"])


def test_zero_observation_has_no_recoverable_residual():
    response = np.array([[2.0, 0.5], [-1.0, 3.0]])
    observation = np.zeros((3, 2))
    audit = corrected_snapshot_audit(response, observation, np.zeros_like(response))
    assert audit["passes"]
    assert np.allclose(audit["A_recoverable"], 0.0)
    assert np.allclose(audit["E"], 0.0)


def test_snapshot_reconstructs_response_and_full_observation_endpoint():
    rng = np.random.default_rng(8)
    response = rng.normal(size=(4, 3))
    observation = rng.normal(size=(5, 3))
    audit = corrected_snapshot_audit(response, observation, np.zeros_like(response))
    assert audit["passes"]
    assert audit["decomposition_residual"] < 1e-10
    assert audit["full_observation_recoverable_residual"] < 1e-10


def test_spectral_fixtures_separate_exactness_static_tax_and_placement():
    fixtures = counterexamples()
    assert fixtures["nonzero_E_within_slack"]["condition_holds"]
    assert fixtures["nonzero_E_within_slack"]["E_operator_norm"] > 0.0
    assert not fixtures["small_E_in_zero_slack_direction"]["condition_holds"]
    assert not fixtures["static_steering_breaks_optimality"]["full_minimax_condition"]
    assert fixtures["same_norm_different_placement_good"]["condition_holds"]
    assert not fixtures["same_norm_different_placement_bad"]["condition_holds"]


def test_metric_transport_and_cross_operator_identities():
    response = np.diag([2.0, 1.0])
    observation = np.array([[1.0, 0.0]])
    pi_o = np.array([[-1.5, 0.0], [0.0, 0.0]])
    audit = affine_policy_audit(np.zeros(2), response, observation, pi_o)
    assert audit["orthogonality"]["Airr_E_star_norm"] < 1e-10
    assert audit["orthogonality"]["E_Airr_star_norm"] < 1e-10
    coordinate = transported_coordinate_audit(
        np.zeros(2), response, observation, pi_o, np.eye(2), np.diag([0.7, 1.3])
    )
    assert coordinate["pass"]


def test_slack_support_is_not_reclassified_by_recoverable_scale():
    # S = diag(0, 1): the second response coordinate is its support.  The
    # support classification must not disappear when E is scaled up.
    irreducible = np.array([[1.0], [0.0]])
    small = slack_ratio(irreducible, np.array([[0.0], [0.1]]))
    large = slack_ratio(irreducible, np.array([[0.0], [1e8]]))
    assert small["support_compatible"]
    assert large["support_compatible"]
