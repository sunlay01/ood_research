import numpy as np

from ood_repr_reg.round3_algebra import (
    AlgebraTolerance,
    coordinate_nullspace,
    factorization_from_matrices,
    incremental_rank,
    inclusion_residual,
    intersection_dimension,
    model_lift,
    principal_angles,
    quotient_diagnostics,
    response_factorization,
    risk_response_pairing,
    pure_family_liftings,
    rank_with_tolerance,
    svec_symmetric,
    shift_lift,
    structural_reachability_liftings,
    subspace_basis,
)
from ood_repr_reg.round3_response_matrix import build_response_matrices
from ood_repr_reg.round3_shift_ensemble import base_environment, shift_ensemble
from ood_repr_reg.round3_trained_models import risk, train_models


def test_svec_preserves_frobenius_product():
    first = np.array([[1.0, 2.0], [2.0, -3.0]])
    second = np.array([[0.5, -1.0], [-1.0, 4.0]])
    assert np.isclose(svec_symmetric(first) @ svec_symmetric(second), np.sum(first * second))


def test_coordinate_nullspace_is_annihilated_and_has_expected_anchor():
    matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    result = coordinate_nullspace(matrix)
    assert result["free_columns"] == (2,)
    assert np.max(np.abs(result["residual"])) < 1e-12


def test_lifts_have_declared_dimensions_and_pairing_is_exact():
    base = base_environment()
    probe = shift_ensemble(base, count=1, seed=7)[0]
    weights = np.array([0.1, 1.0, -0.7, 0.2, -0.3])
    assert model_lift(weights).shape == (21,)
    assert shift_lift(probe.target, base).shape == (21,)
    direct = risk(weights, probe.target) - risk(weights, base)
    assert np.isclose(risk_response_pairing(weights, probe.target, base), direct)


def test_factorization_matches_raw_response_matrix():
    base = base_environment()
    sources = (base,)
    models = train_models(sources, seeds=(0, 1), lambdas=(0.1,))
    probes = shift_ensemble(base, count=3, seed=8)
    matrices = build_response_matrices(models, base, probes)
    result = factorization_from_matrices(models, base, probes, matrices)
    assert result["residual_max_absolute"] < 1e-9
    assert result["response_rank"] <= min(result["phi_rank"], result["psi_rank"])


def test_quotient_dimensions_are_rank_differences():
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    psi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = quotient_diagnostics(phi, psi)
    assert result["quotient_dimension"] == 2
    assert result["model_blind_dimension"] == 0
    assert result["shift_blind_dimension"] == 0


def test_subspace_geometry_and_incremental_rank():
    first = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    second = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    assert intersection_dimension(first, second) == 1
    assert np.allclose(principal_angles(first, first), 0.0)
    assert inclusion_residual(first[:, :1], first) < 1e-12
    assert incremental_rank(first[:, 1:2], second) == 1
    assert subspace_basis(first).shape == (3, 2)


def test_rank_and_geometry_do_not_require_target_risk_selection():
    base = base_environment()
    models = train_models((base,), seeds=(0,), lambdas=(0.1,))
    probes = shift_ensemble(base, count=2, seed=9)
    phi, psi, response = response_factorization(models, base, probes)
    assert phi.shape[0] == len(models)
    assert psi.shape[0] == len(probes)
    assert np.isfinite(response).all()


def test_pure_b_and_innovation_mean_families_are_the_same_observed_subspace():
    base = base_environment()
    families, _ = pure_family_liftings(base)
    assert rank_with_tolerance(families["b"]) == 4
    assert rank_with_tolerance(families["mu_xi"]) == 4
    assert np.allclose(families["b"], families["mu_xi"])


def test_structural_reachability_has_only_intercept_invariant_when_task_varies():
    base = base_environment()
    liftings = structural_reachability_liftings(base, count=80, seed=17, vary_task=True)
    assert rank_with_tolerance(liftings) == 20
