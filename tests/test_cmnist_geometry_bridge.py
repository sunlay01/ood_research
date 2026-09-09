import numpy as np

from ood_repr_reg.cmnist_geometry_bridge import (
    CMNISTFamily,
    RepresentationBank,
    build_geometry,
    head_from_state,
    moment_state,
    source_observation,
    source_state_stack,
    task_state,
    response_decomposition,
    method_row,
)


def bank(seed=4, n=120):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(float)
    red = rng.normal(size=(n, 4)) + labels[:, None] * np.array([1.0, 0.2, 0.0, 0.0])
    green = rng.normal(size=(n, 4)) + labels[:, None] * np.array([0.5, 0.0, 0.2, 0.0])
    return RepresentationBank(red, green, labels)


def test_task_state_and_moments_are_finite():
    b = bank()
    m, c, y2 = moment_state(b, 0.8)
    state = task_state(m, c, y2)
    assert m.shape == (5, 5)
    assert state.ndim == 1 and np.isfinite(state).all()


def test_family_is_explicit_and_source_hidden_direction_is_legal():
    f = CMNISTFamily("mechanism_defined_hidden")
    assert f.dimension == 3
    assert f.source_rhos_at(np.array([0.0, 0.0, 0.01])) == f.source_rhos
    assert f.target_rho_at(np.array([0.0, 0.0, 0.01])) != f.target_rho
    assert f.metadata()["world_semantics"] == "declared_source_target_coupling"


def test_central_difference_rejects_correlation_boundary_crossing():
    f = CMNISTFamily("mechanism_defined_hidden", target_rho=0.9999)
    assert f.legal_step_radius(np.zeros(f.dimension), "rho_hidden") < 1e-3
    with np.testing.assert_raises(ValueError):
        f.validate_central_step(np.zeros(f.dimension), 2, 1e-3)


def test_source_observation_does_not_use_target_response():
    b = bank()
    f = CMNISTFamily("mechanism_defined_hidden")
    o = source_observation(b, f)
    assert o.shape[1] == 3
    assert np.linalg.norm(o[:, 2]) < 1e-8


def test_exposed_direction_changes_source_observation():
    b = bank()
    f = CMNISTFamily("mechanism_defined_exposed", hidden_exposed=True)
    o = source_observation(b, f)
    assert np.linalg.norm(o[:, 2]) > 1e-6


def test_head_optimum_is_stationary_for_erm():
    b = bank()
    f = CMNISTFamily("source_induced", directions=("rho_source_1", "rho_source_2"))
    state = source_state_stack(b, f, np.zeros(2))
    w = head_from_state(state, b.dimension, method="erm")
    width = b.dimension * (b.dimension + 1) // 2 + b.dimension + 1
    m = []; c = []
    for start in range(0, len(state), width):
        block = state[start:start + width]
        nmat = b.dimension * (b.dimension + 1) // 2
        from ood_repr_reg.cmnist_geometry_bridge import decode_svec
        m.append(decode_svec(block, b.dimension)); c.append(block[nmat:nmat+b.dimension])
    assert np.linalg.norm(np.mean([2*(mm @ w-cc) for mm,cc in zip(m,c)], axis=0)) < 1e-8


def test_geometry_has_expected_hidden_kernel_and_exposure_reduces_it():
    b = bank()
    hidden = build_geometry(b, CMNISTFamily("mechanism_defined_hidden"))
    exposed = build_geometry(b, CMNISTFamily("mechanism_defined_exposed", hidden_exposed=True))
    assert hidden["kernel_dimension"] >= exposed["kernel_dimension"]
    assert hidden["alpha"] >= exposed["alpha"] - 1e-8
    assert hidden["information_floor"] >= 0 and exposed["information_floor"] >= 0


def test_source_induced_family_is_lower_dimensional():
    b = bank()
    f = CMNISTFamily("source_induced", directions=("rho_source_1", "rho_source_2"))
    assert source_state_stack(b, f, np.zeros(2)).size > 0
    assert source_observation(b, f).shape[1] == 2


def test_irrelevant_source_direction_preserves_hidden_ambiguity():
    b = bank()
    hidden = build_geometry(b, CMNISTFamily("mechanism_defined_hidden"))
    control = build_geometry(b, CMNISTFamily(
        "irrelevant_source_diversity",
        directions=("rho_source_1", "rho_source_2", "rho_hidden", "brightness_nuisance"),
    ))
    assert control["rank_O"] >= hidden["rank_O"]
    assert abs(control["alpha"] - hidden["alpha"]) < 1e-8


def test_geometry_step_audit_is_stable():
    result = build_geometry(bank(), CMNISTFamily("mechanism_defined_hidden"))
    assert max(result["A_step_relative_errors"]) < 1e-3
    assert max(result["O_step_relative_errors"]) < 1e-6


def test_ift_audit_does_not_mutate_torch_default_dtype():
    import torch
    from ood_repr_reg.cmnist_geometry_bridge import ift_matrices
    b = bank()
    f = CMNISTFamily("source_induced", directions=("rho_source_1", "rho_source_2"))
    state = source_state_stack(b, f, np.zeros(2))
    w = head_from_state(state, b.dimension)
    before = torch.get_default_dtype()
    ift_matrices(state, b.dimension, "l2", 0.001, w)
    assert torch.get_default_dtype() is before


def test_response_decomposition_full_observability_uses_recoverable_response():
    a = np.array([[1.0, 2.0], [-0.5, 3.0]])
    o = np.eye(2)
    irr, rec, p = response_decomposition(a, o)
    assert np.allclose(p, 0.0)
    assert np.allclose(irr, 0.0)
    assert np.allclose(rec, a)
    assert np.linalg.norm(a - irr - rec) < 1e-12


def test_response_decomposition_no_observation_has_zero_recoverable_part():
    a = np.array([[1.0, 2.0], [-0.5, 3.0]])
    o = np.zeros((3, 2))
    irr, rec, p = response_decomposition(a, o)
    assert np.allclose(p, np.eye(2))
    assert np.allclose(irr, a)
    assert np.allclose(rec, 0.0)
    assert np.linalg.norm(a - irr - rec) < 1e-12
    assert np.allclose(rec + np.zeros_like(a), 0.0)


def test_response_decomposition_always_reconstructs_response():
    rng = np.random.default_rng(17)
    a = rng.normal(size=(4, 5))
    o = rng.normal(size=(7, 5))
    irr, rec, _ = response_decomposition(a, o)
    assert np.linalg.norm(a - irr - rec) < 1e-10


def test_source_induced_method_residual_includes_recoverable_response():
    b = bank()
    f = CMNISTFamily("source_induced", directions=("rho_source_1", "rho_source_2"))
    o = source_observation(b, f)
    a = build_geometry(b, f)
    irr, rec, _ = response_decomposition(
        np.asarray(a["A"]), o
    )
    assert a["kernel_dimension"] == 0
    assert np.allclose(irr, 0.0)
    assert np.allclose(rec, np.asarray(a["A"]))


def test_source_induced_method_row_does_not_report_pi_o_as_e():
    row = method_row(
        bank(),
        CMNISTFamily("source_induced", directions=("rho_source_1", "rho_source_2")),
        "erm", 0.0, "source_induced", (1e-3, 5e-4),
    )
    assert row["decomposition_residual"] < 1e-10
    assert abs(row["E_operator_norm"] - row["pi_O_operator_norm"]) > 1e-6
