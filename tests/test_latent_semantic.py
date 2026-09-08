import numpy as np
import pytest
import torch

from ood_repr_reg.latent_semantic import (
    COMPONENTS,
    METHODS,
    LinearLatentModel,
    component_risk_accounting,
    conservative_component_bound,
    default_scm,
    fit_semantic_decomposition,
    gaussian_rbf_mmd,
    input_moments,
    projector_diagnostics,
    regularizer_value,
    sample_batches,
    source_environments,
    target_environments,
)


@pytest.fixture(scope="module")
def setup():
    scm = default_scm()
    sources = source_environments(scm)
    estimator = sample_batches(scm, sources, 512, 101)
    validation = sample_batches(scm, sources, 512, 202)
    model = LinearLatentModel(scm.input_dim, scm.input_dim, 0)
    decomposition = fit_semantic_decomposition(model, scm, estimator, validation)
    return scm, sources, model, decomposition, estimator


def test_factorial_source_design_covers_two_axes_and_reserves_third():
    scm = default_scm()
    sources = source_environments(scm)
    design = np.stack([environment.design for environment in sources])
    assert design.shape == (13, 6)
    assert np.linalg.matrix_rank(np.column_stack((np.ones(13), design))) == 7
    for environment in sources:
        assert environment.relation[2, 2] == pytest.approx(scm.relation_base[2, 2])


def test_population_moments_match_large_sample():
    scm = default_scm()
    environment = source_environments(scm)[3]
    mean, second, cross, y_second = input_moments(scm, environment)
    batch = sample_batches(scm, (environment,), 50_000, 7)[0]
    empirical_mean = batch.x.mean(0).numpy()
    empirical_second = (batch.x.T @ batch.x / len(batch.x)).numpy()
    empirical_cross = (batch.x.T @ batch.y / len(batch.x)).numpy()
    assert empirical_mean == pytest.approx(mean, abs=0.03)
    assert empirical_second == pytest.approx(second, abs=0.04)
    assert empirical_cross == pytest.approx(cross, abs=0.04)
    assert float((batch.y @ batch.y) / len(batch.y)) == pytest.approx(y_second, abs=0.04)


def test_five_projectors_are_orthogonal_and_complete(setup):
    _, _, _, decomposition, _ = setup
    assert set(decomposition.projectors) == set(COMPONENTS)
    diagnostics = projector_diagnostics(decomposition.projectors)
    assert max(diagnostics.values()) < 1e-8


def test_semantic_operators_recover_covered_generator_blocks(setup):
    _, _, _, decomposition, _ = setup
    assert decomposition.diagnostics["oracle_recovery_relation"] > 0.7
    assert decomposition.diagnostics["oracle_recovery_mean"] > 0.7
    assert decomposition.diagnostics["oracle_recovery_covariance"] > 0.7
    recovered = np.mean(
        [decomposition.diagnostics[f"oracle_recovery_{name}"] for name in COMPONENTS[:-1]]
    )
    permuted = np.mean(
        [decomposition.diagnostics[f"permuted_recovery_{name}"] for name in COMPONENTS[:-1]]
    )
    assert recovered > permuted


def test_risk_and_shapley_accounting_close_for_covered_and_unseen(setup):
    scm, sources, model, decomposition, _ = setup
    targets = target_environments(scm)
    assert any(target.covered for target in targets)
    assert any(not target.covered for target in targets)
    for target in targets:
        row = component_risk_accounting(model, scm, sources, target, decomposition)
        assert row["accounting_residual"] == pytest.approx(0.0, abs=1e-8)
        assert sum(row[f"shapley_{name}"] for name in COMPONENTS) == pytest.approx(
            row["target_transport"], abs=1e-8
        )


def test_all_regularizers_support_arbitrary_environment_count(setup):
    _, _, model, _, batches = setup
    for count in (3, 13):
        for method in METHODS:
            value = regularizer_value(method, model, batches[:count])
            assert value.ndim == 0
            assert torch.isfinite(value)
            assert float(value.detach()) >= -1e-8


def test_objective_formulas_match_direct_calculations(setup):
    _, _, model, _, batches = setup
    selected = batches[:3]
    parameters = torch.cat([parameter.flatten() for parameter in model.parameters()])
    assert float(regularizer_value("l1", model, selected).detach()) == pytest.approx(
        float(parameters.abs().mean().detach())
    )
    assert float(regularizer_value("l2", model, selected).detach()) == pytest.approx(
        float(parameters.square().mean().detach())
    )

    encoded = [model.encode(batch.x) for batch in selected]
    covariances = []
    radial, gradients, hessians = [], [], []
    head = model.head.weight.squeeze(0)
    bias = model.head.bias.squeeze(0)
    for z, batch in zip(encoded, selected, strict=True):
        centered = z - z.mean(0, keepdim=True)
        covariances.append(centered.T @ centered / (len(z) - 1))
        prediction = z @ head + bias
        residual = prediction - batch.y
        augmented = torch.cat((z, torch.ones((len(z), 1), dtype=z.dtype)), dim=1)
        radial.append(2.0 * (residual * prediction).mean())
        gradients.append(2.0 * augmented.T @ residual / len(z))
        hessians.append(2.0 * augmented.T @ augmented / len(z))
    direct_irm = torch.stack(radial).square().mean()
    direct_coral = torch.stack(
        [
            (covariances[i] - covariances[j]).square().mean()
            for i, j in ((0, 1), (0, 2), (1, 2))
        ]
    ).mean()
    stacked_gradients = torch.stack(gradients)
    stacked_hessians = torch.stack(hessians)
    assert float(regularizer_value("irmv1", model, selected).detach()) == pytest.approx(
        float(direct_irm.detach())
    )
    assert float(regularizer_value("coral", model, selected).detach()) == pytest.approx(
        float(direct_coral.detach())
    )
    assert float(regularizer_value("grad_align", model, selected).detach()) == pytest.approx(
        float(
            (stacked_gradients - stacked_gradients.mean(0, keepdim=True))
            .square()
            .mean()
            .detach()
        )
    )
    assert float(regularizer_value("hess_align", model, selected).detach()) == pytest.approx(
        float(
            (stacked_hessians - stacked_hessians.mean(0, keepdim=True))
            .square()
            .mean()
            .detach()
        )
    )
    direct_mmd = torch.stack(
            [
                gaussian_rbf_mmd(encoded[i], encoded[j])
                for i, j in ((0, 1), (0, 2), (1, 2))
            ]
        ).mean()
    assert float(regularizer_value("mmd", model, selected).detach()) == pytest.approx(
        float(direct_mmd.detach())
    )


def test_gaussian_mmd_is_symmetric_and_zero_on_same_distribution():
    generator = torch.Generator().manual_seed(3)
    left = torch.randn((128, 4), generator=generator, dtype=torch.float64)
    right = 0.5 + 1.2 * torch.randn((128, 4), generator=generator, dtype=torch.float64)
    assert gaussian_rbf_mmd(left, left) == pytest.approx(torch.tensor(0.0), abs=1e-10)
    assert gaussian_rbf_mmd(left, right) == pytest.approx(gaussian_rbf_mmd(right, left))


def test_latent_orthogonal_reparameterization_preserves_accounting(setup):
    scm, sources, model, decomposition, estimator = setup
    target = target_environments(scm)[0]
    original = component_risk_accounting(model, scm, sources, target, decomposition)
    generator = torch.Generator().manual_seed(44)
    rotation, _ = torch.linalg.qr(
        torch.randn((scm.input_dim, scm.input_dim), generator=generator, dtype=torch.float64)
    )
    rotated = LinearLatentModel(scm.input_dim, scm.input_dim, 99)
    with torch.no_grad():
        rotated.encoder.weight.copy_(rotation @ model.encoder.weight)
        rotated.head.weight.copy_(model.head.weight @ rotation.T)
        rotated.head.bias.copy_(model.head.bias)
    validation = sample_batches(scm, sources, 512, 303)
    rotated_decomposition = fit_semantic_decomposition(rotated, scm, estimator, validation)
    transformed = component_risk_accounting(
        rotated, scm, sources, target, rotated_decomposition
    )
    assert transformed["target_transport"] == pytest.approx(original["target_transport"], abs=1e-8)
    for component in COMPONENTS:
        assert transformed[f"head_energy_{component}"] == pytest.approx(
            original[f"head_energy_{component}"], rel=0.03, abs=1e-5
        )


def test_projectors_do_not_require_target_argument():
    parameters = fit_semantic_decomposition.__annotations__
    assert "target" not in parameters
    assert "target_batches" not in parameters


def test_fold_exchange_is_aggregated(setup):
    scm, sources, model, decomposition, estimator = setup
    validation = sample_batches(scm, sources, 512, 202)
    reversed_decomposition = fit_semantic_decomposition(
        model, scm, validation, estimator, permutation_seed=0
    )
    for component in COMPONENTS:
        assert decomposition.diagnostics[f"crossfit_overlap_{component}"] == pytest.approx(
            reversed_decomposition.diagnostics[f"crossfit_overlap_{component}"], abs=1e-10
        )
        assert decomposition.diagnostics[f"oracle_recovery_{component}"] == pytest.approx(
            reversed_decomposition.diagnostics[f"oracle_recovery_{component}"], abs=1e-10
        )


def test_bound_is_target_independent_and_covers_frozen_targets(setup):
    scm, sources, model, decomposition, _ = setup
    first = conservative_component_bound(model, scm, sources, decomposition)
    second = conservative_component_bound(model, scm, sources, decomposition)
    assert first == second
    for target in target_environments(scm):
        transport = component_risk_accounting(model, scm, sources, target, decomposition)[
            "target_transport"
        ]
        assert abs(transport) <= first["component_bound"] + 1e-10
