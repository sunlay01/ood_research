import pytest
import torch

from ood_repr_reg.decompositions import (
    log_loss_potential,
    log_loss_potential_gradient,
    proper_loss_decomposition,
    regularization_path_derivative,
    nested_oracle_decomposition,
    squared_risk_decomposition,
    transport_decomposition,
)


def test_squared_loss_pythagorean_decomposition_with_nonzero_noise() -> None:
    conditional_mean_x = torch.tensor([0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0])
    y = conditional_mean_x + torch.tensor([0.5, -0.5] * 4)
    conditional_mean_z = torch.tensor([1.0] * 4 + [2.0] * 4)
    prediction = torch.tensor([0.5] * 4 + [2.5] * 4)

    decomposition = squared_risk_decomposition(
        y, conditional_mean_x, conditional_mean_z, prediction
    )

    assert decomposition.noise == pytest.approx(torch.tensor(0.25))
    assert decomposition.representation_insufficiency == pytest.approx(torch.tensor(1.0))
    assert decomposition.head_mismatch == pytest.approx(torch.tensor(0.25))
    assert decomposition.risk == pytest.approx(decomposition.explained_sum)
    assert decomposition.residual == pytest.approx(torch.tensor(0.0), abs=1e-7)


def test_nested_oracle_decomposition_telescopes_and_exposes_signed_increments() -> None:
    decomposition = nested_oracle_decomposition(
        torch.tensor(0.20),
        torch.tensor(0.35),
        torch.tensor(0.25),
        torch.tensor(0.50),
    )

    assert decomposition.increments[0] == pytest.approx(torch.tensor(0.20))
    assert decomposition.test_inseparability == pytest.approx(torch.tensor(0.15))
    assert decomposition.training_test_misalignment == pytest.approx(torch.tensor(-0.10))
    assert decomposition.classifier_noninvariance == pytest.approx(torch.tensor(0.25))
    assert decomposition.final_error == pytest.approx(
        sum(decomposition.increments)
    )
    assert decomposition.residual == pytest.approx(torch.tensor(0.0), abs=1e-7)


def test_target_path_difference_splits_into_information_and_head_changes() -> None:
    conditional_mean_x = torch.tensor([0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0])
    y = conditional_mean_x + torch.tensor([0.5, -0.5] * 4)
    erm = squared_risk_decomposition(y, conditional_mean_x, conditional_mean_x, conditional_mean_x)
    regularized = squared_risk_decomposition(
        y,
        conditional_mean_x,
        torch.tensor([1.0] * 4 + [2.0] * 4),
        torch.tensor([0.5] * 4 + [2.5] * 4),
    )

    risk_change = regularized.risk - erm.risk
    component_change = (
        regularized.representation_insufficiency
        - erm.representation_insufficiency
        + regularized.head_mismatch
        - erm.head_mismatch
    )

    assert risk_change == pytest.approx(component_change)
    assert risk_change == pytest.approx(torch.tensor(1.25))


def test_log_loss_bregman_decomposition_matches_conditional_mutual_information() -> None:
    conditional_prob_x = torch.tensor(
        [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]]
    )
    conditional_prob_z = torch.tensor(
        [[0.8, 0.2], [0.8, 0.2], [0.3, 0.7], [0.3, 0.7]]
    )
    prediction = torch.tensor(
        [[0.75, 0.25], [0.75, 0.25], [0.35, 0.65], [0.35, 0.65]]
    )

    decomposition = proper_loss_decomposition(
        conditional_prob_x,
        conditional_prob_z,
        prediction,
        log_loss_potential,
        log_loss_potential_gradient,
    )
    conditional_mutual_information = (
        conditional_prob_x
        * (conditional_prob_x.log() - conditional_prob_z.log())
    ).sum(dim=-1).mean()

    assert decomposition.representation_insufficiency == pytest.approx(
        conditional_mutual_information
    )
    assert decomposition.regret == pytest.approx(
        decomposition.representation_insufficiency + decomposition.head_mismatch
    )
    assert decomposition.residual == pytest.approx(torch.tensor(0.0), abs=1e-7)


def test_transport_refinement_separates_shared_and_singular_terms() -> None:
    decomposition = transport_decomposition(
        source_weights=torch.tensor([0.6, 0.4]),
        density_ratio=torch.tensor([0.5, 1.5]),
        source_integrand=torch.tensor([1.0, 2.0]),
        target_integrand_on_source=torch.tensor([3.0, 4.0]),
        singular_weights=torch.tensor([0.1]),
        target_integrand_on_singular=torch.tensor([5.0]),
    )

    assert decomposition.singular_coverage == pytest.approx(torch.tensor(0.5))
    assert decomposition.difference == pytest.approx(
        decomposition.functional_shift
        + decomposition.density_reweighting
        + decomposition.singular_coverage
    )
    assert decomposition.residual == pytest.approx(torch.tensor(0.0), abs=1e-7)


def test_collapse_and_head_mismatch_are_distinct_failure_modes() -> None:
    y = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    collapse = squared_risk_decomposition(y, y, torch.zeros_like(y), torch.zeros_like(y))
    wrong_head = squared_risk_decomposition(y, y, y, torch.zeros_like(y))

    assert collapse.representation_insufficiency == pytest.approx(torch.tensor(1.0))
    assert collapse.head_mismatch == pytest.approx(torch.tensor(0.0))
    assert wrong_head.representation_insufficiency == pytest.approx(torch.tensor(0.0))
    assert wrong_head.head_mismatch == pytest.approx(torch.tensor(1.0))


def test_encoder_head_rescaling_preserves_predictions_but_not_l2_penalty() -> None:
    x = torch.tensor([[1.0, -2.0], [0.5, 3.0]])
    encoder = torch.tensor([[2.0, -1.0]])
    head = torch.tensor([3.0])
    scale = 5.0

    original_prediction = (x @ encoder.T) @ head
    rescaled_prediction = (x @ (scale * encoder).T) @ (head / scale)
    original_l2 = encoder.square().sum() + head.square().sum()
    rescaled_l2 = (scale * encoder).square().sum() + (head / scale).square().sum()

    assert original_prediction == pytest.approx(rescaled_prediction)
    assert original_l2 != pytest.approx(rescaled_l2)


def test_regularization_path_derivative_matches_implicit_quadratic_solution() -> None:
    hessian = torch.diag(torch.tensor([2.0, 4.0]))
    gradient_of_error = torch.tensor([3.0, -2.0])
    gradient_of_regularizer = torch.tensor([2.0, 8.0])

    derivative = regularization_path_derivative(
        gradient_of_error, hessian, gradient_of_regularizer
    )

    assert derivative == pytest.approx(torch.tensor(1.0))
