"""Generic source-only response correction.

This module intentionally contains no dataset, channel, color, or target logic.
It only consumes source risks, a source-statistic penalty, and model gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ResponseCorrection:
    direct_gradient: Tensor
    direct_update: Tensor
    cross_gradient: Tensor
    cross_update: Tensor
    correction: Tensor
    shield: Tensor


def response_balanced_correction(
    *,
    basis: Tensor,
    cross_gradient: Tensor,
    response_basis: Tensor,
    risk_gradient: Tensor,
    penalty_gradient: Tensor,
    ridge: float,
    budget_ratio: float,
    descent_tolerance: float,
) -> Tensor:
    """Solve a source-only response cancellation problem in the contrast span.

    ``basis`` contains source environment-gradient contrast directions and
    ``response_basis[:, j]`` is the source-risk HVP applied to basis column
    ``j``. No semantic feature labels are used.
    """
    if basis.shape[1] == 0:
        return torch.zeros_like(penalty_gradient)
    gram = basis.T @ response_basis
    rhs = basis.T @ cross_gradient
    system = gram + ridge * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    coefficients = torch.linalg.solve(system, rhs)
    correction = -(basis @ coefficients)
    budget = budget_ratio * risk_gradient.norm()
    correction = cap_norm(correction, budget)
    candidate = penalty_gradient + correction
    risk_norm = risk_gradient.norm().clamp_min(1e-12)
    if torch.dot(risk_gradient, candidate) < -descent_tolerance * risk_norm * candidate.norm():
        correction = correction - torch.dot(risk_gradient, candidate) / risk_norm.square() * risk_gradient
        correction = cap_norm(correction, budget)
    return correction.detach()


def radial_rate_balance_correction(
    *,
    parameters: Tensor,
    basis: Tensor,
    risk_update: Tensor,
    penalty_update: Tensor,
    radial_margin: float,
    coordinate_eps: float,
    budget_ratio: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Normalize source contrast-mode radial rates without semantic labels.

    For each source environment-contrast mode ``u_j``, the observable
    coordinate is ``a_j = <u_j, theta>`` and its relative update is
    ``a_j <u_j, delta> / (a_j^2 + eps)``. The penalty radial update is clipped
    to the source-risk radial magnitude plus a fixed margin. Tangential
    components are preserved. The returned values are correction, risk rates,
    and penalty rates.
    """
    if basis.shape[1] == 0:
        empty = torch.zeros_like(parameters)
        return empty, empty, empty
    coordinates = basis.T @ parameters
    risk_coordinates = basis.T @ risk_update
    penalty_coordinates = basis.T @ penalty_update
    denominator = coordinates.square() + coordinate_eps
    risk_rates = coordinates * risk_coordinates / denominator
    penalty_rates = coordinates * penalty_coordinates / denominator
    allowed = risk_rates.abs() + radial_margin
    scales = torch.minimum(torch.ones_like(penalty_rates), allowed / penalty_rates.abs().clamp_min(1e-12))
    balanced_coordinates = penalty_coordinates * scales
    correction = basis @ (balanced_coordinates - penalty_coordinates)
    correction = cap_norm(correction, budget_ratio * risk_update.norm())
    return correction.detach(), risk_rates.detach(), penalty_rates.detach()


def response_radial_rate_balance_correction(
    *,
    basis: Tensor,
    response_coordinate: Tensor,
    risk_response: Tensor,
    penalty_response: Tensor,
    learning_rate: float,
    radial_margin: float,
    coordinate_eps: float,
    budget_ratio: float,
    risk_update_norm: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Balance radial rates of source-risk response coordinates.

    The coordinate is a source-observable contrast response, not a semantic
    invariant/shortcut coordinate. HVPs provide its source-risk and penalty
    velocities, so the rule follows the radial-rate theorem without feature
    labels.
    """
    if basis.shape[1] == 0:
        empty = torch.zeros_like(risk_response)
        return empty, empty, empty
    denominator = response_coordinate.square() + coordinate_eps
    risk_rates = response_coordinate * risk_response / denominator
    penalty_rates = response_coordinate * penalty_response / denominator
    allowed = risk_rates.abs() + radial_margin
    scales = torch.minimum(torch.ones_like(penalty_rates), allowed / penalty_rates.abs().clamp_min(1e-12))
    balanced_response = penalty_response * scales
    correction = learning_rate * basis @ (balanced_response - penalty_response)
    correction = cap_norm(correction, budget_ratio * risk_update_norm)
    return correction.detach(), risk_rates.detach(), penalty_rates.detach()


def project(vector: Tensor, basis: Tensor) -> Tensor:
    if basis.shape[1] == 0:
        return torch.zeros_like(vector)
    return basis @ (basis.T @ vector)


def clip_actual_contrast_transfer(
    *,
    actual_update: Tensor,
    risk_update: Tensor,
    basis: Tensor,
    budget_ratio: float,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Limit observed source-contrast transfer after the optimizer step.

    The residual is measured after the optimizer preconditioner, so the rule
    controls the update that actually reaches parameters. ``basis`` is built
    from source-environment contrasts and carries no semantic feature labels.
    """
    if basis.shape[1] == 0:
        empty = torch.zeros_like(actual_update)
        return empty, empty, empty
    actual_contrast = basis @ (basis.T @ actual_update.detach())
    risk_contrast = basis @ (basis.T @ risk_update.detach())
    residual = actual_contrast - risk_contrast
    budget = budget_ratio * risk_contrast.norm().clamp_min(eps)
    residual_norm = residual.norm()
    clipped_residual = residual * torch.clamp(budget / residual_norm.clamp_min(eps), max=1.0)
    correction = clipped_residual - residual
    return correction.detach(), residual.detach(), clipped_residual.detach()


def clip_curvature_aligned_excess_transfer(
    *,
    actual_update: Tensor,
    risk_update: Tensor,
    curvature_direction: Tensor,
    basis: Tensor,
    budget_ratio: float,
    budget_update: Tensor | None = None,
    correction_budget_ratio: float | None = None,
    correction_gain: float = 1.0,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Clip only transfer aligned with the source curvature response.

    The correction is source-only: the gate uses the inner product between
    the observed contrast residual and ``curvature_direction``. Components
    orthogonal to that response are preserved.
    """
    if basis.shape[1] == 0:
        empty = torch.zeros_like(actual_update)
        return empty, empty, empty, torch.zeros(())
    actual_contrast = basis @ (basis.T @ actual_update.detach())
    risk_contrast = basis @ (basis.T @ risk_update.detach())
    residual = actual_contrast - risk_contrast
    direction = basis @ (basis.T @ curvature_direction.detach())
    direction_norm_sq = direction.square().sum().clamp_min(eps)
    signed_response = torch.dot(residual, direction)
    gate = (signed_response > 0).to(residual.dtype)
    aligned = gate * signed_response / direction_norm_sq * direction
    budget_reference = risk_contrast if budget_update is None else budget_update.detach()
    budget = budget_ratio * budget_reference.norm().clamp_min(eps)
    aligned_norm = aligned.norm()
    clipped_aligned = aligned * torch.clamp(budget / aligned_norm.clamp_min(eps), max=1.0)
    correction = correction_gain * (clipped_aligned - aligned)
    if correction_budget_ratio is not None:
        correction = cap_norm(correction, correction_budget_ratio * budget_reference.norm().clamp_min(eps))
    return correction.detach(), residual.detach(), clipped_aligned.detach(), signed_response.detach()


def clip_common_statistic_radial_update(
    *,
    actual_update: Tensor,
    risk_update: Tensor,
    statistic_gradient: Tensor,
    budget_ratio: float,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Limit excess radial response of a source-observable common statistic."""
    gradient_norm_sq = statistic_gradient.detach().square().sum().clamp_min(eps)
    actual_response = torch.dot(statistic_gradient.detach(), actual_update.detach())
    risk_response = torch.dot(statistic_gradient.detach(), risk_update.detach())
    residual_response = actual_response - risk_response
    budget = budget_ratio * risk_response.detach().abs().clamp_min(eps)
    clipped_response = residual_response.clamp(min=-budget, max=budget)
    correction = (clipped_response - residual_response) * statistic_gradient.detach() / gradient_norm_sq
    return correction.detach(), residual_response.detach(), clipped_response.detach()


def cap_norm(vector: Tensor, maximum: Tensor) -> Tensor:
    norm = vector.norm()
    return vector * torch.clamp(maximum / norm.clamp_min(1e-12), max=1.0)


def make_correction(
    *,
    direct_gradient: Tensor,
    penalty_update: Tensor,
    cross_gradient: Tensor,
    learning_rate: float,
    risk_update_norm: Tensor,
    curvature_kappa: float,
    direct_budget_ratio: float,
    max_correction_ratio: float,
    max_shield_ratio: float,
    response_shield: float,
) -> ResponseCorrection:
    direct_update = penalty_update
    cross_update = learning_rate * cross_gradient
    correction = -curvature_kappa * cross_update
    correction = cap_norm(correction, max_correction_ratio * risk_update_norm)
    direct_excess = torch.clamp(
        direct_update.norm() - direct_budget_ratio * risk_update_norm,
        min=0.0,
    )
    shield = -response_shield * direct_excess * direct_update / direct_update.norm().clamp_min(1e-12)
    shield = cap_norm(shield, max_shield_ratio * risk_update_norm)
    return ResponseCorrection(
        direct_gradient=direct_gradient,
        direct_update=direct_update,
        cross_gradient=cross_gradient,
        cross_update=cross_update,
        correction=correction,
        shield=shield,
    )
