"""Correct frozen-head A geometry from smooth expectation derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .smooth_world import SmoothPool, SmoothWorlds, environment_parameters, outcome_weight


HEAD_DIMENSION = 65
SOURCE_HESSIAN_DAMPING_RELATIVE = 1e-6


@dataclass(frozen=True)
class Whitening:
    root: Tensor
    inverse_root: Tensor
    damping: float
    condition_number: float
    identity_error: float


@dataclass(frozen=True)
class CorrectedGeometry:
    A: Tensor
    target_gradient_derivative: Tensor
    source_gradient_derivative: Tensor
    delta_gradient_derivative: Tensor
    whitening: Whitening


def augmented_head(model: torch.nn.Module) -> Tensor:
    return torch.cat((model.head.weight.detach().cpu().double().reshape(-1), model.head.bias.detach().cpu().double().reshape(-1)))


def logits(features: Tensor, weights: Tensor) -> Tensor:
    return features.double() @ weights[:-1, None] + weights[-1]


@torch.no_grad()
def encoded_outcomes(model: torch.nn.Module, pool: SmoothPool, *, batch_size: int = 4096) -> tuple[tuple[Tensor, Tensor, int, int], ...]:
    model.eval()
    collected: list[list[Tensor]] = [[], [], [], []]
    labels: list[Tensor | None] = [None, None, None, None]
    outcomes = pool.outcome_batches()
    for start in range(0, pool.images.shape[0], batch_size):
        stop = min(start + batch_size, pool.images.shape[0])
        for index, (images, label, _, _) in enumerate(outcomes):
            collected[index].append(model.encode(images[start:stop]).detach().cpu().double())
            labels[index] = label.detach().cpu().double()
    return tuple((torch.cat(collected[index]), labels[index], outcomes[index][2], outcomes[index][3]) for index in range(4))  # type: ignore[arg-type,return-value]


def expected_head_risk(outcomes: tuple[tuple[Tensor, Tensor, int, int], ...], p: Tensor, q: Tensor, weights: Tensor) -> Tensor:
    total = weights.new_zeros(())
    for features, labels, label_flip, color_flip in outcomes:
        loss = F.binary_cross_entropy_with_logits(logits(features, weights), labels.double())
        total = total + outcome_weight(p, q, label_flip, color_flip) * loss
    return total


def _environment_gradient(outcomes: tuple[tuple[Tensor, Tensor, int, int], ...], p: Tensor, q: Tensor, weights: Tensor) -> Tensor:
    risk = expected_head_risk(outcomes, p, q, weights)
    return torch.autograd.grad(risk, weights, create_graph=True)[0]


def _average_gradient(outcomes: tuple[tuple[tuple[Tensor, Tensor, int, int], ...], tuple[tuple[Tensor, Tensor, int, int], ...]], theta: Tensor, weights: Tensor, *, evaluation: bool) -> Tensor:
    parts = []
    for environment, item in enumerate(outcomes):
        p, q = environment_parameters(theta, environment=environment, evaluation=evaluation)
        parts.append(_environment_gradient(item, p, q, weights))
    return torch.stack(parts).mean(dim=0)


def source_risk_hessian(source_outcomes, theta: Tensor, weights: Tensor) -> Tensor:
    point = weights.detach().cpu().double().clone().requires_grad_(True)
    return torch.autograd.functional.hessian(lambda value: _average_risk(source_outcomes, theta, value, evaluation=False), point).detach().cpu().double()


def _average_risk(outcomes, theta: Tensor, weights: Tensor, *, evaluation: bool) -> Tensor:
    values = []
    for environment, item in enumerate(outcomes):
        p, q = environment_parameters(theta, environment=environment, evaluation=evaluation)
        values.append(expected_head_risk(item, p, q, weights))
    return torch.stack(values).mean()


def whiten(hessian: Tensor) -> Whitening:
    symmetric = (hessian + hessian.T).double() / 2.0
    values, vectors = torch.linalg.eigh(symmetric)
    damping = SOURCE_HESSIAN_DAMPING_RELATIVE * max(float(torch.trace(symmetric).item() / HEAD_DIMENSION), 1e-8)
    effective = values + damping
    if bool((effective <= 0).any()):
        raise ValueError("source risk Hessian is not positive after fixed damping")
    root = vectors @ torch.diag(effective.sqrt()) @ vectors.T
    inverse_root = vectors @ torch.diag(effective.rsqrt()) @ vectors.T
    identity_error = float((inverse_root @ (root @ root) @ inverse_root - torch.eye(HEAD_DIMENSION, dtype=torch.double)).norm().item() / HEAD_DIMENSION)
    return Whitening(root=root, inverse_root=inverse_root, damping=damping, condition_number=float((effective.max() / effective.min()).item()), identity_error=identity_error)


def corrected_geometry(model: torch.nn.Module, worlds: SmoothWorlds) -> CorrectedGeometry:
    source = tuple(encoded_outcomes(model, pool) for pool in worlds.source)
    evaluation = tuple(encoded_outcomes(model, pool) for pool in worlds.evaluation)
    weights = augmented_head(model).requires_grad_(True)
    theta = worlds.base_theta.detach().clone().requires_grad_(True)
    target_jacobian = torch.autograd.functional.jacobian(lambda value: _average_gradient(evaluation, value, weights, evaluation=True), theta)
    source_jacobian = torch.autograd.functional.jacobian(lambda value: _average_gradient(source, value, weights, evaluation=False), theta)
    whitening = whiten(source_risk_hessian(source, theta, weights))
    delta = target_jacobian - source_jacobian
    return CorrectedGeometry(A=whitening.inverse_root @ delta, target_gradient_derivative=target_jacobian.detach(), source_gradient_derivative=source_jacobian.detach(), delta_gradient_derivative=delta.detach(), whitening=whitening)


def directional_derivative(columns: Tensor, direction: Tensor) -> Tensor:
    return columns @ direction.detach().cpu().double()


def independent_A_direction(model: torch.nn.Module, worlds: SmoothWorlds, direction: Tensor) -> Tensor:
    """JVP path used only to test linearity against the three stored columns."""
    source = tuple(encoded_outcomes(model, pool) for pool in worlds.source)
    evaluation = tuple(encoded_outcomes(model, pool) for pool in worlds.evaluation)
    weights = augmented_head(model).requires_grad_(True)
    theta = worlds.base_theta.detach().clone().requires_grad_(True)
    _, target = torch.autograd.functional.jvp(lambda value: _average_gradient(evaluation, value, weights, evaluation=True), (theta,), (direction,))
    _, source_direction = torch.autograd.functional.jvp(lambda value: _average_gradient(source, value, weights, evaluation=False), (theta,), (direction,))
    whitening = whiten(source_risk_hessian(source, theta, weights))
    return (whitening.inverse_root @ (target - source_direction)).detach().cpu().double()


def smooth_fd_consistency(model: torch.nn.Module, worlds: SmoothWorlds, *, epsilon: float = 1e-5) -> float:
    """Maximum relative error between autograd and central FD of smooth gradients."""
    source = tuple(encoded_outcomes(model, pool) for pool in worlds.source)
    evaluation = tuple(encoded_outcomes(model, pool) for pool in worlds.evaluation)
    weights = augmented_head(model).requires_grad_(True)
    theta = worlds.base_theta.detach().clone().requires_grad_(True)
    analytic = torch.autograd.functional.jacobian(
        lambda value: _average_gradient(evaluation, value, weights, evaluation=True) - _average_gradient(source, value, weights, evaluation=False),
        theta,
    )
    errors = []
    for index in range(3):
        direction = torch.zeros(3, dtype=torch.double); direction[index] = 1.0
        plus = _average_gradient(evaluation, theta + epsilon * direction, weights, evaluation=True) - _average_gradient(source, theta + epsilon * direction, weights, evaluation=False)
        minus = _average_gradient(evaluation, theta - epsilon * direction, weights, evaluation=True) - _average_gradient(source, theta - epsilon * direction, weights, evaluation=False)
        finite = (plus - minus) / (2.0 * epsilon)
        errors.append(float((finite - analytic[:, index]).norm().detach() / max(float(analytic[:, index].norm().detach()), 1e-12)))
    return max(errors)
