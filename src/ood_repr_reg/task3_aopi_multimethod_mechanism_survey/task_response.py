"""Frozen-encoder task response geometry in the declared five-dimensional world."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .smooth_world5 import SmoothPool, SmoothWorld5, environment_parameters, outcome_weight, subset_pool


HEAD_DIMENSION = 65


@dataclass(frozen=True)
class Whitening:
    root: Tensor
    inverse_root: Tensor
    damping: float
    condition_number: float
    identity_error: float


@dataclass(frozen=True)
class TaskGeometry:
    A: Tensor
    source_jacobian: Tensor
    evaluation_jacobian: Tensor
    delta_jacobian: Tensor
    whitening: Whitening


def augmented_head(model: nn.Module) -> Tensor:
    return torch.cat((model.head.weight.detach().double().reshape(-1), model.head.bias.detach().double().reshape(-1)))


def _logits(features: Tensor, weights: Tensor) -> Tensor:
    return features.double() @ weights[:-1, None] + weights[-1]


@torch.no_grad()
def encoded_outcomes(model: nn.Module, pool: SmoothPool, *, batch_size: int = 1024) -> tuple[tuple[Tensor, Tensor, int, int], ...]:
    model.eval()
    collected: list[list[Tensor]] = [[], [], [], []]
    outcomes = pool.outcome_batches()
    for start in range(0, pool.images.shape[0], batch_size):
        stop = min(start + batch_size, pool.images.shape[0])
        for index, (images, labels, label_flip, color_flip) in enumerate(outcomes):
            collected[index].append(model.encode(images[start:stop]).detach().cpu().double())
    return tuple(
        (torch.cat(collected[index]), outcomes[index][1].detach().cpu().double(), outcomes[index][2], outcomes[index][3])
        for index in range(4)
    )


def expected_head_risk(outcomes: tuple[tuple[Tensor, Tensor, int, int], ...], p: Tensor, q: Tensor, weights: Tensor) -> Tensor:
    result = weights.new_zeros(())
    for features, labels, label_flip, color_flip in outcomes:
        loss = F.binary_cross_entropy_with_logits(_logits(features, weights), labels)
        result = result + outcome_weight(p, q, label_flip, color_flip) * loss
    return result


def _average_risk(outcomes: tuple[tuple[tuple[Tensor, Tensor, int, int], ...], ...], delta: Tensor, weights: Tensor, *, evaluation: bool) -> Tensor:
    risks = []
    for environment, item in enumerate(outcomes):
        p, q = environment_parameters(delta, environment=environment, evaluation=evaluation)
        risks.append(expected_head_risk(item, p, q, weights))
    return torch.stack(risks).mean()


def _average_gradient(outcomes: tuple[tuple[tuple[Tensor, Tensor, int, int], ...], ...], delta: Tensor, weights: Tensor, *, evaluation: bool) -> Tensor:
    return torch.autograd.grad(_average_risk(outcomes, delta, weights, evaluation=evaluation), weights, create_graph=True)[0]


def environment_gradient(outcomes, delta: Tensor, weights: Tensor, *, environment: int) -> Tensor:
    p, q = environment_parameters(delta, environment=environment, evaluation=False)
    return torch.autograd.grad(expected_head_risk(outcomes, p, q, weights), weights, create_graph=True)[0]


def source_hessian(source_outcomes, delta: Tensor, weights: Tensor) -> Tensor:
    point = weights.detach().clone().requires_grad_(True)
    value = torch.autograd.functional.hessian(lambda w: _average_risk(source_outcomes, delta, w, evaluation=False), point)
    return ((value + value.T) / 2.0).detach()


def whiten(hessian: Tensor, relative_damping: float = 1e-6) -> Whitening:
    symmetric = ((hessian.double() + hessian.double().T) / 2.0).detach()
    eigenvalues, vectors = torch.linalg.eigh(symmetric)
    damping = float(relative_damping) * max(float(torch.trace(symmetric).item()) / HEAD_DIMENSION, 1e-8)
    effective = eigenvalues + damping
    if bool((effective <= 0).any()):
        raise ValueError("damped source Hessian is not positive definite")
    root = vectors @ torch.diag(effective.sqrt()) @ vectors.T
    inverse_root = vectors @ torch.diag(effective.rsqrt()) @ vectors.T
    identity = inverse_root @ (root @ root) @ inverse_root
    error = float((identity - torch.eye(HEAD_DIMENSION, dtype=torch.double)).norm().item() / HEAD_DIMENSION)
    return Whitening(root, inverse_root, damping, float((effective.max() / effective.min()).item()), error)


def task_geometry(model: nn.Module, worlds: SmoothWorld5, *, relative_damping: float = 1e-6, pool_size: int = 1024) -> TaskGeometry:
    source_pools = tuple(subset_pool(pool, pool_size) for pool in worlds.source)
    eval_pools = tuple(subset_pool(pool, pool_size) for pool in worlds.evaluation)
    source = tuple(encoded_outcomes(model, pool) for pool in source_pools)
    evaluation = tuple(encoded_outcomes(model, pool) for pool in eval_pools)
    weights = augmented_head(model).requires_grad_(True)
    delta = worlds.base_delta.detach().clone().requires_grad_(True)
    source_jacobian = torch.autograd.functional.jacobian(lambda value: _average_gradient(source, value, weights, evaluation=False), delta)
    evaluation_jacobian = torch.autograd.functional.jacobian(lambda value: _average_gradient(evaluation, value, weights, evaluation=True), delta)
    hessian = source_hessian(source, delta, weights)
    whitening = whiten(hessian, relative_damping)
    difference = evaluation_jacobian - source_jacobian
    return TaskGeometry(whitening.inverse_root @ difference, source_jacobian.detach(), evaluation_jacobian.detach(), difference.detach(), whitening)
