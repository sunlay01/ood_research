"""Source-only frozen-encoder head objectives and exact IFT responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F


HEAD_DIMENSION = 65
L2_WEIGHT = 0.001
IRM_WEIGHT = 10000.0
IFT_DAMPING_RELATIVE = 1e-8
# The faithful post-anneal IRMv1 objective is divided by 10000, so its
# absolute gradient scale is not comparable to the ERM objective scale.
REFERENCE_GRADIENT_NORM_MAX = 1e-4


@dataclass(frozen=True)
class HeadReference:
    method: str
    weights: Tensor
    original_weights: Tensor
    source_objective: float
    gradient_norm: float
    initial_displacement_norm: float
    converged: bool


@dataclass(frozen=True)
class PiResult:
    pi: Tensor
    objective_hessian: Tensor
    effective_hessian: Tensor
    damping: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float


def augmented_head_weights(weight: Tensor, bias: Tensor) -> Tensor:
    return torch.cat((weight.detach().cpu().double().reshape(-1), bias.detach().cpu().double().reshape(-1)))


def split_head_weights(weights: Tensor) -> tuple[Tensor, Tensor]:
    w = weights.reshape(-1).double()
    if w.numel() != HEAD_DIMENSION:
        raise ValueError("frozen CMNIST head must have 64 weights plus one bias")
    return w[:-1], w[-1]


def head_logits(features: Tensor, weights: Tensor) -> Tensor:
    weight, bias = split_head_weights(weights)
    return features.double() @ weight[:, None] + bias


def risk_loss(features: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(head_logits(features, weights), labels.double())


def irm_penalty(features: Tensor, labels: Tensor, weights: Tensor) -> Tensor:
    scale = torch.ones((), dtype=weights.dtype, device=weights.device, requires_grad=True)
    loss = F.binary_cross_entropy_with_logits(head_logits(features, weights) * scale, labels.double())
    gradient = torch.autograd.grad(loss, scale, create_graph=True)[0]
    return gradient.square()


def source_objective(
    method: str,
    source_features: tuple[Tensor, Tensor],
    source_labels: tuple[Tensor, Tensor],
    weights: Tensor,
) -> Tensor:
    risks = tuple(risk_loss(x, y, weights) for x, y in zip(source_features, source_labels))
    risk = torch.stack(risks).mean()
    l2 = weights.square().sum()
    if method == "ERM":
        return risk + L2_WEIGHT * l2
    if method == "IRMv1":
        penalties = tuple(irm_penalty(x, y, weights) for x, y in zip(source_features, source_labels))
        return (risk + L2_WEIGHT * l2 + IRM_WEIGHT * torch.stack(penalties).mean()) / IRM_WEIGHT
    raise ValueError(f"unsupported audit method: {method}")


def refine_head(
    method: str,
    source_features: tuple[Tensor, Tensor],
    source_labels: tuple[Tensor, Tensor],
    initial_weights: Tensor,
    *,
    max_iter: int = 100,
    tolerance_grad: float = 1e-9,
    tolerance_change: float = 1e-12,
) -> HeadReference:
    original = initial_weights.detach().cpu().double().clone()
    weights = torch.nn.Parameter(original.clone())
    optimizer = torch.optim.LBFGS(
        [weights], lr=1.0, max_iter=int(max_iter), tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change), line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        objective = source_objective(method, source_features, source_labels, weights)
        objective.backward()
        return objective

    objective = optimizer.step(closure)
    final_objective = source_objective(method, source_features, source_labels, weights)
    gradient = torch.autograd.grad(final_objective, weights)[0]
    gradient_norm = float(gradient.detach().norm().item())
    return HeadReference(
        method=method,
        weights=weights.detach().cpu().double().clone(),
        original_weights=original,
        source_objective=float(final_objective.detach().cpu().item()),
        gradient_norm=gradient_norm,
        initial_displacement_norm=float((weights.detach().cpu() - original).norm().item()),
        converged=bool(torch.isfinite(final_objective.detach()) and gradient_norm <= REFERENCE_GRADIENT_NORM_MAX),
    )


def source_state(
    source_features: tuple[Tensor, Tensor],
    source_labels: tuple[Tensor, Tensor],
    weights: Tensor,
    *,
    create_graph: bool,
) -> Tensor:
    state_parts: list[Tensor] = []
    for features, labels in zip(source_features, source_labels):
        local = weights if weights.requires_grad else weights.detach().clone().requires_grad_(True)
        risk = risk_loss(features, labels, local)
        penalty = irm_penalty(features, labels, local)
        state_parts.extend((
            torch.autograd.grad(risk, local, create_graph=create_graph, retain_graph=True)[0],
            torch.autograd.grad(penalty, local, create_graph=create_graph, retain_graph=True)[0],
        ))
    return torch.cat(state_parts)


def selector(method: str, *, dtype: torch.dtype = torch.double) -> Tensor:
    """Map `[risk0, penalty0, risk1, penalty1]` gradient state to F's forcing."""
    output = torch.zeros((HEAD_DIMENSION, 4 * HEAD_DIMENSION), dtype=dtype)
    risk_scale = 0.5 if method == "ERM" else 0.5 / IRM_WEIGHT
    penalty_scale = 0.0 if method == "ERM" else 0.5
    for block, scale in ((0, risk_scale), (1, penalty_scale), (2, risk_scale), (3, penalty_scale)):
        output[:, block * HEAD_DIMENSION:(block + 1) * HEAD_DIMENSION] = torch.eye(HEAD_DIMENSION, dtype=dtype) * scale
    return output


def objective_hessian(method: str, source_features: tuple[Tensor, Tensor], source_labels: tuple[Tensor, Tensor], weights: Tensor) -> Tensor:
    point = weights.detach().cpu().double().clone().requires_grad_(True)
    hessian = torch.autograd.functional.hessian(lambda value: source_objective(method, source_features, source_labels, value), point)
    return ((hessian + hessian.T) / 2.0).detach().cpu().double()


def pi_operator(
    method: str,
    source_features: tuple[Tensor, Tensor],
    source_labels: tuple[Tensor, Tensor],
    weights: Tensor,
    source_root: Tensor,
) -> PiResult:
    hessian = objective_hessian(method, source_features, source_labels, weights)
    scale = max(float(torch.trace(hessian).abs().item() / HEAD_DIMENSION), 1e-8)
    damping = IFT_DAMPING_RELATIVE * scale
    effective = hessian + damping * torch.eye(HEAD_DIMENSION, dtype=torch.double)
    eigenvalues = torch.linalg.eigvalsh(effective)
    singular_values = torch.linalg.svdvals(effective)
    if float(singular_values.min()) <= 1e-12 * max(float(singular_values.max()), 1.0):
        raise ValueError("effective IFT Jacobian is numerically singular")
    response = -source_root @ torch.linalg.solve(effective, selector(method))
    return PiResult(
        pi=response,
        objective_hessian=hessian,
        effective_hessian=effective,
        damping=damping,
        min_eigenvalue=float(eigenvalues.min().item()),
        max_eigenvalue=float(eigenvalues.max().item()),
        condition_number=float((singular_values.max() / singular_values.min()).item()),
    )


def feature_tuple(items: Iterable[Tensor]) -> tuple[Tensor, Tensor]:
    result = tuple(item.detach().cpu().double() for item in items)
    if len(result) != 2:
        raise ValueError("expected two source environments")
    return result[0], result[1]
