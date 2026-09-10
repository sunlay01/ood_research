"""Mathematical objectives for the CPU-minimal ColoredMNIST probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResponsePenalty:
    penalty: Tensor
    raw_head_grads: Tensor
    centered_head_grads: Tensor
    hessian: Tensor | None = None
    damped_hessian: Tensor | None = None
    damping: float | None = None
    min_eigenvalue: float | None = None
    condition_number: float | None = None
    solver: str = "identity"


@dataclass(frozen=True)
class ObjectiveParts:
    objective: Tensor
    risk: Tensor
    l2_weight_norm: Tensor
    penalty: Tensor
    applied_penalty_weight: float
    rescaled_after_anneal: bool


@dataclass(frozen=True)
class Calibration:
    scale_c: float
    risk_grad_norm: float
    penalty_grad_norm: float
    realized_initial_update_ratio: float
    valid: bool
    reason: str


def mean_nll(logits: Tensor, labels: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def weight_norm_squared(model: nn.Module) -> Tensor:
    total = next(model.parameters()).new_zeros(())
    for parameter in model.parameters():
        total = total + parameter.norm().pow(2)
    return total


def source_environment_losses(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    return tuple(mean_nll(model(x), y) for x, y in batches)  # type: ignore[return-value]


def source_forward_with_features(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    losses: list[Tensor] = []
    features: list[Tensor] = []
    logits: list[Tensor] = []
    for x, y in batches:
        z = model.encode(x)
        out = model.head(z)
        losses.append(mean_nll(out, y))
        features.append(z)
        logits.append(out)
    return (losses[0], losses[1]), (features[0], features[1]), (logits[0], logits[1])


def base_source_objective(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    l2_regularizer_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    losses = source_environment_losses(model, batches)
    risk = torch.stack(losses).mean()
    l2 = weight_norm_squared(model)
    return risk + l2_regularizer_weight * l2, risk, l2


def irmv1_penalty_from_logits(logits: Tensor, labels: Tensor) -> Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    risk = mean_nll(logits * scale, labels)
    grad = autograd.grad(risk, [scale], create_graph=True)[0]
    return grad.square().sum()


def irmv1_objective(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    step: int,
    l2_regularizer_weight: float,
    penalty_anneal_iters: int,
    penalty_weight: float,
) -> ObjectiveParts:
    losses = []
    penalties = []
    for x, y in batches:
        logits = model(x)
        losses.append(mean_nll(logits, y))
        penalties.append(irmv1_penalty_from_logits(logits, y))
    risk = torch.stack(losses).mean()
    l2 = weight_norm_squared(model)
    penalty = torch.stack(penalties).mean()
    applied = float(penalty_weight if step >= penalty_anneal_iters else 1.0)
    objective = risk + l2_regularizer_weight * l2 + applied * penalty
    rescaled = bool(applied > 1.0)
    if rescaled:
        objective = objective / applied
    return ObjectiveParts(objective, risk, l2, penalty, applied, rescaled)


def head_gradient_vector(loss: Tensor, head: nn.Linear) -> Tensor:
    grads = autograd.grad(
        loss,
        (head.weight, head.bias),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    return torch.cat([grads[0].reshape(-1), grads[1].reshape(-1)])


def centered_head_gradients(losses: tuple[Tensor, Tensor], head: nn.Linear) -> tuple[Tensor, Tensor]:
    raw = torch.stack([head_gradient_vector(loss, head) for loss in losses])
    centered = raw - raw.mean(dim=0, keepdim=True)
    return raw, centered


def identity_metric_penalty(centered_grads: Tensor) -> Tensor:
    return 0.5 * centered_grads.square().sum()


def analytic_head_hessian(features: tuple[Tensor, Tensor], logits: tuple[Tensor, Tensor]) -> Tensor:
    with torch.no_grad():
        z = torch.cat([item.detach() for item in features], dim=0).double()
        out = torch.cat([item.detach() for item in logits], dim=0).reshape(-1).double()
        ones = torch.ones((z.shape[0], 1), dtype=z.dtype, device=z.device)
        augmented = torch.cat([z, ones], dim=1)
        coeff = torch.sigmoid(out) * (1.0 - torch.sigmoid(out)) / max(int(out.numel()), 1)
        hessian = augmented.T @ (augmented * coeff[:, None])
        return ((hessian + hessian.T) / 2.0).detach()


def damped_head_hessian(hessian: Tensor, damping_epsilon: float) -> tuple[Tensor, float, Tensor, float]:
    sym = ((hessian.detach().double() + hessian.detach().double().T) / 2.0).detach()
    dim = sym.shape[0]
    trace_scale = float(torch.trace(sym).item() / max(dim, 1))
    damping = float(damping_epsilon) * max(trace_scale, 1e-8)
    matrix = sym + damping * torch.eye(dim, dtype=sym.dtype, device=sym.device)
    chol = torch.linalg.cholesky(matrix)
    eigenvalues = torch.linalg.eigvalsh(matrix)
    condition = float(eigenvalues.max().item() / max(eigenvalues.min().item(), 1e-12))
    return matrix.detach(), damping, chol.detach(), condition


def inverse_hessian_metric_penalty(
    centered_grads: Tensor,
    hessian: Tensor,
    *,
    damping_epsilon: float,
) -> tuple[Tensor, Tensor, float, float, float]:
    matrix, damping, chol, condition = damped_head_hessian(hessian, damping_epsilon)
    rhs = centered_grads.double().T
    solved = torch.cholesky_solve(rhs, chol)
    penalty = 0.5 * torch.sum(centered_grads.double().T * solved)
    min_eigenvalue = float(torch.linalg.eigvalsh(matrix).min().item())
    return penalty, matrix, damping, min_eigenvalue, condition


def grad_response_penalty(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
) -> ResponsePenalty:
    losses, _, _ = source_forward_with_features(model, batches)
    raw, centered = centered_head_gradients(losses, model.head)
    return ResponsePenalty(
        penalty=identity_metric_penalty(centered),
        raw_head_grads=raw,
        centered_head_grads=centered,
        solver="identity",
    )


def local_response_penalty(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    damping_epsilon: float,
) -> ResponsePenalty:
    losses, features, logits = source_forward_with_features(model, batches)
    raw, centered = centered_head_gradients(losses, model.head)
    hessian = analytic_head_hessian(features, logits)
    penalty, damped, damping, min_eigenvalue, condition = inverse_hessian_metric_penalty(
        centered,
        hessian,
        damping_epsilon=damping_epsilon,
    )
    return ResponsePenalty(
        penalty=penalty,
        raw_head_grads=raw,
        centered_head_grads=centered,
        hessian=hessian,
        damped_hessian=damped,
        damping=damping,
        min_eigenvalue=min_eigenvalue,
        condition_number=condition,
        solver="cholesky",
    )


def response_penalty_for_method(
    method: str,
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    damping_epsilon: float,
) -> ResponsePenalty:
    if method == "GRAD":
        return grad_response_penalty(model, batches)
    if method == "LOCAL_RESPONSE":
        return local_response_penalty(model, batches, damping_epsilon=damping_epsilon)
    raise ValueError(f"response penalty is undefined for {method}")


def gradient_norm(objective: Tensor, parameters: tuple[nn.Parameter, ...]) -> float:
    grads = autograd.grad(objective, parameters, retain_graph=False, allow_unused=True)
    total = objective.detach().new_zeros(())
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().double().square().sum()
    return float(torch.sqrt(total).item())


def calibrate_response_scale(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    method: str,
    l2_regularizer_weight: float,
    damping_epsilon: float,
    calibration_update_ratio: float,
) -> Calibration:
    parameters = tuple(model.parameters())
    base, _, _ = base_source_objective(
        model,
        batches,
        l2_regularizer_weight=l2_regularizer_weight,
    )
    risk_norm = gradient_norm(base, parameters)
    response = response_penalty_for_method(
        method,
        model,
        batches,
        damping_epsilon=damping_epsilon,
    )
    penalty_norm = gradient_norm(response.penalty, parameters)
    valid = bool(penalty_norm >= 1e-12 and risk_norm > 0.0)
    if not valid:
        return Calibration(float("nan"), risk_norm, penalty_norm, float("nan"), False, "INVALID_ZERO_PENALTY_GRAD")
    scale_c = risk_norm / (penalty_norm + 1e-12)
    realized = calibration_update_ratio * scale_c * penalty_norm / risk_norm
    return Calibration(float(scale_c), risk_norm, penalty_norm, float(realized), True, "OK")


def method_objective(
    method: str,
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    step: int,
    config: dict[str, Any],
    calibration: Calibration | None = None,
) -> ObjectiveParts:
    training_cfg = config["training"]
    response_cfg = config["response"]
    if method == "IRMv1":
        return irmv1_objective(
            model,
            batches,
            step=step,
            l2_regularizer_weight=float(training_cfg["l2_regularizer_weight"]),
            penalty_anneal_iters=int(config["irmv1"]["penalty_anneal_iters"]),
            penalty_weight=float(config["irmv1"]["penalty_weight"]),
        )
    base, risk, l2 = base_source_objective(
        model,
        batches,
        l2_regularizer_weight=float(training_cfg["l2_regularizer_weight"]),
    )
    if method == "ERM":
        zero = base.detach().new_zeros(())
        return ObjectiveParts(base, risk, l2, zero, 0.0, False)
    if method in {"GRAD", "LOCAL_RESPONSE"}:
        if calibration is None or not calibration.valid:
            raise ValueError(f"valid source-only calibration is required for {method}")
        response = response_penalty_for_method(
            method,
            model,
            batches,
            damping_epsilon=float(response_cfg["damping_epsilon"]),
        )
        applied = float(response_cfg["calibration_update_ratio"]) * calibration.scale_c
        return ObjectiveParts(base + applied * response.penalty, risk, l2, response.penalty, applied, False)
    raise ValueError(f"unknown method: {method}")
