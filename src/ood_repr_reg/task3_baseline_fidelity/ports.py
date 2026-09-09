"""Faithful small ports of the baseline update equations.

The functions here are deliberately tiny and auditable.  They do not implement
the Task 3 local-response algorithm; they only encode the upstream identities
needed by the baseline-fidelity gate:

* Facebook IRM Colored MNIST scalar-scale IRMv1 objective.
* DomainBed IGA full-network gradient alignment objective.
* DomainBed/Fish clone, sequential inner update, and meta interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Iterable

import torch
from torch import Tensor, autograd, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ObjectiveAudit:
    objective: Tensor
    mean_loss: Tensor
    penalty: Tensor
    applied_penalty_weight: float
    rescaled_after_anneal: bool
    l2_weight_norm: Tensor


@dataclass(frozen=True)
class FishUpdateAudit:
    final_inner_loss: float
    meta_lr: float
    parameter_delta_norm: float
    optimizer_inner_state: dict


class BaselineMLP(nn.Module):
    """Small MLP with a named feature block and classifier head.

    Tests use the feature/head split to catch accidental head-only IGA ports.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 5, output_dim: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.features(x))


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)


def toy_minibatches(
    *,
    n_envs: int = 2,
    n: int = 8,
    input_dim: int = 4,
    output_dim: int = 1,
    seed: int = 17,
) -> tuple[tuple[Tensor, Tensor], ...]:
    generator = torch.Generator().manual_seed(seed)
    batches: list[tuple[Tensor, Tensor]] = []
    for env in range(n_envs):
        x = torch.randn(n, input_dim, generator=generator) + 0.25 * env
        linear = x[:, 0] - 0.7 * x[:, 1] + 0.2 * env
        if output_dim == 1:
            y = (linear > 0).float()[:, None]
        else:
            y = (linear > 0).long()
        batches.append((x, y))
    return tuple(batches)


def clone_module(module: nn.Module) -> nn.Module:
    return copy.deepcopy(module)


def parameter_vector(module: nn.Module) -> Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in module.parameters()])


def _parameters(module: nn.Module) -> tuple[nn.Parameter, ...]:
    params = tuple(module.parameters())
    if not params:
        raise ValueError("model has no parameters")
    return params


def _head_parameters(module: nn.Module) -> tuple[nn.Parameter, ...]:
    head = getattr(module, "head", None)
    if head is None:
        raise ValueError("model must expose a .head module for head-only surrogate audits")
    return tuple(head.parameters())


def _loss_from_logits(logits: Tensor, y: Tensor) -> Tensor:
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits, y.float())
    return F.cross_entropy(logits, y.long())


def _accuracy_from_logits(logits: Tensor, y: Tensor) -> float:
    if logits.shape[-1] == 1:
        pred = (logits > 0.0).float()
        return float((pred == y.float()).float().mean().detach().cpu())
    pred = logits.argmax(dim=1)
    return float((pred == y.long()).float().mean().detach().cpu())


def weight_norm_squared(module: nn.Module) -> Tensor:
    total = next(module.parameters()).new_zeros(())
    for parameter in module.parameters():
        total = total + parameter.norm().pow(2)
    return total


def erm_loss(module: nn.Module, minibatches: tuple[tuple[Tensor, Tensor], ...]) -> Tensor:
    all_x = torch.cat([x for x, _ in minibatches], dim=0)
    all_y = torch.cat([y for _, y in minibatches], dim=0)
    return _loss_from_logits(module(all_x), all_y)


def irmv1_penalty_from_logits(logits: Tensor, y: Tensor) -> Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = _loss_from_logits(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return grad.square().sum()


def irmv1_objective(
    module: nn.Module,
    minibatches: tuple[tuple[Tensor, Tensor], ...],
    *,
    step: int,
    penalty_anneal_iters: int = 100,
    penalty_weight: float = 10000.0,
    l2_regularizer_weight: float = 1e-3,
) -> ObjectiveAudit:
    env_losses: list[Tensor] = []
    env_penalties: list[Tensor] = []
    for x, y in minibatches:
        logits = module(x)
        env_losses.append(_loss_from_logits(logits, y))
        env_penalties.append(irmv1_penalty_from_logits(logits, y))
    mean_loss = torch.stack(env_losses).mean()
    penalty = torch.stack(env_penalties).mean()
    weight_norm = weight_norm_squared(module)
    applied = float(penalty_weight if step >= penalty_anneal_iters else 1.0)
    objective = mean_loss + l2_regularizer_weight * weight_norm + applied * penalty
    rescaled = applied > 1.0
    if rescaled:
        objective = objective / applied
    return ObjectiveAudit(
        objective=objective,
        mean_loss=mean_loss,
        penalty=penalty,
        applied_penalty_weight=applied,
        rescaled_after_anneal=rescaled,
        l2_weight_norm=weight_norm,
    )


def iga_objective(
    module: nn.Module,
    minibatches: tuple[tuple[Tensor, Tensor], ...],
    *,
    penalty_weight: float = 1000.0,
) -> ObjectiveAudit:
    params = _parameters(module)
    total_loss = next(module.parameters()).new_zeros(())
    grads: list[tuple[Tensor, ...]] = []
    for x, y in minibatches:
        env_loss = _loss_from_logits(module(x), y)
        total_loss = total_loss + env_loss
        grads.append(autograd.grad(env_loss, params, create_graph=True, retain_graph=True))
    mean_loss = total_loss / len(minibatches)
    mean_grad = autograd.grad(mean_loss, params, retain_graph=True)
    penalty = next(module.parameters()).new_zeros(())
    for grad in grads:
        for g_env, g_mean in zip(grad, mean_grad):
            penalty = penalty + (g_env - g_mean).pow(2).sum()
    objective = mean_loss + penalty_weight * penalty
    return ObjectiveAudit(
        objective=objective,
        mean_loss=mean_loss,
        penalty=penalty,
        applied_penalty_weight=float(penalty_weight),
        rescaled_after_anneal=False,
        l2_weight_norm=next(module.parameters()).new_zeros(()),
    )


def head_only_gradient_variance_surrogate(
    module: nn.Module,
    minibatches: tuple[tuple[Tensor, Tensor], ...],
) -> Tensor:
    params = _head_parameters(module)
    losses = [_loss_from_logits(module(x), y) for x, y in minibatches]
    grads = [autograd.grad(loss, params, create_graph=True, retain_graph=True) for loss in losses]
    means = tuple(torch.stack([grad[i] for grad in grads]).mean(dim=0) for i in range(len(params)))
    penalty = next(module.parameters()).new_zeros(())
    for grad in grads:
        for g_env, g_mean in zip(grad, means):
            penalty = penalty + (g_env - g_mean).pow(2).sum()
    return penalty


def head_gradient_variance_surrogate_notice() -> dict[str, str | bool]:
    return {
        "method_label": "HEAD_GRADIENT_VARIANCE_SURROGATE",
        "not_a_reproduction_of_iga_or_fish": True,
        "why_not_iga": "uses head-only gradients and earlier Task3 used a fixed penalty subset, while DomainBed IGA aligns full-network gradients over minibatches",
        "why_not_fish": "no inner clone, no sequential domain updates, no carried inner optimizer state, and no outer interpolation/meta-update",
    }


def one_step_update(
    module: nn.Module,
    objective: Tensor,
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> Tensor:
    before = parameter_vector(module)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()
    return parameter_vector(module) - before


def fish_outer_update(
    module: nn.Module,
    minibatches: tuple[tuple[Tensor, Tensor], ...],
    *,
    lr: float = 1e-3,
    meta_lr: float = 0.5,
    weight_decay: float = 0.0,
    optimizer_inner_state: dict | None = None,
) -> FishUpdateAudit:
    before_vector = parameter_vector(module)
    before_state = {key: value.detach().clone() for key, value in module.state_dict().items()}
    inner = clone_module(module)
    inner_optimizer = torch.optim.Adam(inner.parameters(), lr=lr, weight_decay=weight_decay)
    if optimizer_inner_state is not None:
        inner_optimizer.load_state_dict(optimizer_inner_state)

    final_loss = float("nan")
    for x, y in minibatches:
        loss = _loss_from_logits(inner(x), y)
        inner_optimizer.zero_grad()
        loss.backward()
        inner_optimizer.step()
        final_loss = float(loss.detach().cpu())

    inner_state = {key: value.detach().clone() for key, value in inner.state_dict().items()}
    meta_state = {
        key: before_state[key] + float(meta_lr) * (inner_state[key] - before_state[key])
        for key in before_state
    }
    module.load_state_dict(meta_state)
    delta = float((parameter_vector(module) - before_vector).norm().detach().cpu())
    return FishUpdateAudit(
        final_inner_loss=final_loss,
        meta_lr=float(meta_lr),
        parameter_delta_norm=delta,
        optimizer_inner_state=copy.deepcopy(inner_optimizer.state_dict()),
    )


def evaluate_source_only(module: nn.Module, minibatches: tuple[tuple[Tensor, Tensor], ...]) -> dict[str, float]:
    with torch.no_grad():
        losses = []
        accuracies = []
        for x, y in minibatches:
            logits = module(x)
            losses.append(float(_loss_from_logits(logits, y).detach().cpu()))
            accuracies.append(_accuracy_from_logits(logits, y))
    return {
        "mean_source_loss": float(sum(losses) / len(losses)),
        "mean_source_accuracy": float(sum(accuracies) / len(accuracies)),
        "worst_source_accuracy": float(min(accuracies)),
    }


def train_controlled_step(
    module: nn.Module,
    minibatches: tuple[tuple[Tensor, Tensor], ...],
    *,
    method: str,
    step: int,
    lr: float,
    irm_penalty_weight: float = 10000.0,
    iga_penalty_weight: float = 1000.0,
    fish_meta_lr: float = 0.5,
    l2_regularizer_weight: float = 1e-3,
    fish_optimizer_inner_state: dict | None = None,
) -> tuple[dict[str, float | str | bool], dict | None]:
    key = method.upper()
    if key == "ERM":
        audit = ObjectiveAudit(
            objective=erm_loss(module, minibatches) + l2_regularizer_weight * weight_norm_squared(module),
            mean_loss=erm_loss(module, minibatches),
            penalty=next(module.parameters()).new_zeros(()),
            applied_penalty_weight=0.0,
            rescaled_after_anneal=False,
            l2_weight_norm=weight_norm_squared(module),
        )
        delta = one_step_update(module, audit.objective, lr=lr)
        inner_state = None
    elif key == "IRMV1":
        audit = irmv1_objective(
            module,
            minibatches,
            step=step,
            penalty_weight=irm_penalty_weight,
            l2_regularizer_weight=l2_regularizer_weight,
        )
        delta = one_step_update(module, audit.objective, lr=lr)
        inner_state = None
    elif key == "IGA":
        audit = iga_objective(module, minibatches, penalty_weight=iga_penalty_weight)
        delta = one_step_update(module, audit.objective, lr=lr)
        inner_state = None
    elif key == "FISH":
        fish = fish_outer_update(
            module,
            minibatches,
            lr=lr,
            meta_lr=fish_meta_lr,
            optimizer_inner_state=fish_optimizer_inner_state,
        )
        return {
            "method": "Fish",
            "mean_loss": fish.final_inner_loss,
            "penalty": 0.0,
            "applied_penalty_weight": fish_meta_lr,
            "rescaled_after_anneal": False,
            "parameter_delta_norm": fish.parameter_delta_norm,
        }, fish.optimizer_inner_state
    else:
        raise ValueError(f"unknown baseline method: {method}")
    return {
        "method": key if key != "IRMV1" else "IRMv1",
        "mean_loss": float(audit.mean_loss.detach().cpu()),
        "penalty": float(audit.penalty.detach().cpu()),
        "applied_penalty_weight": float(audit.applied_penalty_weight),
        "rescaled_after_anneal": bool(audit.rescaled_after_anneal),
        "parameter_delta_norm": float(delta.norm().detach().cpu()),
    }, inner_state


def all_parameter_names(module: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in module.named_parameters())


def full_gradient_parameter_names(module: nn.Module, minibatches: tuple[tuple[Tensor, Tensor], ...]) -> tuple[str, ...]:
    audit = iga_objective(module, minibatches, penalty_weight=1000.0)
    grads = autograd.grad(audit.objective, tuple(module.parameters()), allow_unused=False)
    names = []
    for (name, _), grad in zip(module.named_parameters(), grads):
        if grad is not None and bool(torch.isfinite(grad).all()):
            names.append(name)
    return tuple(names)


def contains_target_reference(source: str) -> bool:
    lowered = source.lower()
    forbidden = ("target", "test_acc", "ood", "validation_target")
    return any(token in lowered for token in forbidden)
