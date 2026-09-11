"""First-order MLDG algorithm definition for the expanded CMNIST survey."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothPool, SmoothWorld5, environment_parameters, outcome_weight
from .base import AlgorithmState, BaseAlgorithm, SmoothStepResult, StepResult


def _ordered_parameters(model: nn.Module) -> tuple[list[str], tuple[nn.Parameter, ...]]:
    items = list(model.named_parameters())
    return [name for name, _ in items], tuple(parameter for _, parameter in items)


def _functional_logits(model: nn.Module, parameters: dict[str, Tensor], images: Tensor) -> Tensor:
    return functional_call(model, parameters, (images,))


def _smooth_env_risk_functional(model: nn.Module, parameters: dict[str, Tensor], pool: SmoothPool, indices: Tensor, p: Tensor, q: Tensor) -> Tensor:
    value = next(model.parameters()).new_zeros(())
    for images, labels, label_flip, color_flip in pool.outcome_batches(indices):
        logits = _functional_logits(model, parameters, images)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        value = value + outcome_weight(p, q, label_flip, color_flip).to(loss) * loss
    return value


class MLDGAlgorithm(BaseAlgorithm):
    name = "MLDG"
    formula_id = "CMNIST_FIRST_ORDER_MLDG_V1"
    reference_id = "MLDG_FIRST_ORDER_SOURCE_ENV_META_SPLIT"
    variant_id = "MLDG_BETA1_INNERLR0P001_ONE_STEP"
    admission_role = "INTERMEDIATE"

    @property
    def beta(self) -> float:
        return float(self.config["mldg"]["beta"])

    @property
    def inner_lr(self) -> float:
        return float(self.config["mldg"]["inner_lr"])

    def _roles(self, step: int) -> tuple[int, int]:
        return (0, 1) if step % 2 == 0 else (1, 0)

    def _state_after(self, algorithm_state: AlgorithmState, *, step: int, train_env: int, test_env: int) -> AlgorithmState:
        payload = dict(algorithm_state.payload)
        payload.update({"steps_completed": int(step) + 1, "last_meta_train_env": int(train_env), "last_meta_test_env": int(test_env)})
        return AlgorithmState(payload)

    def objective(
        self,
        model: nn.Module,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
    ) -> ObjectiveParts:
        losses = torch.stack(source_environment_losses(model, batches))
        train_env, test_env = self._roles(step)
        risk = losses.mean()
        l2 = weight_norm_squared(model)
        meta_gap = losses[test_env]
        objective = losses[train_env] + self.beta * meta_gap + self.l2_weight * l2
        return ObjectiveParts(objective, risk, l2, meta_gap, self.beta, False)

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
        learning_rate: float,
        algorithm_state: AlgorithmState,
    ) -> StepResult:
        train_env, test_env = self._roles(step)
        names, parameters = _ordered_parameters(model)
        x_i, y_i = batches[train_env]
        x_j, y_j = batches[test_env]
        loss_i = F.binary_cross_entropy_with_logits(model(x_i), y_i.float())
        grads_i = torch.autograd.grad(loss_i, parameters, create_graph=False, retain_graph=False)
        fast_params = {name: parameter - self.inner_lr * grad for name, parameter, grad in zip(names, parameters, grads_i)}
        loss_j = F.binary_cross_entropy_with_logits(_functional_logits(model, fast_params, x_j), y_j.float())
        grads_j = torch.autograd.grad(loss_j, tuple(fast_params.values()), create_graph=False, retain_graph=False)
        l2 = self.l2_weight * self.l2_norm(model)
        grads_l2 = torch.autograd.grad(l2, parameters, create_graph=False, retain_graph=False)
        optimizer.zero_grad(set_to_none=True)
        for parameter, grad_i, grad_j, grad_l2 in zip(parameters, grads_i, grads_j, grads_l2):
            parameter.grad = (grad_i + self.beta * grad_j + grad_l2).detach().clone()
        optimizer.step()
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        l2_value = weight_norm_squared(model)
        parts = ObjectiveParts((loss_i + self.beta * loss_j + l2).detach(), risk.detach(), l2_value.detach(), loss_j.detach(), self.beta, False)
        return StepResult(parts, optimizer, self._state_after(algorithm_state, step=step, train_env=train_env, test_env=test_env), False)

    def smooth_objective(
        self,
        model: nn.Module,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
    ) -> Tensor:
        train_env, test_env = self._roles(step)
        parameters = dict(model.named_parameters())
        p_i, q_i = environment_parameters(delta, environment=train_env, evaluation=False)
        p_j, q_j = environment_parameters(delta, environment=test_env, evaluation=False)
        return (
            _smooth_env_risk_functional(model, parameters, worlds.source[train_env], indices[train_env], p_i, q_i)
            + self.beta * _smooth_env_risk_functional(model, parameters, worlds.source[test_env], indices[test_env], p_j, q_j)
            + self.l2_weight * self.l2_norm(model)
        )

    def smooth_train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
        learning_rate: float,
        algorithm_state: AlgorithmState,
    ) -> SmoothStepResult:
        train_env, test_env = self._roles(step)
        names, parameters = _ordered_parameters(model)
        current = {name: parameter for name, parameter in zip(names, parameters)}
        p_i, q_i = environment_parameters(delta, environment=train_env, evaluation=False)
        p_j, q_j = environment_parameters(delta, environment=test_env, evaluation=False)
        loss_i = _smooth_env_risk_functional(model, current, worlds.source[train_env], indices[train_env], p_i, q_i)
        grads_i = torch.autograd.grad(loss_i, parameters, create_graph=False, retain_graph=False)
        fast_params = {name: parameter - self.inner_lr * grad for name, parameter, grad in zip(names, parameters, grads_i)}
        loss_j = _smooth_env_risk_functional(model, fast_params, worlds.source[test_env], indices[test_env], p_j, q_j)
        grads_j = torch.autograd.grad(loss_j, tuple(fast_params.values()), create_graph=False, retain_graph=False)
        l2 = self.l2_weight * self.l2_norm(model)
        grads_l2 = torch.autograd.grad(l2, parameters, create_graph=False, retain_graph=False)
        optimizer.zero_grad(set_to_none=True)
        for parameter, grad_i, grad_j, grad_l2 in zip(parameters, grads_i, grads_j, grads_l2):
            parameter.grad = (grad_i + self.beta * grad_j + grad_l2).detach().clone()
        optimizer.step()
        objective = (loss_i + self.beta * loss_j + l2).detach()
        return SmoothStepResult(objective, optimizer, self._state_after(algorithm_state, step=step, train_env=train_env, test_env=test_env), False, bool(torch.isfinite(objective)))
