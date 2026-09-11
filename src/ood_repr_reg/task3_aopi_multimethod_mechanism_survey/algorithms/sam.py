"""Sharpness-Aware Minimization under the common CMNIST harness."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import AlgorithmState, BaseAlgorithm, SmoothStepResult, StepResult, smooth_source_risk_vector


def parameter_grad_norm(model: nn.Module, *, adaptive: bool = False, eta: float = 0.0) -> Tensor:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device)
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        scale = parameter.detach().abs() + eta if adaptive else torch.ones_like(parameter)
        total = total + (scale * parameter.grad).square().sum()
    return total.sqrt()


def make_sam_perturbations(model: nn.Module, *, rho: float, adaptive: bool = False, eta: float = 0.0) -> list[tuple[nn.Parameter, Tensor]]:
    norm = parameter_grad_norm(model, adaptive=adaptive, eta=eta)
    scale = float(rho) / (norm + 1e-12)
    perturbations: list[tuple[nn.Parameter, Tensor]] = []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        factor = (parameter.detach().abs() + eta).square() if adaptive else torch.ones_like(parameter)
        perturbation = factor * parameter.grad * scale.to(parameter)
        perturbations.append((parameter, perturbation.detach().clone()))
    return perturbations


def apply_perturbations(perturbations: list[tuple[nn.Parameter, Tensor]], *, sign: float) -> None:
    with torch.no_grad():
        for parameter, perturbation in perturbations:
            parameter.add_(perturbation, alpha=sign)


def restore_parameters(originals: list[tuple[nn.Parameter, Tensor]]) -> None:
    with torch.no_grad():
        for parameter, original in originals:
            parameter.copy_(original)


class SAMAlgorithm(BaseAlgorithm):
    name = "SAM"
    formula_id = "CMNIST_SAM_TWO_STEP_SOURCE_RISK_V1"
    reference_id = "FORET_2020_SHARPNESS_AWARE_MINIMIZATION"
    variant_id = "SAM_RHO0P05_COMMON_ADAM"
    admission_role = "FLATNESS_GEOMETRY"
    forward_pass_equivalents_per_step = 2.0
    backward_pass_equivalents_per_step = 2.0

    @property
    def rho(self) -> float:
        return float(self.config["sam"]["rho"])

    @property
    def adaptive(self) -> bool:
        return False

    @property
    def eta(self) -> float:
        return 0.0

    def _source_objective(self, model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]) -> ObjectiveParts:
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        l2 = weight_norm_squared(model)
        zero = risk.detach().new_zeros(())
        return ObjectiveParts(risk + self.l2_weight * l2, risk, l2, zero, 0.0, False)

    def objective(self, model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int) -> ObjectiveParts:
        return self._source_objective(model, batches)

    def smooth_objective(self, model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int) -> Tensor:
        return smooth_source_risk_vector(model, worlds, indices, delta).mean() + self.l2_weight * self.l2_norm(model)

    def train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int, learning_rate: float, algorithm_state: AlgorithmState) -> StepResult:
        first = self._source_objective(model, batches)
        optimizer.zero_grad(set_to_none=True)
        first.objective.backward()
        perturbations = make_sam_perturbations(model, rho=self.rho, adaptive=self.adaptive, eta=self.eta)
        originals = [(parameter, parameter.detach().clone()) for parameter, _ in perturbations]
        apply_perturbations(perturbations, sign=1.0)
        second = self._source_objective(model, batches)
        optimizer.zero_grad(set_to_none=True)
        second.objective.backward()
        restore_parameters(originals)
        optimizer.step()
        state = algorithm_state.clone()
        state.payload["sam_steps"] = int(state.payload.get("sam_steps", 0)) + 1
        state.payload["adaptive"] = self.adaptive
        state.payload["rho"] = self.rho
        return StepResult(second, optimizer, state, False)

    def smooth_train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int, learning_rate: float, algorithm_state: AlgorithmState) -> SmoothStepResult:
        first = self.smooth_objective(model, worlds, indices, delta, step=step)
        optimizer.zero_grad(set_to_none=True)
        first.backward()
        perturbations = make_sam_perturbations(model, rho=self.rho, adaptive=self.adaptive, eta=self.eta)
        originals = [(parameter, parameter.detach().clone()) for parameter, _ in perturbations]
        apply_perturbations(perturbations, sign=1.0)
        second = self.smooth_objective(model, worlds, indices, delta, step=step)
        optimizer.zero_grad(set_to_none=True)
        second.backward()
        restore_parameters(originals)
        optimizer.step()
        state = algorithm_state.clone()
        state.payload["sam_steps"] = int(state.payload.get("sam_steps", 0)) + 1
        state.payload["adaptive"] = self.adaptive
        state.payload["rho"] = self.rho
        return SmoothStepResult(second.detach(), optimizer, state, False, bool(torch.isfinite(second.detach())))
