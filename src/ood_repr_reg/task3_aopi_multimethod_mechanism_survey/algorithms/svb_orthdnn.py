"""Singular Value Bounding / OrthDNN projection method."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import AlgorithmState, BaseAlgorithm, SmoothStepResult, StepResult, smooth_source_risk_vector


def svb_project_weight(weight: Tensor, *, factor: float) -> Tensor:
    lower = 1.0 / (1.0 + factor)
    upper = 1.0 + factor
    u, s, vh = torch.linalg.svd(weight.detach(), full_matrices=False)
    return (u * s.clamp(lower, upper)) @ vh


def apply_svb_projection(model: nn.Module, *, factor: float) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(svb_project_weight(module.weight, factor=factor))


class SVBOrthDNNAlgorithm(BaseAlgorithm):
    name = "SVB_ORTHDNN"
    formula_id = "CMNIST_SVB_ORTHDNN_POST_STEP_PROJECTION_V1"
    reference_id = "ORTHOGONAL_DEEP_NEURAL_NETWORKS_SVB"
    variant_id = "SVB_FACTOR0P05_ALL_LINEAR_POST_STEP"
    admission_role = "SPECTRAL_GEOMETRY"
    projection_or_svd_operations_per_step = 3.0

    @property
    def factor(self) -> float:
        return float(self.config["svb_orthdnn"]["svb_factor"])

    @property
    def projection_frequency(self) -> int:
        return int(self.config["svb_orthdnn"]["projection_frequency"])

    def should_project(self, step: int) -> bool:
        return (int(step) + 1) % self.projection_frequency == 0

    def objective(self, model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int) -> ObjectiveParts:
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        l2 = weight_norm_squared(model)
        zero = risk.detach().new_zeros(())
        return ObjectiveParts(risk + self.l2_weight * l2, risk, l2, zero, 0.0, False)

    def smooth_objective(self, model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int) -> Tensor:
        return smooth_source_risk_vector(model, worlds, indices, delta).mean() + self.l2_weight * self.l2_norm(model)

    def train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]], *, step: int, learning_rate: float, algorithm_state: AlgorithmState) -> StepResult:
        result = super().train_step(model, optimizer, batches, step=step, learning_rate=learning_rate, algorithm_state=algorithm_state)
        state = result.algorithm_state.clone()
        if self.should_project(step):
            apply_svb_projection(model, factor=self.factor)
            state.payload["projection_count"] = int(state.payload.get("projection_count", 0)) + 1
        return StepResult(result.parts, result.optimizer, state, result.did_reset_optimizer)

    def smooth_train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int, learning_rate: float, algorithm_state: AlgorithmState) -> SmoothStepResult:
        result = super().smooth_train_step(model, optimizer, worlds, indices, delta, step=step, learning_rate=learning_rate, algorithm_state=algorithm_state)
        state = result.algorithm_state.clone()
        if self.should_project(step):
            apply_svb_projection(model, factor=self.factor)
            state.payload["projection_count"] = int(state.payload.get("projection_count", 0)) + 1
        return SmoothStepResult(result.objective, result.optimizer, state, result.did_reset_optimizer, result.finite)
