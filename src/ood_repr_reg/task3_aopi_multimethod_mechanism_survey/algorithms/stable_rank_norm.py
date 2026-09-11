"""Stable-rank normalization by post-step singular-value projection."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import AlgorithmState, BaseAlgorithm, SmoothStepResult, StepResult, smooth_source_risk_vector


def stable_rank_project_weight(weight: Tensor, *, target_rank: float, spectral_norm_target: float) -> Tensor:
    u, s, vh = torch.linalg.svd(weight.detach(), full_matrices=False)
    if s.numel() <= 1 or float(s[0]) <= 0.0:
        return weight.detach().clone()
    target = max(1.0, min(float(target_rank), float(s.numel())))
    tail_sq = s[1:].square().sum()
    if float(tail_sq) > 0.0:
        desired_tail_sq = max(target - 1.0, 0.0) * s[0].square()
        scale = torch.sqrt(desired_tail_sq / tail_sq).clamp(max=1.0)
        s = torch.cat((s[:1], s[1:] * scale))
    if spectral_norm_target > 0.0 and float(s[0]) > 0.0:
        s = s * (float(spectral_norm_target) / s[0])
    return (u * s) @ vh


def apply_stable_rank_projection(model: nn.Module, *, target_rank: float, spectral_norm_target: float) -> None:
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(stable_rank_project_weight(module.weight, target_rank=target_rank, spectral_norm_target=spectral_norm_target))


class StableRankNormAlgorithm(BaseAlgorithm):
    name = "STABLE_RANK_NORM"
    formula_id = "CMNIST_STABLE_RANK_NORM_POST_STEP_PROJECTION_V1"
    reference_id = "STABLE_RANK_NORMALIZATION_ICLR2020"
    variant_id = "SRN_TARGET8_SIGMA1_TARGET1_ALL_LINEAR"
    admission_role = "SPECTRAL_GEOMETRY"
    projection_or_svd_operations_per_step = 3.0

    @property
    def target_rank(self) -> float:
        return float(self.config["stable_rank_norm"]["target_rank"])

    @property
    def spectral_norm_target(self) -> float:
        return float(self.config["stable_rank_norm"]["spectral_norm_target"])

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
        apply_stable_rank_projection(model, target_rank=self.target_rank, spectral_norm_target=self.spectral_norm_target)
        state = result.algorithm_state.clone()
        state.payload["projection_count"] = int(state.payload.get("projection_count", 0)) + 1
        return StepResult(result.parts, result.optimizer, state, result.did_reset_optimizer)

    def smooth_train_step(self, model: nn.Module, optimizer: torch.optim.Optimizer, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, *, step: int, learning_rate: float, algorithm_state: AlgorithmState) -> SmoothStepResult:
        result = super().smooth_train_step(model, optimizer, worlds, indices, delta, step=step, learning_rate=learning_rate, algorithm_state=algorithm_state)
        apply_stable_rank_projection(model, target_rank=self.target_rank, spectral_norm_target=self.spectral_norm_target)
        state = result.algorithm_state.clone()
        state.payload["projection_count"] = int(state.payload.get("projection_count", 0)) + 1
        return SmoothStepResult(result.objective, result.optimizer, state, result.did_reset_optimizer, result.finite)
