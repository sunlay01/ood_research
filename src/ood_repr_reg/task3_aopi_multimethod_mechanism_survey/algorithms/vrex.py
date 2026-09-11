"""V-REx algorithm definition for the multi-method CMNIST survey."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm, smooth_source_risk_vector


def vrex_penalty_from_losses(losses: Tensor) -> Tensor:
    if losses.ndim != 1 or losses.numel() < 2:
        raise ValueError("V-REx requires a vector of at least two environment risks")
    if losses.numel() != 2:
        raise ValueError("this CMNIST survey pins V-REx to exactly two source environments")
    return (losses[0] - losses[1]).square()


class VRExAlgorithm(BaseAlgorithm):
    name = "VREX"
    formula_id = "CMNIST_VREX_ANNEALED_V1"
    reference_id = "REx_OFFICIAL_COLOREDMNIST_SCALE"
    variant_id = "VREX_LAMBDA10000_ANNEAL100_RESCALE"
    admission_role = "INTERMEDIATE"

    def prepare_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        step: int,
        learning_rate: float,
    ) -> tuple[torch.optim.Optimizer, bool]:
        if step == int(self.config["vrex"]["penalty_anneal_iters"]):
            return torch.optim.Adam(model.parameters(), lr=learning_rate), True
        return optimizer, False

    def applied_weight(self, step: int) -> float:
        settings = self.config["vrex"]
        return float(
            settings["post_anneal_penalty_weight"]
            if step >= int(settings["penalty_anneal_iters"])
            else settings["pre_anneal_penalty_weight"]
        )

    def maybe_rescale(self, objective: Tensor, applied: float) -> tuple[Tensor, bool]:
        rescaled = bool(self.config["vrex"].get("whole_loss_rescale_after_anneal") and applied > 1.0)
        return (objective / applied if rescaled else objective), rescaled

    def objective(
        self,
        model: nn.Module,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
    ) -> ObjectiveParts:
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        l2 = weight_norm_squared(model)
        penalty = vrex_penalty_from_losses(losses)
        applied = self.applied_weight(step)
        objective, rescaled = self.maybe_rescale(risk + self.l2_weight * l2 + applied * penalty, applied)
        return ObjectiveParts(objective, risk, l2, penalty, applied, rescaled)

    def smooth_objective(
        self,
        model: nn.Module,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
    ) -> Tensor:
        risk_vector = smooth_source_risk_vector(model, worlds, indices, delta)
        applied = self.applied_weight(step)
        objective = risk_vector.mean() + self.l2_weight * self.l2_norm(model) + applied * vrex_penalty_from_losses(risk_vector)
        return self.maybe_rescale(objective, applied)[0]
