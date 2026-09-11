"""Weight nuclear-norm rank probe for the expanded CMNIST survey."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothWorld5
from .base import BaseAlgorithm, smooth_source_risk_vector


def encoder_linear_weights(model: nn.Module) -> tuple[Tensor, Tensor]:
    layers = [module for module in model.encoder if isinstance(module, nn.Linear)]
    if len(layers) != 2:
        raise ValueError("weight nuclear probe expects exactly two encoder linear layers")
    return layers[0].weight, layers[1].weight


def weight_nuclear_penalty(model: nn.Module) -> Tensor:
    return sum(torch.linalg.svdvals(weight).sum() for weight in encoder_linear_weights(model))


class WeightNuclearAlgorithm(BaseAlgorithm):
    name = "WEIGHT_NUCLEAR"
    formula_id = "CMNIST_WEIGHT_NUCLEAR_ENCODER_ONLY_V1"
    reference_id = "SOURCE_ONLY_ENCODER_WEIGHT_NUCLEAR_PROBE"
    variant_id = "WEIGHT_NUCLEAR_LAMBDA0P001_ENCODER_ONLY"
    admission_role = "RANK_PROBE"

    @property
    def penalty_weight(self) -> float:
        return float(self.config["weight_nuclear"]["lambda"])

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
        penalty = weight_nuclear_penalty(model)
        return ObjectiveParts(risk + self.l2_weight * l2 + self.penalty_weight * penalty, risk, l2, penalty, self.penalty_weight, False)

    def smooth_objective(
        self,
        model: nn.Module,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
    ) -> Tensor:
        risk = smooth_source_risk_vector(model, worlds, indices, delta).mean()
        return risk + self.l2_weight * self.l2_norm(model) + self.penalty_weight * weight_nuclear_penalty(model)
