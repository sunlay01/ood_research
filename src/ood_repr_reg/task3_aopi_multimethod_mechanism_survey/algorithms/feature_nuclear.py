"""Feature nuclear-norm rank probe for the expanded CMNIST survey."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, source_forward_with_features, weight_norm_squared
from ..smooth_world5 import SmoothWorld5, environment_parameters, outcome_weight
from .base import BaseAlgorithm, smooth_source_risk_vector


def feature_nuclear_penalty(features: tuple[Tensor, Tensor]) -> Tensor:
    matrix = torch.cat(features, dim=0)
    return torch.linalg.svdvals(matrix).sum()


def smooth_feature_nuclear_penalty(model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor) -> Tensor:
    chunks = []
    for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
        p, q = environment_parameters(delta, environment=environment, evaluation=False)
        for images, _, label_flip, color_flip in pool.outcome_batches(index):
            weight = outcome_weight(p, q, label_flip, color_flip).to(images)
            chunks.append(model.encode(images) * weight.sqrt())
    return torch.linalg.svdvals(torch.cat(chunks, dim=0)).sum()


class FeatureNuclearAlgorithm(BaseAlgorithm):
    name = "FEATURE_NUCLEAR"
    formula_id = "CMNIST_FEATURE_NUCLEAR_SOURCE_FEATURES_V1"
    reference_id = "DNNR_FEATURE_NUCLEAR_LOSS_PLUS_LAMBDA_SUM_SINGULAR_VALUES"
    variant_id = "FEATURE_NUCLEAR_LAMBDA0P0001_SOURCE_ONLY"
    admission_role = "RANK_PROBE"

    @property
    def penalty_weight(self) -> float:
        return float(self.config["feature_nuclear"]["lambda"])

    def objective(
        self,
        model: nn.Module,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
    ) -> ObjectiveParts:
        losses, features, _ = source_forward_with_features(model, batches)
        risk = torch.stack(losses).mean()
        l2 = weight_norm_squared(model)
        penalty = feature_nuclear_penalty(features)
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
        return risk + self.l2_weight * self.l2_norm(model) + self.penalty_weight * smooth_feature_nuclear_penalty(model, worlds, indices, delta)
