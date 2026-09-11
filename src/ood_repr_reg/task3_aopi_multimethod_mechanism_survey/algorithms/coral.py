"""CORAL algorithm definition for the multi-method CMNIST survey."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_forward_with_features, weight_norm_squared
from ..smooth_world5 import SmoothWorld5, environment_parameters, outcome_weight
from .base import BaseAlgorithm, smooth_source_risk_vector


def coral_penalty(features: tuple[Tensor, Tensor]) -> Tensor:
    """Mean and covariance alignment on unnormalized encoder representations."""
    if len(features) != 2 or features[0].ndim != 2 or features[1].ndim != 2:
        raise ValueError("CORAL requires two representation matrices")
    if features[0].shape[1] != features[1].shape[1]:
        raise ValueError("CORAL feature dimensions must match")
    if min(features[0].shape[0], features[1].shape[0]) < 2:
        raise ValueError("CORAL covariance requires at least two rows per environment")
    dimension = features[0].shape[1]
    means = tuple(value.mean(dim=0) for value in features)
    centered = tuple(value - mean for value, mean in zip(features, means))
    covariances = tuple(value.T @ value / (value.shape[0] - 1) for value in centered)
    mean_term = (means[0] - means[1]).square().sum() / dimension
    covariance_term = (covariances[0] - covariances[1]).square().sum() / (dimension**2)
    return mean_term + covariance_term


def smooth_coral_penalty(model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor) -> Tensor:
    representations = []
    for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
        p, q = environment_parameters(delta, environment=environment, evaluation=False)
        chunks = []
        weights = []
        for images, _, label_flip, color_flip in pool.outcome_batches(index):
            chunks.append(model.encode(images))
            weights.append(outcome_weight(p, q, label_flip, color_flip).expand(images.shape[0]))
        z = torch.cat(chunks)
        w = torch.cat(weights).to(z)
        w = w / w.sum()
        mean = (z * w[:, None]).sum(dim=0)
        centered = z - mean
        covariance = ((centered * w[:, None]).T @ centered) * (len(index) / (len(index) - 1))
        representations.append((mean, covariance))
    dimension = representations[0][0].numel()
    mean_term = (representations[0][0] - representations[1][0]).square().sum() / dimension
    covariance_term = (representations[0][1] - representations[1][1]).square().sum() / dimension**2
    return mean_term + covariance_term


class CORALAlgorithm(BaseAlgorithm):
    name = "CORAL"
    formula_id = "CMNIST_REPRESENTATION_CORAL_V1"

    @property
    def gamma(self) -> float:
        return float(self.config["coral"]["gamma"])

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
        penalty = coral_penalty(features)
        objective = risk + self.l2_weight * l2 + self.gamma * penalty
        return ObjectiveParts(objective, risk, l2, penalty, self.gamma, False)

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
        return risk + self.l2_weight * self.l2_norm(model) + self.gamma * smooth_coral_penalty(model, worlds, indices, delta)
