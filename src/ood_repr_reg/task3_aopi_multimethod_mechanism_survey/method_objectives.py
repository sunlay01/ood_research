"""Source-only V-REx/CORAL objectives and canonical ERM/IRM dispatch."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.methods import (
    ObjectiveParts,
    method_objective as canonical_method_objective,
    source_environment_losses,
    source_forward_with_features,
    weight_norm_squared,
)


def vrex_penalty_from_losses(losses: Tensor) -> Tensor:
    if losses.ndim != 1 or losses.numel() < 2:
        raise ValueError("V-REx requires a vector of at least two environment risks")
    return (losses - losses.mean()).square().mean()


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


def survey_method_objective(
    method: str,
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    step: int,
    config: dict[str, Any],
) -> ObjectiveParts:
    """Return the exact source-only objective for one survey training step."""
    if method in {"ERM", "IRMv1"}:
        return canonical_method_objective(method, model, batches, step=step, config=config)

    training = config["training"]
    l2 = weight_norm_squared(model)
    if method == "VREX":
        losses = torch.stack(source_environment_losses(model, batches))
        risk = losses.mean()
        penalty = vrex_penalty_from_losses(losses)
        settings = config["vrex"]
        applied = float(
            settings["post_anneal_penalty_weight"]
            if step >= int(settings["penalty_anneal_iters"])
            else settings["pre_anneal_penalty_weight"]
        )
        objective = risk + float(training["l2_regularizer_weight"]) * l2 + applied * penalty
        return ObjectiveParts(objective, risk, l2, penalty, applied, False)

    if method == "CORAL":
        losses, features, _ = source_forward_with_features(model, batches)
        risk = torch.stack(losses).mean()
        penalty = coral_penalty(features)
        applied = float(config["coral"]["gamma"])
        objective = risk + float(training["l2_regularizer_weight"]) * l2 + applied * penalty
        return ObjectiveParts(objective, risk, l2, penalty, applied, False)
    raise ValueError(f"unsupported survey method: {method}")
