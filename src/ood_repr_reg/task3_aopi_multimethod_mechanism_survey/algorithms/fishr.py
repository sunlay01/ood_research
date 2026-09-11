"""Fishr algorithm definition for the expanded CMNIST survey."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts, source_environment_losses, weight_norm_squared
from ..smooth_world5 import SmoothPool, SmoothWorld5, environment_parameters, outcome_weight
from .base import BaseAlgorithm, smooth_source_risk_vector


def classifier_gradient_matrix(model: nn.Module, images: Tensor, labels: Tensor) -> Tensor:
    """Per-example gradient of BCE loss with respect to final classifier weight+bias."""
    features = model.encode(images)
    logits = model.head(features)
    residual = torch.sigmoid(logits) - labels.float()
    return torch.cat((residual * features, residual), dim=1)


def centered_diagonal_variance(gradients: Tensor) -> Tensor:
    centered = gradients - gradients.mean(dim=0, keepdim=True)
    return centered.square().mean(dim=0)


def fishr_penalty_from_variances(variances: tuple[Tensor, Tensor]) -> Tensor:
    stacked = torch.stack(variances)
    average = stacked.mean(dim=0, keepdim=True)
    return (stacked - average).square().sum()


def fishr_penalty(model: nn.Module, batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]) -> Tensor:
    variances = tuple(centered_diagonal_variance(classifier_gradient_matrix(model, x, y)) for x, y in batches)
    return fishr_penalty_from_variances((variances[0], variances[1]))


def smooth_fishr_variance(model: nn.Module, pool: SmoothPool, indices: Tensor, p: Tensor, q: Tensor) -> Tensor:
    gradients: list[Tensor] = []
    weights: list[Tensor] = []
    for images, labels, label_flip, color_flip in pool.outcome_batches(indices):
        gradients.append(classifier_gradient_matrix(model, images, labels))
        weights.append(outcome_weight(p, q, label_flip, color_flip).to(gradients[-1]))
    mean = sum(weight * grad.mean(dim=0) for grad, weight in zip(gradients, weights))
    return sum(weight * (grad - mean).square().mean(dim=0) for grad, weight in zip(gradients, weights))


def smooth_fishr_penalty(model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor) -> Tensor:
    variances = []
    for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
        p, q = environment_parameters(delta, environment=environment, evaluation=False)
        variances.append(smooth_fishr_variance(model, pool, index, p, q))
    return fishr_penalty_from_variances((variances[0], variances[1]))


class FishrAlgorithm(BaseAlgorithm):
    name = "FISHR"
    formula_id = "CMNIST_FISHR_CLASSIFIER_GRAD_VARIANCE_V1"
    reference_id = "FISHR_COLOREDMNIST_CLASSIFIER_BATCH_GRAD"
    variant_id = "FISHR_COMMON_HARNESS_LAMBDA10000_ANNEAL100"
    admission_role = "INTERMEDIATE"

    def prepare_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        step: int,
        learning_rate: float,
    ) -> tuple[torch.optim.Optimizer, bool]:
        if bool(self.config["fishr"].get("reset_adam_at_anneal", True)) and step == int(self.config["fishr"]["penalty_anneal_iters"]):
            return torch.optim.Adam(model.parameters(), lr=learning_rate), True
        return optimizer, False

    def applied_weight(self, step: int) -> float:
        settings = self.config["fishr"]
        return float(
            settings["post_anneal_penalty_weight"]
            if step >= int(settings["penalty_anneal_iters"])
            else settings["pre_anneal_penalty_weight"]
        )

    def maybe_rescale(self, objective: Tensor, applied: float) -> tuple[Tensor, bool]:
        rescaled = bool(self.config["fishr"].get("whole_loss_rescale_after_anneal") and applied > 1.0)
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
        penalty = fishr_penalty(model, batches)
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
        objective = risk_vector.mean() + self.l2_weight * self.l2_norm(model) + applied * smooth_fishr_penalty(model, worlds, indices, delta)
        return self.maybe_rescale(objective, applied)[0]
