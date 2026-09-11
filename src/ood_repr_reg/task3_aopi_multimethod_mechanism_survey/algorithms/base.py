"""Shared algorithm interface and smooth source-risk helpers."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts
from ..smooth_world5 import SmoothPool, SmoothWorld5, environment_parameters, outcome_weight


class SurveyAlgorithm(Protocol):
    name: str
    formula_id: str

    def objective(
        self,
        model: nn.Module,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
    ) -> ObjectiveParts: ...

    def smooth_objective(
        self,
        model: nn.Module,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
    ) -> Tensor: ...

    def prepare_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        step: int,
        learning_rate: float,
    ) -> tuple[torch.optim.Optimizer, bool]: ...


class BaseAlgorithm:
    name = "BASE"
    formula_id = "BASE"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @property
    def l2_weight(self) -> float:
        return float(self.config["training"]["l2_regularizer_weight"])

    def prepare_step(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        *,
        step: int,
        learning_rate: float,
    ) -> tuple[torch.optim.Optimizer, bool]:
        return optimizer, False

    def l2_norm(self, model: nn.Module) -> Tensor:
        return sum(parameter.square().sum() for parameter in model.parameters())


def smooth_environment_risk(model: nn.Module, pool: SmoothPool, indices: Tensor, p: Tensor, q: Tensor) -> Tensor:
    value = next(model.parameters()).new_zeros(())
    for images, labels, label_flip, color_flip in pool.outcome_batches(indices):
        weight = outcome_weight(p, q, label_flip, color_flip).to(images.device)
        value = value + weight * F.binary_cross_entropy_with_logits(model(images), labels.float())
    return value


def smooth_source_risk_vector(model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor) -> Tensor:
    risks = []
    for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
        p, q = environment_parameters(delta, environment=environment, evaluation=False)
        risks.append(smooth_environment_risk(model, pool, index, p, q))
    return torch.stack(risks)
