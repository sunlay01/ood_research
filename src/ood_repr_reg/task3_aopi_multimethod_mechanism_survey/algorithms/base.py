"""Shared algorithm interface and smooth source-risk helpers."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ...task3_cmnist_cpu_minimal.methods import ObjectiveParts
from ..smooth_world5 import SmoothPool, SmoothWorld5, environment_parameters, outcome_weight


def _hash_value(hasher: "hashlib._Hash", value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(str(tensor.dtype).encode())
        hasher.update(str(tuple(tensor.shape)).encode())
        hasher.update(tensor.numpy().tobytes())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            hasher.update(str(key).encode())
            _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _hash_value(hasher, item)
    else:
        hasher.update(repr(value).encode())


@dataclass(frozen=True)
class AlgorithmState:
    """Small algorithm-owned state payload cloned with optimizer/model state."""

    payload: dict[str, Any]

    def clone(self) -> "AlgorithmState":
        return AlgorithmState(copy.deepcopy(self.payload))

    def hash(self) -> str:
        hasher = hashlib.sha256()
        _hash_value(hasher, self.payload)
        return hasher.hexdigest()


@dataclass(frozen=True)
class StepResult:
    parts: ObjectiveParts
    optimizer: torch.optim.Optimizer
    algorithm_state: AlgorithmState
    did_reset_optimizer: bool


@dataclass(frozen=True)
class SmoothStepResult:
    objective: Tensor
    optimizer: torch.optim.Optimizer
    algorithm_state: AlgorithmState
    did_reset_optimizer: bool
    finite: bool


class SurveyAlgorithm(Protocol):
    name: str
    formula_id: str
    reference_id: str
    variant_id: str
    admission_role: str
    admits_to_pi: bool

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

    def initial_state(self, *, seed: int) -> AlgorithmState: ...

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
        learning_rate: float,
        algorithm_state: AlgorithmState,
    ) -> StepResult: ...

    def smooth_train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
        learning_rate: float,
        algorithm_state: AlgorithmState,
    ) -> SmoothStepResult: ...


class BaseAlgorithm:
    name = "BASE"
    formula_id = "BASE"
    reference_id = "LOCAL"
    variant_id = "BASE"
    admission_role = "PERFORMANCE_ONLY"
    admits_to_pi = True

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

    def initial_state(self, *, seed: int) -> AlgorithmState:
        return AlgorithmState({"algorithm": self.name, "seed": int(seed), "variant_id": self.variant_id})

    def l2_norm(self, model: nn.Module) -> Tensor:
        return sum(parameter.square().sum() for parameter in model.parameters())

    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
        *,
        step: int,
        learning_rate: float,
        algorithm_state: AlgorithmState,
    ) -> StepResult:
        optimizer, did_reset = self.prepare_step(optimizer, model, step=step, learning_rate=learning_rate)
        parts = self.objective(model, batches, step=step)
        optimizer.zero_grad(set_to_none=True)
        parts.objective.backward()
        optimizer.step()
        return StepResult(parts, optimizer, algorithm_state.clone(), did_reset)

    def smooth_train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        worlds: SmoothWorld5,
        indices: tuple[Tensor, Tensor],
        delta: Tensor,
        *,
        step: int,
        learning_rate: float,
        algorithm_state: AlgorithmState,
    ) -> SmoothStepResult:
        optimizer, did_reset = self.prepare_step(optimizer, model, step=step, learning_rate=learning_rate)
        objective = self.smooth_objective(model, worlds, indices, delta, step=step)
        finite = bool(torch.isfinite(objective.detach()))
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        return SmoothStepResult(objective, optimizer, algorithm_state.clone(), did_reset, finite)


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
