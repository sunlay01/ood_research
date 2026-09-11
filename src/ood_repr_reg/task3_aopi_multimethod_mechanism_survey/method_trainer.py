"""Deterministic source-only training with continuation-ready Adam state."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.data import BatchSchedule, ColoredEnvironment, scheduled_source_batches
from ..task3_cmnist_cpu_minimal.model import parameter_hash
from ..task3_cmnist_cpu_minimal.trainer import batch_schedule_hash
from .algorithms.base import AlgorithmState
from .algorithms.registry import get_algorithm


def _hash_value(hasher: Any, value: Any) -> None:
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


def optimizer_state_hash(state: dict[str, Any]) -> str:
    hasher = hashlib.sha256()
    _hash_value(hasher, state)
    return hasher.hexdigest()


def clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@dataclass(frozen=True)
class SurveyTrainResult:
    model: nn.Module
    optimizer_state: dict[str, Any]
    algorithm_state: AlgorithmState
    checkpoint_state_dicts: dict[int, dict[str, Tensor]]
    seed: int
    method: str
    initial_parameter_hash: str
    batch_schedule_hash: str
    final_parameter_hash: str
    optimizer_state_hash: str
    algorithm_state_hash: str
    optimizer_reset_count: int
    objective_formula_id: str
    algorithm_reference_id: str
    algorithm_variant_id: str
    admission_role: str
    admitted_to_pi: bool
    admits_to_training: bool
    deferred_reason: str
    forward_pass_equivalents_per_step: float
    backward_pass_equivalents_per_step: float
    projection_or_svd_operations_per_step: float
    training_wall_clock_seconds: float
    final_loss: float
    final_risk: float
    final_penalty: float
    finite: bool
    invalid_reason: str


def train_survey_method(
    *,
    model: nn.Module,
    source_envs: tuple[ColoredEnvironment, ColoredEnvironment],
    batch_schedule: BatchSchedule,
    method: str,
    config: dict[str, Any],
    seed: int,
    initial_parameter_hash: str,
) -> SurveyTrainResult:
    """Train one learner from the two supplied source environments only."""
    started = time.time()
    learning_rate = float(config["training"]["learning_rate"])
    algorithm = get_algorithm(method, config)
    if not algorithm.admits_to_training:
        raise ValueError(f"{method} is not admitted to training: {algorithm.deferred_reason}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    algorithm_state = algorithm.initial_state(seed=seed)
    reset_count = 0
    parts = None
    invalid_reason = "OK"
    checkpoints: dict[int, dict[str, Tensor]] = {}
    checkpoint_steps = set(int(step) for step in config["training"].get("checkpoint_steps", []))
    if 0 in checkpoint_steps:
        checkpoints[0] = clone_state_dict(model)
    for step in range(int(config["training"]["steps"])):
        batches = scheduled_source_batches(source_envs, batch_schedule, step, config["device"])
        try:
            step_result = algorithm.train_step(
                model,
                optimizer,
                batches,
                step=step,
                learning_rate=learning_rate,
                algorithm_state=algorithm_state,
            )
            parts = step_result.parts
            optimizer = step_result.optimizer
            algorithm_state = step_result.algorithm_state
            reset_count += int(step_result.did_reset_optimizer)
            if not bool(torch.isfinite(parts.objective.detach())):
                invalid_reason = "NONFINITE_OBJECTIVE"
                break
            if step in checkpoint_steps and step != 0:
                checkpoints[int(step)] = clone_state_dict(model)
        except Exception as exc:
            invalid_reason = f"TRAINING_FAILED:{type(exc).__name__}:{exc}"
            break
    finite = invalid_reason == "OK" and parts is not None
    state = copy.deepcopy(optimizer.state_dict())
    return SurveyTrainResult(
        model=model.cpu(), optimizer_state=state, seed=int(seed), method=method,
        algorithm_state=algorithm_state.clone(),
        checkpoint_state_dicts=checkpoints,
        initial_parameter_hash=initial_parameter_hash,
        batch_schedule_hash=batch_schedule_hash(batch_schedule),
        final_parameter_hash=parameter_hash(model),
        optimizer_state_hash=optimizer_state_hash(state),
        algorithm_state_hash=algorithm_state.hash(),
        optimizer_reset_count=reset_count,
        objective_formula_id=algorithm.formula_id,
        algorithm_reference_id=algorithm.reference_id,
        algorithm_variant_id=str(config.get("_active_variant_id", algorithm.variant_id)),
        admission_role=algorithm.admission_role,
        admitted_to_pi=bool(algorithm.admits_to_pi),
        admits_to_training=bool(algorithm.admits_to_training),
        deferred_reason=algorithm.deferred_reason,
        forward_pass_equivalents_per_step=float(algorithm.forward_pass_equivalents_per_step),
        backward_pass_equivalents_per_step=float(algorithm.backward_pass_equivalents_per_step),
        projection_or_svd_operations_per_step=float(algorithm.projection_or_svd_operations_per_step),
        training_wall_clock_seconds=float(time.time() - started),
        final_loss=float("nan") if parts is None else float(parts.objective.detach().cpu()),
        final_risk=float("nan") if parts is None else float(parts.risk.detach().cpu()),
        final_penalty=float("nan") if parts is None else float(parts.penalty.detach().cpu()),
        finite=finite, invalid_reason=invalid_reason,
    )
