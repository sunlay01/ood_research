"""Training loop for the CPU-minimal ColoredMNIST probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import copy
import hashlib

import torch
from torch import Tensor, nn

from .data import BatchSchedule, ColoredEnvironment, scheduled_source_batches
from .methods import Calibration, calibrate_response_scale, method_objective


@dataclass(frozen=True)
class CheckpointState:
    step: int
    state_dict: dict[str, Tensor]
    loss: float
    risk: float
    penalty: float


@dataclass(frozen=True)
class TrainResult:
    model: nn.Module
    method: str
    seed: int
    initial_parameter_hash: str
    batch_schedule_hash: str
    calibration: Calibration | None
    checkpoints: tuple[CheckpointState, ...]
    finite: bool
    invalid_reason: str
    optimizer_recreated_each_step: bool


def batch_schedule_hash(schedule: BatchSchedule) -> str:
    hasher = hashlib.sha256()
    for indices in schedule.indices:
        hasher.update(indices.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def clone_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_method(
    *,
    model: nn.Module,
    source_envs: tuple[ColoredEnvironment, ColoredEnvironment],
    batch_schedule: BatchSchedule,
    method: str,
    config: dict[str, Any],
    seed: int,
    initial_parameter_hash: str,
) -> TrainResult:
    training_cfg = config["training"]
    response_cfg = config["response"]
    steps = int(training_cfg["steps"])
    checkpoint_steps = set(int(step) for step in training_cfg["checkpoint_steps"])
    device = torch.device(config["device"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))
    schedule_hash = batch_schedule_hash(batch_schedule)
    calibration: Calibration | None = None
    checkpoints: list[CheckpointState] = []

    if method in {"GRAD", "LOCAL_RESPONSE"}:
        first_batches = scheduled_source_batches(source_envs, batch_schedule, 0, device)
        calibration = calibrate_response_scale(
            model,
            first_batches,
            method=method,
            l2_regularizer_weight=float(training_cfg["l2_regularizer_weight"]),
            damping_epsilon=float(response_cfg["damping_epsilon"]),
            calibration_update_ratio=float(response_cfg["calibration_update_ratio"]),
        )
        if not calibration.valid:
            return TrainResult(
                model=model,
                method=method,
                seed=int(seed),
                initial_parameter_hash=initial_parameter_hash,
                batch_schedule_hash=schedule_hash,
                calibration=calibration,
                checkpoints=tuple(checkpoints),
                finite=False,
                invalid_reason=calibration.reason,
                optimizer_recreated_each_step=False,
            )

    if 0 in checkpoint_steps:
        batches = scheduled_source_batches(source_envs, batch_schedule, 0, device)
        try:
            parts = method_objective(method, model, batches, step=0, config=config, calibration=calibration)
            checkpoints.append(
                CheckpointState(0, clone_state_dict(model), float(parts.objective.detach().cpu()), float(parts.risk.detach().cpu()), float(parts.penalty.detach().cpu()))
            )
        except Exception as exc:
            return TrainResult(
                model=model,
                method=method,
                seed=int(seed),
                initial_parameter_hash=initial_parameter_hash,
                batch_schedule_hash=schedule_hash,
                calibration=calibration,
                checkpoints=tuple(checkpoints),
                finite=False,
                invalid_reason=f"INITIAL_OBJECTIVE_FAILED:{type(exc).__name__}:{exc}",
                optimizer_recreated_each_step=False,
            )

    for step in range(steps):
        batches = scheduled_source_batches(source_envs, batch_schedule, step, device)
        try:
            parts = method_objective(method, model, batches, step=step, config=config, calibration=calibration)
            if not torch.isfinite(parts.objective.detach()):
                return TrainResult(
                    model=model,
                    method=method,
                    seed=int(seed),
                    initial_parameter_hash=initial_parameter_hash,
                    batch_schedule_hash=schedule_hash,
                    calibration=calibration,
                    checkpoints=tuple(checkpoints),
                    finite=False,
                    invalid_reason="NONFINITE_LOSS",
                    optimizer_recreated_each_step=False,
                )
            optimizer.zero_grad(set_to_none=True)
            parts.objective.backward()
            optimizer.step()
        except Exception as exc:
            return TrainResult(
                model=model,
                method=method,
                seed=int(seed),
                initial_parameter_hash=initial_parameter_hash,
                batch_schedule_hash=schedule_hash,
                calibration=calibration,
                checkpoints=tuple(checkpoints),
                finite=False,
                invalid_reason=f"TRAINING_FAILED:{type(exc).__name__}:{exc}",
                optimizer_recreated_each_step=False,
            )
        if step in checkpoint_steps and step != 0:
            checkpoints.append(
                CheckpointState(
                    step,
                    copy.deepcopy(clone_state_dict(model)),
                    float(parts.objective.detach().cpu()),
                    float(parts.risk.detach().cpu()),
                    float(parts.penalty.detach().cpu()),
                )
            )

    return TrainResult(
        model=model,
        method=method,
        seed=int(seed),
        initial_parameter_hash=initial_parameter_hash,
        batch_schedule_hash=schedule_hash,
        calibration=calibration,
        checkpoints=tuple(checkpoints),
        finite=True,
        invalid_reason="OK",
        optimizer_recreated_each_step=False,
    )
