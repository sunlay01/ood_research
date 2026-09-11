"""Deterministic source-only training with continuation-ready Adam state."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.data import BatchSchedule, ColoredEnvironment, scheduled_source_batches
from ..task3_cmnist_cpu_minimal.model import parameter_hash
from ..task3_cmnist_cpu_minimal.trainer import batch_schedule_hash
from .method_objectives import survey_method_objective


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


@dataclass(frozen=True)
class SurveyTrainResult:
    model: nn.Module
    optimizer_state: dict[str, Any]
    seed: int
    method: str
    initial_parameter_hash: str
    batch_schedule_hash: str
    final_parameter_hash: str
    optimizer_state_hash: str
    optimizer_reset_count: int
    objective_formula_id: str
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
    learning_rate = float(config["training"]["learning_rate"])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    reset_count = 0
    parts = None
    invalid_reason = "OK"
    for step in range(int(config["training"]["steps"])):
        if method == "VREX" and step == int(config["vrex"]["penalty_anneal_iters"]):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            reset_count += 1
        batches = scheduled_source_batches(source_envs, batch_schedule, step, config["device"])
        try:
            parts = survey_method_objective(method, model, batches, step=step, config=config)
            if not bool(torch.isfinite(parts.objective.detach())):
                invalid_reason = "NONFINITE_OBJECTIVE"
                break
            optimizer.zero_grad(set_to_none=True)
            parts.objective.backward()
            optimizer.step()
        except Exception as exc:
            invalid_reason = f"TRAINING_FAILED:{type(exc).__name__}:{exc}"
            break
    finite = invalid_reason == "OK" and parts is not None
    state = copy.deepcopy(optimizer.state_dict())
    formula = {
        "ERM": "CPU_MINIMAL_ERM_V1",
        "IRMv1": "CPU_MINIMAL_IRMV1_V1",
        "VREX": "CMNIST_VREX_ANNEALED_V1",
        "CORAL": "CMNIST_REPRESENTATION_CORAL_V1",
    }.get(method, "UNKNOWN")
    return SurveyTrainResult(
        model=model.cpu(), optimizer_state=state, seed=int(seed), method=method,
        initial_parameter_hash=initial_parameter_hash,
        batch_schedule_hash=batch_schedule_hash(batch_schedule),
        final_parameter_hash=parameter_hash(model),
        optimizer_state_hash=optimizer_state_hash(state),
        optimizer_reset_count=reset_count,
        objective_formula_id=formula,
        final_loss=float("nan") if parts is None else float(parts.objective.detach().cpu()),
        final_risk=float("nan") if parts is None else float(parts.risk.detach().cpu()),
        final_penalty=float("nan") if parts is None else float(parts.penalty.detach().cpu()),
        finite=finite, invalid_reason=invalid_reason,
    )
