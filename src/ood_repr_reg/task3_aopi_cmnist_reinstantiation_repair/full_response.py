"""Full-network finite-time response from exact reconstructed Adam states."""

from __future__ import annotations

import copy
import hashlib
import io
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from ..task3_cmnist_cpu_minimal.data import build_task3_data, make_batch_schedule
from ..task3_cmnist_cpu_minimal.methods import method_objective, weight_norm_squared
from ..task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash
from .smooth_world import SmoothWorlds, environment_parameters, outcome_weight


DELTA = 0.01
STEPS = (1, 5, 20)


def _state_hash(value: object) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


@dataclass(frozen=True)
class ReconstructedState:
    model: torch.nn.Module
    optimizer_state: dict[str, Any]
    parameter_hash: str
    optimizer_hash: str


def reconstruct_adam_state(config: dict[str, Any], seed: int, method: str, expected_hash: str, *, data_root: str) -> ReconstructedState:
    data = build_task3_data(config, seed, data_root=data_root, download=False)
    torch.manual_seed(seed)
    model = build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    for step in range(int(config["training"]["steps"])):
        batches = tuple((env.images[index[step]], env.labels[index[step]]) for env, index in zip(data.source_envs, data.batch_schedule.indices))
        parts = method_objective(method, model, batches, step=step, config=config)
        optimizer.zero_grad(set_to_none=True)
        parts.objective.backward()
        optimizer.step()
    observed = parameter_hash(model)
    if observed != expected_hash:
        raise ValueError(f"reconstruction hash mismatch for seed={seed} method={method}")
    state = copy.deepcopy(optimizer.state_dict())
    return ReconstructedState(model=model.cpu(), optimizer_state=state, parameter_hash=observed, optimizer_hash=_state_hash(state))


def _batch_outcomes(pool, indices: Tensor):
    return pool.outcome_batches(indices)


def _expected_loss(model: torch.nn.Module, pool, indices: Tensor, p: Tensor, q: Tensor) -> Tensor:
    total = next(model.parameters()).new_zeros(())
    for images, labels, label_flip, color_flip in _batch_outcomes(pool, indices):
        loss = F.binary_cross_entropy_with_logits(model(images), labels.float())
        total = total + outcome_weight(p, q, label_flip, color_flip).to(loss.dtype) * loss
    return total


def smooth_method_objective(model: torch.nn.Module, worlds: SmoothWorlds, indices: tuple[Tensor, Tensor], theta: Tensor, method: str) -> Tensor:
    losses: list[Tensor] = []
    penalties: list[Tensor] = []
    for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
        p, q = environment_parameters(theta, environment=environment, evaluation=False)
        risk = _expected_loss(model, pool, index, p, q)
        losses.append(risk)
        scale = torch.ones((), requires_grad=True)
        scaled = _expected_loss_scaled(model, pool, index, p, q, scale)
        penalties.append(torch.autograd.grad(scaled, scale, create_graph=True)[0].square())
    risk = torch.stack(losses).mean()
    l2 = weight_norm_squared(model)
    if method == "ERM":
        return risk + 0.001 * l2
    return (risk + 0.001 * l2 + 10000.0 * torch.stack(penalties).mean()) / 10000.0


def _expected_loss_scaled(model: torch.nn.Module, pool, indices: Tensor, p: Tensor, q: Tensor, scale: Tensor) -> Tensor:
    total = next(model.parameters()).new_zeros(())
    for images, labels, label_flip, color_flip in _batch_outcomes(pool, indices):
        loss = F.binary_cross_entropy_with_logits(model(images) * scale, labels.float())
        total = total + outcome_weight(p, q, label_flip, color_flip).to(loss.dtype) * loss
    return total


def _continuation_schedule(worlds: SmoothWorlds, seed: int) -> tuple[Tensor, Tensor]:
    schedule = make_batch_schedule(source_pool_sizes=(len(worlds.source[0].digits), len(worlds.source[1].digits)), steps=max(STEPS), batch_size_per_environment=512, seed=seed + 271828)
    return schedule.indices


@torch.no_grad()
def _functional_logits(model: torch.nn.Module, worlds: SmoothWorlds) -> Tensor:
    images = worlds.source[0].outcome_batches(torch.arange(256))[0][0]
    return model(images).detach().cpu().double().reshape(-1)


def _clone_with_optimizer(state: ReconstructedState, learning_rate: float) -> tuple[torch.nn.Module, torch.optim.Adam]:
    model = copy.deepcopy(state.model).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(copy.deepcopy(state.optimizer_state))
    return model, optimizer


def _run_path(state: ReconstructedState, worlds: SmoothWorlds, theta: Tensor, method: str, seed: int, learning_rate: float) -> dict[int, tuple[Tensor, str, str]]:
    model, optimizer = _clone_with_optimizer(state, learning_rate)
    schedule = _continuation_schedule(worlds, seed)
    snapshots: dict[int, tuple[Tensor, str, str]] = {}
    for step in range(max(STEPS)):
        indices = (schedule[0][step], schedule[1][step])
        objective = smooth_method_objective(model, worlds, indices, theta, method)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
        checkpoint = step + 1
        if checkpoint in STEPS:
            snapshots[checkpoint] = (_functional_logits(model, worlds), parameter_hash(model), _state_hash(optimizer.state_dict()))
    return snapshots


def full_response_rows(config: dict[str, Any], worlds: SmoothWorlds, state: ReconstructedState, *, seed: int, method: str) -> list[dict[str, Any]]:
    theta = worlds.base_theta.detach().clone()
    control = _run_path(state, worlds, theta, method, seed, float(config["training"]["learning_rate"]))
    rows: list[dict[str, Any]] = []
    for direction_index, direction_name in enumerate(("source_env0_color", "source_env1_color", "shared_label_noise")):
        unit = torch.zeros(3, dtype=torch.double); unit[direction_index] = 1.0
        plus = _run_path(state, worlds, theta + DELTA * unit, method, seed, float(config["training"]["learning_rate"]))
        minus = _run_path(state, worlds, theta - DELTA * unit, method, seed, float(config["training"]["learning_rate"]))
        replay_plus = _run_path(state, worlds, theta + DELTA * unit, method, seed, float(config["training"]["learning_rate"]))
        for step in STEPS:
            central = (plus[step][0] - minus[step][0]) / (2.0 * DELTA)
            drift = control[step][0] - _functional_logits(state.model, worlds)
            rows.append({
                "seed": seed, "method": method, "tangent": direction_name, "K": step,
                "central_source_logit_response_norm": float(central.norm()), "control_source_logit_drift_norm": float(drift.norm()),
                "plus_parameter_hash": plus[step][1], "minus_parameter_hash": minus[step][1], "control_parameter_hash": control[step][1],
                "plus_optimizer_hash": plus[step][2], "minus_optimizer_hash": minus[step][2], "control_optimizer_hash": control[step][2],
                "replay_match": bool(plus[step][1:] == replay_plus[step][1:] and torch.equal(plus[step][0], replay_plus[step][0])),
                "finite": bool(torch.isfinite(central).all()),
            })
    return rows
