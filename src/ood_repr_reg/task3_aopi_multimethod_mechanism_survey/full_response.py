"""Finite-time full-network response with faithful source-only continuations."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..task3_cmnist_cpu_minimal.model import parameter_hash
from .functional_banks import FunctionalBanks, bank_logits
from .method_objectives import vrex_penalty_from_losses
from .method_trainer import SurveyTrainResult, optimizer_state_hash
from .smooth_world5 import all_direction_vectors, SmoothPool, SmoothWorld5, environment_parameters, outcome_weight


def _smooth_environment_risk(model: nn.Module, pool: SmoothPool, indices: Tensor, p: Tensor, q: Tensor) -> Tensor:
    value = next(model.parameters()).new_zeros(())
    for images, labels, label_flip, color_flip in pool.outcome_batches(indices):
        value = value + outcome_weight(p, q, label_flip, color_flip).to(images.device) * F.binary_cross_entropy_with_logits(model(images), labels.float())
    return value


def _smooth_coral(model: nn.Module, pools: tuple[SmoothPool, SmoothPool], indices: tuple[Tensor, Tensor], delta: Tensor) -> Tensor:
    representations = []
    for environment, (pool, index) in enumerate(zip(pools, indices)):
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
    return (representations[0][0] - representations[1][0]).square().sum() / dimension + (representations[0][1] - representations[1][1]).square().sum() / dimension**2


def smooth_source_objective(model: nn.Module, worlds: SmoothWorld5, indices: tuple[Tensor, Tensor], delta: Tensor, method: str, config: dict[str, Any], step: int) -> Tensor:
    risks = []
    for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
        p, q = environment_parameters(delta, environment=environment, evaluation=False)
        risks.append(_smooth_environment_risk(model, pool, index, p, q))
    risk_vector = torch.stack(risks)
    risk = risk_vector.mean()
    l2 = sum(parameter.square().sum() for parameter in model.parameters())
    weight = float(config["training"]["l2_regularizer_weight"])
    if method == "ERM":
        return risk + weight * l2
    if method == "IRMv1":
        penalties = []
        for environment, (pool, index) in enumerate(zip(worlds.source, indices)):
            p, q = environment_parameters(delta, environment=environment, evaluation=False)
            scale = torch.ones((), requires_grad=True)
            weighted = []
            for images, labels, label_flip, color_flip in pool.outcome_batches(index):
                loss = F.binary_cross_entropy_with_logits(model(images) * scale, labels.float())
                weighted.append(outcome_weight(p, q, label_flip, color_flip).to(loss) * loss)
            penalty_gradient = torch.autograd.grad(torch.stack(weighted).sum(), scale, create_graph=True)[0]
            penalties.append(penalty_gradient.square())
        applied = float(config["irmv1"]["penalty_weight"] if step >= int(config["irmv1"]["penalty_anneal_iters"]) else 1.0)
        objective = risk + weight * l2 + applied * torch.stack(penalties).mean()
        return objective / applied if applied > 1.0 else objective
    if method == "VREX":
        applied = float(config["vrex"]["post_anneal_penalty_weight"] if step >= int(config["vrex"]["penalty_anneal_iters"]) else config["vrex"]["pre_anneal_penalty_weight"])
        return risk + weight * l2 + applied * vrex_penalty_from_losses(risk_vector)
    if method == "CORAL":
        return risk + weight * l2 + float(config["coral"]["gamma"]) * _smooth_coral(model, worlds.source, indices, delta)
    raise ValueError(f"unknown continuation method: {method}")


def _clone_state(result: SurveyTrainResult) -> tuple[nn.Module, torch.optim.Adam]:
    model = copy.deepcopy(result.model).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(copy.deepcopy(result.optimizer_state))
    return model, optimizer


def _run_path(result: SurveyTrainResult, worlds: SmoothWorld5, banks: FunctionalBanks, delta: Tensor, method: str, config: dict[str, Any], schedule: tuple[Tensor, Tensor], horizons: tuple[int, ...]) -> dict[int, tuple[str, str, dict[str, Tensor]]]:
    model, optimizer = _clone_state(result)
    snapshots = {0: (parameter_hash(model), optimizer_state_hash(optimizer.state_dict()), bank_logits(model, banks))}
    with torch.enable_grad():
        for step in range(max(horizons)):
            indices = (schedule[0][step], schedule[1][step])
            objective = smooth_source_objective(model, worlds, indices, delta, method, config, int(config["training"]["steps"]) + step)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            if step + 1 in horizons:
                snapshots[step + 1] = (parameter_hash(model), optimizer_state_hash(optimizer.state_dict()), bank_logits(model, banks))
    return snapshots


def full_response_rows(result: SurveyTrainResult, worlds: SmoothWorld5, banks: FunctionalBanks, *, config: dict[str, Any], seed: int, method: str) -> list[dict[str, Any]]:
    max_horizon = max(int(value) for value in config["response"]["horizons"])
    generator = torch.Generator().manual_seed(int(seed) + 271828)
    schedule = (
        torch.randint(len(worlds.source[0].digits), (max_horizon, int(config["training"]["batch_size_per_environment"])), generator=generator),
        torch.randint(len(worlds.source[1].digits), (max_horizon, int(config["training"]["batch_size_per_environment"])), generator=generator),
    )
    zero = torch.zeros(5, dtype=torch.double)
    horizons = tuple(int(value) for value in config["response"]["horizons"])
    control = _run_path(result, worlds, banks, zero, method, config, schedule, horizons)
    exact: dict[tuple[int, int], tuple[dict[str, Tensor], tuple[str, str, str, str, str, str], bool]] = {}
    for direction_index in (0, 1, 3):
        direction = torch.zeros(5, dtype=torch.double)
        direction[direction_index] = 1.0
        plus = _run_path(result, worlds, banks, float(config["response"]["delta"]) * direction, method, config, schedule, horizons)
        minus = _run_path(result, worlds, banks, -float(config["response"]["delta"]) * direction, method, config, schedule, horizons)
        replay = _run_path(result, worlds, banks, float(config["response"]["delta"]) * direction, method, config, schedule, horizons)
        for horizon in horizons:
            plus_hash, plus_optimizer_hash, plus_logits = plus[horizon]
            minus_hash, minus_optimizer_hash, minus_logits = minus[horizon]
            replay_hash, replay_optimizer_hash, replay_logits = replay[horizon]
            response = {name: (plus_logits[name] - minus_logits[name]) / (2.0 * float(config["response"]["delta"])) for name in plus_logits}
            exact[(direction_index, horizon)] = (
                response,
                (plus_hash, minus_hash, plus_optimizer_hash, minus_optimizer_hash, replay_hash, replay_optimizer_hash),
                bool(replay_hash == plus_hash and replay_optimizer_hash == plus_optimizer_hash and all(torch.equal(replay_logits[name], plus_logits[name]) for name in plus_logits)),
            )

    rows = []
    directions = all_direction_vectors()
    for direction_index, direction in enumerate(directions):
        for horizon in horizons:
            response = {}
            for bank_name in control[horizon][2]:
                response[bank_name] = sum(
                    float(direction[basis]) * exact[(basis, horizon)][0][bank_name]
                    for basis in (0, 1, 3)
                )
            exact_basis = direction_index in (0, 1, 3)
            hashes = exact[(direction_index, horizon)][1] if exact_basis else ("LINEAR_COMBINATION",) * 6
            replay_match = exact[(direction_index, horizon)][2] if exact_basis else all(exact[(basis, horizon)][2] for basis in (0, 1, 3) if float(direction[basis]) != 0.0)
            rows.append({
                "seed": int(seed), "method": method, "opaque_direction_id": f"u{direction_index:03d}", "basis_index": direction_index if direction_index < 5 else -1, "K": horizon,
                "source_bank_response_norm": float(response["source"].norm()), "counterfactual_bank_response_norm": float(torch.cat((response["counterfactual_red"], response["counterfactual_green"])).norm()), "clean_task_bank_response_norm": float(response["clean_task"].norm()),
                "control_parameter_hash": control[horizon][0], "plus_parameter_hash": hashes[0], "minus_parameter_hash": hashes[1],
                "plus_optimizer_hash": hashes[2], "minus_optimizer_hash": hashes[3], "replay_parameter_hash": hashes[4], "replay_optimizer_hash": hashes[5],
                "replay_match": replay_match,
                "finite": bool(all(torch.isfinite(value).all() for value in response.values())),
                "target_only_learner_update": False,
            })
    return rows
