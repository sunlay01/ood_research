"""Finite-time full-network response with faithful source-only continuations."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.model import parameter_hash
from .algorithms.registry import get_algorithm
from .functional_banks import FunctionalBanks, bank_logits
from .method_trainer import SurveyTrainResult, optimizer_state_hash
from .smooth_world5 import all_direction_vectors, SmoothWorld5


def _clone_state(result: SurveyTrainResult) -> tuple[nn.Module, torch.optim.Adam, Any]:
    model = copy.deepcopy(result.model).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(copy.deepcopy(result.optimizer_state))
    return model, optimizer, result.algorithm_state.clone()


def _run_path(result: SurveyTrainResult, worlds: SmoothWorld5, banks: FunctionalBanks, delta: Tensor, method: str, config: dict[str, Any], schedule: tuple[Tensor, Tensor], horizons: tuple[int, ...]) -> dict[int, tuple[str, str, str, dict[str, Tensor]]]:
    model, optimizer, algorithm_state = _clone_state(result)
    algorithm = get_algorithm(method, config)
    learning_rate = float(config["training"]["learning_rate"])
    snapshots = {0: (parameter_hash(model), optimizer_state_hash(optimizer.state_dict()), algorithm_state.hash(), bank_logits(model, banks))}
    with torch.enable_grad():
        for step in range(max(horizons)):
            indices = (schedule[0][step], schedule[1][step])
            step_result = algorithm.smooth_train_step(
                model,
                optimizer,
                worlds,
                indices,
                delta,
                step=int(config["training"]["steps"]) + step,
                learning_rate=learning_rate,
                algorithm_state=algorithm_state,
            )
            optimizer = step_result.optimizer
            algorithm_state = step_result.algorithm_state
            if not step_result.finite:
                raise ValueError(f"non-finite smooth continuation objective for {method}")
            if step + 1 in horizons:
                snapshots[step + 1] = (parameter_hash(model), optimizer_state_hash(optimizer.state_dict()), algorithm_state.hash(), bank_logits(model, banks))
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
    exact: dict[tuple[int, int], tuple[dict[str, Tensor], tuple[str, str, str, str, str, str, str, str, str], bool]] = {}
    for direction_index in (0, 1, 3):
        direction = torch.zeros(5, dtype=torch.double)
        direction[direction_index] = 1.0
        plus = _run_path(result, worlds, banks, float(config["response"]["delta"]) * direction, method, config, schedule, horizons)
        minus = _run_path(result, worlds, banks, -float(config["response"]["delta"]) * direction, method, config, schedule, horizons)
        replay = _run_path(result, worlds, banks, float(config["response"]["delta"]) * direction, method, config, schedule, horizons)
        for horizon in horizons:
            plus_hash, plus_optimizer_hash, plus_algorithm_hash, plus_logits = plus[horizon]
            minus_hash, minus_optimizer_hash, minus_algorithm_hash, minus_logits = minus[horizon]
            replay_hash, replay_optimizer_hash, replay_algorithm_hash, replay_logits = replay[horizon]
            response = {name: (plus_logits[name] - minus_logits[name]) / (2.0 * float(config["response"]["delta"])) for name in plus_logits}
            exact[(direction_index, horizon)] = (
                response,
                (plus_hash, minus_hash, plus_optimizer_hash, minus_optimizer_hash, replay_hash, replay_optimizer_hash, plus_algorithm_hash, minus_algorithm_hash, replay_algorithm_hash),
                bool(replay_hash == plus_hash and replay_optimizer_hash == plus_optimizer_hash and replay_algorithm_hash == plus_algorithm_hash and all(torch.equal(replay_logits[name], plus_logits[name]) for name in plus_logits)),
            )

    rows = []
    directions = all_direction_vectors()
    for direction_index, direction in enumerate(directions):
        for horizon in horizons:
            response = {}
            for bank_name in control[horizon][3]:
                response[bank_name] = sum(
                    float(direction[basis]) * exact[(basis, horizon)][0][bank_name]
                    for basis in (0, 1, 3)
                )
            exact_basis = direction_index in (0, 1, 3)
            hashes = exact[(direction_index, horizon)][1] if exact_basis else ("LINEAR_COMBINATION",) * 9
            replay_match = exact[(direction_index, horizon)][2] if exact_basis else all(exact[(basis, horizon)][2] for basis in (0, 1, 3) if float(direction[basis]) != 0.0)
            rows.append({
                "seed": int(seed), "method": method, "opaque_direction_id": f"u{direction_index:03d}", "basis_index": direction_index if direction_index < 5 else -1, "K": horizon,
                "source_bank_response_norm": float(response["source"].norm()), "counterfactual_bank_response_norm": float(torch.cat((response["counterfactual_red"], response["counterfactual_green"])).norm()), "clean_task_bank_response_norm": float(response["clean_task"].norm()),
                "control_parameter_hash": control[horizon][0], "plus_parameter_hash": hashes[0], "minus_parameter_hash": hashes[1],
                "plus_optimizer_hash": hashes[2], "minus_optimizer_hash": hashes[3], "replay_parameter_hash": hashes[4], "replay_optimizer_hash": hashes[5],
                "control_optimizer_hash": control[horizon][1], "control_algorithm_state_hash": control[horizon][2],
                "plus_algorithm_state_hash": hashes[6], "minus_algorithm_state_hash": hashes[7], "replay_algorithm_state_hash": hashes[8],
                "replay_match": replay_match,
                "finite": bool(all(torch.isfinite(value).all() for value in response.values())),
                "target_only_learner_update": False,
            })
    return rows
