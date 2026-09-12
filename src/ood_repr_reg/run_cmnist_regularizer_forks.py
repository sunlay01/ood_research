"""Matched regularizer-scale forks for source-only CMNIST response data.

Every branch starts from an identical model, Adam, and algorithm state at a
fixed checkpoint and consumes the same deterministic source batches.  The
intervention is deliberately modest: scale only the method's post-anneal
regularizer weight.  It produces controlled perturbation-response data, but
does not identify forcing and filtering separately.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from .task3_aopi_multimethod_mechanism_survey.functional_banks import bank_logits, build_functional_banks
from .task3_aopi_multimethod_mechanism_survey.method_trainer import optimizer_state_hash
from .task3_aopi_multimethod_mechanism_survey.smooth_world5 import build_smooth_world5
from .task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from .task3_cmnist_cpu_minimal.data import build_task3_data, scheduled_source_batches
from .task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json"
OUT = ROOT / "round3_redesign/method_agnostic_mechanism/regularizer_forks"
METHODS = ("IRMv1", "VREX", "FISHR")
SEEDS = (10, 11, 12, 13, 14)
SCALES = (0.0, 0.5, 1.0, 2.0)
CHECKPOINT = 300
HORIZONS = (1, 5, 20)


def _feature_vector(model: torch.nn.Module, banks) -> torch.Tensor:
    logits = bank_logits(model, banks)
    return torch.cat((logits["source"], logits["counterfactual_red"], logits["counterfactual_green"], logits["clean_task"])).detach().cpu().double()


def _config(method: str, scale: float) -> dict:
    raw = json.loads(CONFIG_PATH.read_text())
    raw["methods"] = [method]
    raw["candidate_methods"] = [method]
    config = copy.deepcopy(validate_config(raw))
    if method == "IRMv1":
        config["irmv1"]["penalty_weight"] *= scale
    elif method == "VREX":
        config["vrex"]["post_anneal_penalty_weight"] *= scale
    elif method == "FISHR":
        config["fishr"]["post_anneal_penalty_weight"] *= scale
    else:
        raise ValueError(method)
    return config


def _state_at_checkpoint(method: str, seed: int, checkpoint: int):
    config = _config(method, 1.0)
    data = build_task3_data(config, seed, data_root=ROOT / "data", download=bool(config["execution"]["download_mnist"]))
    world = build_smooth_world5(config, seed, data_root=ROOT / "data", download=False)
    banks = build_functional_banks(world, source_size_per_environment=int(config["banks"]["source_bank_size_per_environment"]), counterfactual_size=int(config["banks"]["counterfactual_bank_size"]))
    torch.manual_seed(seed)
    model = build_model_from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    algorithm = get_algorithm(method, config)
    algorithm_state = algorithm.initial_state(seed=seed)
    for step in range(checkpoint):
        batches = scheduled_source_batches(data.source_envs, data.batch_schedule, step, config["device"])
        result = algorithm.train_step(model, optimizer, batches, step=step, learning_rate=float(config["training"]["learning_rate"]), algorithm_state=algorithm_state)
        optimizer, algorithm_state = result.optimizer, result.algorithm_state
        if not torch.isfinite(result.parts.objective):
            raise RuntimeError(f"non-finite {method} state construction at step {step}")
    return config, data, banks, {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict()), "algorithm_state": algorithm_state.clone()}


def _rollout(method: str, config: dict, data, banks, state: dict, scale: float, checkpoint: int, horizons: tuple[int, ...]):
    branch_config = _config(method, scale)
    model = build_model_from_config(branch_config)
    model.load_state_dict(copy.deepcopy(state["model"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(branch_config["training"]["learning_rate"]))
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    algorithm = get_algorithm(method, branch_config)
    algorithm_state = state["algorithm_state"].clone()
    initial = _feature_vector(model, banks)
    snapshots = {}
    for offset in range(max(horizons)):
        step = checkpoint + offset
        batches = scheduled_source_batches(data.source_envs, data.batch_schedule, step, branch_config["device"])
        result = algorithm.train_step(model, optimizer, batches, step=step, learning_rate=float(branch_config["training"]["learning_rate"]), algorithm_state=algorithm_state)
        optimizer, algorithm_state = result.optimizer, result.algorithm_state
        if not torch.isfinite(result.parts.objective):
            raise RuntimeError(f"non-finite {method} scale={scale} fork at step {step}")
        if offset + 1 in horizons:
            snapshots[offset + 1] = (parameter_hash(model), optimizer_state_hash(optimizer.state_dict()), algorithm_state.hash(), _feature_vector(model, banks) - initial)
    return snapshots


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, vector_rows, vectors = [], [], []
    for method in METHODS:
        for seed in SEEDS:
            config, data, banks, state = _state_at_checkpoint(method, seed, CHECKPOINT)
            control_a = _rollout(method, config, data, banks, state, 1.0, CHECKPOINT, HORIZONS)
            control_b = _rollout(method, config, data, banks, state, 1.0, CHECKPOINT, HORIZONS)
            for horizon in HORIZONS:
                a, b = control_a[horizon], control_b[horizon]
                if a[:3] != b[:3] or not torch.equal(a[3], b[3]):
                    raise AssertionError("control replay mismatch")
            for scale in SCALES:
                branch = control_a if scale == 1.0 else _rollout(method, config, data, banks, state, scale, CHECKPOINT, HORIZONS)
                for horizon, (model_hash, optimizer_hash, algorithm_hash, response) in branch.items():
                    relative = response - control_a[horizon][3]
                    n_source = len(banks.source)
                    n_counterfactual = len(banks.counterfactual_red) + len(banks.counterfactual_green)
                    rows.append({"method": method, "seed": seed, "checkpoint": CHECKPOINT, "horizon": horizon, "regularizer_scale": scale, "functional_response_norm": float(response.norm()), "effect_vs_control_norm": float(relative.norm()), "source_effect_norm": float(relative[:n_source].norm()), "counterfactual_effect_norm": float(relative[n_source:n_source + n_counterfactual].norm()), "clean_effect_norm": float(relative[n_source + n_counterfactual:].norm()), "control_replay_exact": True, "control_parameter_hash": control_a[horizon][0], "control_optimizer_hash": control_a[horizon][1], "control_algorithm_state_hash": control_a[horizon][2]})
                    vector_rows.append({"method": method, "seed": seed, "checkpoint": CHECKPOINT, "horizon": horizon, "regularizer_scale": scale})
                    vectors.append(relative.numpy())
    pd.DataFrame(rows).to_csv(OUT / "fork_response_summary.csv", index=False)
    pd.DataFrame(vector_rows).to_csv(OUT / "fork_response_vector_keys.csv", index=False)
    np.save(OUT / "fork_response_vectors.npy", np.asarray(vectors))
    provenance = {"dataset": "ColoredMNIST", "methods": list(METHODS), "seeds": list(SEEDS), "checkpoint": CHECKPOINT, "horizons": list(HORIZONS), "regularizer_scales": list(SCALES), "same_model_optimizer_algorithm_state": True, "same_source_batch_schedule": True, "control_replay_exact": True, "target_used": False, "intervention": "post-anneal regularizer scale", "interpretation_boundary": "controlled regularizer intervention; does not isolate forcing C from filtering K"}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    (OUT / "report.md").write_text("# Matched CMNIST regularizer forks\n\nEach branch starts from the same step-300 model, Adam state, algorithm state, and source batch schedule. The response is a high-dimensional functional displacement relative to a control replay. Target outcomes are not read. This intervention changes overall regularizer strength and therefore cannot separately identify forcing and filtering.\n")


if __name__ == "__main__":
    run()
