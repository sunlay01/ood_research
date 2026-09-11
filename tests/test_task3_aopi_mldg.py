import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.mldg import MLDGAlgorithm
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[1]


def config(beta=1.0):
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    cfg = {**cfg, "mldg": {**cfg["mldg"], "beta": beta}}
    return cfg


def batches():
    x0 = torch.rand(6, 392)
    y0 = (torch.arange(6) % 2).float()[:, None]
    x1 = torch.rand(6, 392)
    y1 = ((torch.arange(6) + 1) % 2).float()[:, None]
    return ((x0, y0), (x1, y1))


def test_mldg_roles_alternate_deterministically():
    algorithm = MLDGAlgorithm(config())
    assert algorithm._roles(0) == (0, 1)
    assert algorithm._roles(1) == (1, 0)


def test_mldg_first_order_step_updates_parameters_and_state_without_target():
    cfg = config()
    model = build_model_from_config(cfg)
    before = parameter_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    algorithm = MLDGAlgorithm(cfg)
    state = algorithm.initial_state(seed=10)
    result = algorithm.train_step(model, optimizer, batches(), step=0, learning_rate=0.001, algorithm_state=state)
    assert parameter_hash(model) != before
    assert result.algorithm_state.payload["last_meta_train_env"] == 0
    assert result.algorithm_state.payload["last_meta_test_env"] == 1
    assert result.parts.rescaled_after_anneal is False


def test_mldg_beta_zero_removes_meta_test_gradient_contribution():
    cfg = config(beta=0.0)
    model = build_model_from_config(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    algorithm = MLDGAlgorithm(cfg)
    result = algorithm.train_step(model, optimizer, batches(), step=0, learning_rate=0.001, algorithm_state=algorithm.initial_state(seed=1))
    assert result.parts.applied_penalty_weight == 0.0
