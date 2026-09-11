import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.sam import SAMAlgorithm, make_sam_perturbations
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))


def test_sam_perturbation_uses_gradient_norm_and_train_step_restores_before_step():
    cfg = config()
    model = build_model_from_config(cfg)
    batches = ((torch.rand(4, 392), torch.zeros(4, 1)), (torch.rand(4, 392), torch.ones(4, 1)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    before = parameter_hash(model)
    algorithm = SAMAlgorithm(cfg)
    result = algorithm.train_step(model, optimizer, batches, step=0, learning_rate=0.0, algorithm_state=algorithm.initial_state(seed=1))
    assert parameter_hash(model) == before
    assert result.algorithm_state.payload["rho"] == cfg["sam"]["rho"]


def test_sam_perturbation_radius_is_nonzero_when_gradients_exist():
    model = torch.nn.Linear(3, 1)
    loss = model(torch.ones(2, 3)).sum()
    loss.backward()
    perturbations = make_sam_perturbations(model, rho=0.05)
    assert sum(float(delta.norm()) for _, delta in perturbations) > 0.0
