import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.feature_nuclear import feature_nuclear_penalty
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.stable_rank import stable_rank
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.weight_nuclear import encoder_linear_weights, weight_nuclear_penalty
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config


ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))


def test_weight_nuclear_penalizes_encoder_linear_layers_only():
    model = build_model_from_config(config())
    weights = encoder_linear_weights(model)
    expected = sum(torch.linalg.svdvals(weight).sum() for weight in weights)
    assert torch.allclose(weight_nuclear_penalty(model), expected)
    assert all(weight.shape[0] == 64 for weight in weights)


def test_feature_nuclear_uses_positive_sum_of_singular_values():
    features = (torch.eye(4, requires_grad=True), torch.eye(4, requires_grad=True))
    penalty = feature_nuclear_penalty(features)
    assert penalty.item() > 0
    penalty.backward()
    assert features[0].grad is not None
    assert torch.isfinite(features[0].grad).all()


def test_stable_rank_formula_and_diagnostic_deferred_behavior():
    matrix = torch.eye(4)
    assert stable_rank(matrix) == 4.0
    assert config()["stable_rank"]["registered_as_algorithm"] is False
