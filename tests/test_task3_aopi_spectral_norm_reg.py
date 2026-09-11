import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.spectral_norm_reg import spectral_norm_regularizer
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_spectral_norm_reg_is_penalty_not_parametrization():
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    model = build_model_from_config(cfg)
    before_keys = set(model.state_dict())
    penalty = spectral_norm_regularizer(model)
    assert penalty.item() > 0.0
    assert set(model.state_dict()) == before_keys
    assert all("weight_u" not in key for key in model.state_dict())
