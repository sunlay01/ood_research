import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.asam import ASAMAlgorithm
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.sam import make_sam_perturbations
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_asam_perturbation_differs_from_sam_under_parameter_scale():
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    model = torch.nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[10.0, 0.1]]))
        model.bias.zero_()
    model(torch.ones(4, 2)).sum().backward()
    sam = torch.cat([delta.reshape(-1) for _, delta in make_sam_perturbations(model, rho=0.05)])
    asam = torch.cat([delta.reshape(-1) for _, delta in make_sam_perturbations(model, rho=0.5, adaptive=True, eta=0.01)])
    assert not torch.allclose(sam / sam.norm(), asam / asam.norm())
    assert ASAMAlgorithm(cfg).adaptive is True
