import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.spectral_flatness_diagnostics import weight_spectrum_long_rows, representation_spectrum_rows, gradient_spectrum_rows
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_spectral_diagnostic_rows_have_required_rank_fields():
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    model = build_model_from_config(cfg)
    bank = (torch.rand(8, 392), torch.zeros(8, 1))
    rows = weight_spectrum_long_rows(model, seed=10, method="ERM", variant="v", checkpoint=0)
    assert {"singular_values", "spectral_norm", "stable_rank", "effective_rank", "numerical_rank_1e3"} <= set(rows[0])
    assert representation_spectrum_rows(model, bank, seed=10, method="ERM", variant="v", checkpoint=0)[0]["effective_rank"] > 0
    assert gradient_spectrum_rows(model, bank, seed=10, method="ERM", variant="v", checkpoint=0)[0]["stable_rank"] >= 0
