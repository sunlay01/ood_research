import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.spectral_flatness_diagnostics import flatness_diagnostic_rows
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config


ROOT = Path(__file__).resolve().parents[1]


def test_flatness_diagnostics_emit_hvp_trace_and_sharpness_rows():
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    cfg = {**cfg, "diagnostics": {**cfg["diagnostics"], "hessian_power_iterations": 3, "hutchinson_probes": 2, "random_direction_probes": 2}}
    model = build_model_from_config(cfg)
    bank = (torch.rand(8, 392), torch.zeros(8, 1))
    rows = flatness_diagnostic_rows(model, bank, seed=10, method="ERM", variant="v", checkpoint=500, config=cfg)
    assert len(rows) == len(cfg["diagnostics"]["sharpness_radii"])
    assert {"source_loss", "gradient_norm", "hessian_top_eigenvalue", "hessian_trace_estimate", "sam_sharpness_delta"} <= set(rows[0])
    assert all(torch.isfinite(torch.tensor(row["gradient_norm"])) for row in rows)
