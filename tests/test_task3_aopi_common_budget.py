import json
from pathlib import Path

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_common_budget_keeps_outer_steps_and_marks_intrinsic_extra_compute():
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    assert cfg["training"]["steps"] == 501
    assert cfg["training"]["batch_size_per_environment"] == 512
    assert get_algorithm("ERM", cfg).forward_pass_equivalents_per_step == 1.0
    assert get_algorithm("SAM", cfg).forward_pass_equivalents_per_step == 2.0
    assert get_algorithm("ASAM", cfg).backward_pass_equivalents_per_step == 2.0
    assert get_algorithm("SVB_ORTHDNN", cfg).projection_or_svd_operations_per_step > 0
