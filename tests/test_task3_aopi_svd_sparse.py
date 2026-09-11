import json
from pathlib import Path

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.registry import get_algorithm
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config


ROOT = Path(__file__).resolve().parents[1]


def test_svd_sparse_is_deferred_not_nuclear_norm_substitute():
    cfg = validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))
    algorithm = get_algorithm("SVD_SPARSE", cfg)
    assert algorithm.admits_to_training is False
    assert algorithm.admits_to_pi is False
    assert "SVD_SPARSE_DEFERRED" in algorithm.deferred_reason
