import ast
import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.base import AlgorithmState
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.registry import ALGORITHM_CLASSES, get_algorithm
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.signatures import mechanism_signature_rows, method_code_map
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config, parameter_hash


ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))


def test_algorithm_state_hash_clone_replay_is_stable():
    state = AlgorithmState({"method": "ERM", "step": 3, "tensor": torch.tensor([1.0, 2.0])})
    clone = state.clone()
    assert clone.hash() == state.hash()
    clone.payload["step"] = 4
    assert clone.hash() != state.hash()


def test_dynamic_method_codes_follow_config_order_and_remain_opaque():
    methods = config()["methods"]
    codes = method_code_map(methods)
    assert codes["ERM"] == "m000"
    assert codes["ASAM"] == "m011"
    assert len(set(codes.values())) == len(methods)


def test_mechanism_signatures_accept_expanded_methods_without_unknown_columns():
    rows = [{"opaque_direction_id": "u000", "method": method, "A_normalized_norm": float(i + 1)} for i, method in enumerate(config()["methods"])]
    signature = mechanism_signature_rows(rows, [], [], methods=config()["methods"])[0]
    assert "a_m011" in signature
    assert "a_m_unknown" not in signature


def test_no_method_specific_math_outside_algorithm_files_for_new_methods():
    assert not (ROOT / "src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/method_objectives.py").exists()
    forbidden = {'"SPECTRAL_NORM_REG"', '"SPECTRAL_REG_2024"', '"SVB_ORTHDNN"', '"STABLE_RANK_NORM"', '"SAM"', '"ASAM"'}
    for path in [
        ROOT / "src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/full_response.py",
        ROOT / "src/ood_repr_reg/run_task3_aopi_multimethod_mechanism_survey.py",
    ]:
        tree = ast.parse(path.read_text())
        constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        assert not (forbidden & {repr(value).replace("'", '"') for value in constants})


def test_existing_algorithm_default_step_preserves_parameter_update_path():
    cfg = config()
    model = build_model_from_config(cfg)
    before = parameter_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batches = ((torch.rand(4, 392), torch.zeros(4, 1)), (torch.rand(4, 392), torch.ones(4, 1)))
    algorithm = get_algorithm("ERM", cfg)
    result = algorithm.train_step(model, optimizer, batches, step=0, learning_rate=0.001, algorithm_state=algorithm.initial_state(seed=10))
    assert result.parts.applied_penalty_weight == 0.0
    assert parameter_hash(model) != before
    assert set(config()["methods"]) < set(ALGORITHM_CLASSES)
    assert set(config()["candidate_methods"]) <= set(ALGORITHM_CLASSES)
