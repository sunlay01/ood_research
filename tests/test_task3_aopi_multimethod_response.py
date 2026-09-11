import inspect

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.analysis import geometry_rows, world_gate
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.full_response import full_response_rows
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.source_observation import observation_geometry
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.task_response import HEAD_DIMENSION


def test_head_geometry_is_65d_and_source_observation_has_no_method_argument():
    assert HEAD_DIMENSION == 65
    assert "method" not in inspect.signature(observation_geometry).parameters


def test_geometry_rows_require_five_primary_columns():
    with torch.no_grad():
        A = torch.zeros(65, 5, dtype=torch.double)
        O = torch.zeros(130, 5, dtype=torch.double)
    a_rows, o_rows = geometry_rows(10, "ERM", A, O)
    assert len(a_rows) == len(o_rows) == 11
    assert all(row["finite"] for row in a_rows + o_rows)


def test_world_gate_accepts_exact_source_only_zero_structure():
    A = torch.randn(65, 5, dtype=torch.double)
    O = torch.randn(130, 5, dtype=torch.double)
    O[:, 2] = 0.0
    O[:, 4] = 0.0
    assert world_gate(A, O)["passed"] is True


def test_full_response_does_not_use_target_environment_in_continuation_source():
    source = inspect.getsource(full_response_rows)
    assert "worlds.evaluation" not in source
    assert "target_acc" not in source
    assert "target_only_learner_update" in source
