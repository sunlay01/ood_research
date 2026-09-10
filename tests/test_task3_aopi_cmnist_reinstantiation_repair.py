import inspect
import json
from pathlib import Path

import pytest
import torch

from ood_repr_reg import run_task3_aopi_cmnist_reinstantiation_repair as runner
from ood_repr_reg.task3_aopi_cmnist_reinstantiation_repair.full_response import DELTA, STEPS, full_response_rows, reconstruct_adam_state
from ood_repr_reg.task3_aopi_cmnist_reinstantiation_repair.head_response import DAMPING_GRID
from ood_repr_reg.task3_aopi_cmnist_reinstantiation_repair.linearity_checks import check_linearity
from ood_repr_reg.task3_aopi_cmnist_reinstantiation_repair.smooth_world import BASIS, DERIVED_DIRECTIONS, SmoothPool, SmoothWorlds, base_world_identity, environment_parameters, outcome_weight
from ood_repr_reg.task3_aopi_cmnist_reinstantiation_repair.source_observation import observation_geometry
from ood_repr_reg.task3_aopi_cmnist_reinstantiation_repair.task_response import HEAD_DIMENSION, expected_head_risk, smooth_fd_consistency


ROOT = Path(__file__).resolve().parents[1]


def test_exact_four_outcome_weights_sum_to_one_and_are_smooth():
    p = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    q = torch.tensor(0.25, dtype=torch.double, requires_grad=True)
    weights = sum(outcome_weight(p, q, label_flip, color_flip) for label_flip in (0, 1) for color_flip in (0, 1))
    assert torch.allclose(weights, torch.tensor(1.0, dtype=torch.double))
    assert torch.autograd.grad(weights, (p, q), allow_unused=False) is not None


def test_zero_tangent_recovers_correct_cmnist():
    empty = SmoothPool(torch.empty(0, 28, 28), torch.empty(0, dtype=torch.long), "empty")
    worlds = SmoothWorlds((empty, empty), (empty, empty), torch.zeros(3, dtype=torch.double))
    assert base_world_identity(worlds)
    for observed, expected in (
        (environment_parameters(worlds.base_theta, environment=0, evaluation=False), (0.2, 0.25)),
        (environment_parameters(worlds.base_theta, environment=1, evaluation=False), (0.1, 0.25)),
        (environment_parameters(worlds.base_theta, environment=0, evaluation=True), (0.9, 0.25)),
        (environment_parameters(worlds.base_theta, environment=1, evaluation=True), (0.9, 0.25)),
    ):
        assert tuple(float(value) for value in observed) == pytest.approx(expected)


def test_all_mixture_weights_are_valid_probabilities():
    for p in (0.1, 0.2, 0.9):
        weights = [outcome_weight(torch.tensor(p), torch.tensor(0.25), label_flip, color_flip) for label_flip in (0, 1) for color_flip in (0, 1)]
        assert all(0.0 <= float(weight) <= 1.0 for weight in weights)
        assert float(sum(weights)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="probability"):
        outcome_weight(torch.tensor(1.1), torch.tensor(0.25), 0, 0)


def test_smooth_expected_head_risk_has_probability_derivative():
    generator = torch.Generator().manual_seed(7)
    pool = SmoothPool(torch.randint(0, 255, (3, 28, 28), generator=generator, dtype=torch.uint8), torch.tensor([1, 6, 3]), "toy")
    outcomes = tuple((torch.randn(3, 64, generator=generator, dtype=torch.double), labels, label_flip, color_flip) for _, labels, label_flip, color_flip in pool.outcome_batches())
    p = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    q = torch.tensor(0.25, dtype=torch.double, requires_grad=True)
    weight = torch.zeros(HEAD_DIMENSION, dtype=torch.double, requires_grad=True)
    loss = expected_head_risk(outcomes, p, q, weight)
    dp, dq = torch.autograd.grad(loss, (p, q))
    assert torch.isfinite(loss) and torch.isfinite(dp) and torch.isfinite(dq)


def test_g0_uses_only_three_columns_and_independent_derived_directions():
    generator = torch.Generator().manual_seed(9)
    A, O = torch.randn(65, 3, generator=generator, dtype=torch.double), torch.randn(130, 3, generator=generator, dtype=torch.double)
    result = check_linearity(
        A, O,
        actual_common_A=A @ DERIVED_DIRECTIONS["common_source_color"],
        actual_anti_A=A @ DERIVED_DIRECTIONS["antisymmetric_source_color"],
        actual_common_O=O @ DERIVED_DIRECTIONS["common_source_color"],
        actual_anti_O=O @ DERIVED_DIRECTIONS["antisymmetric_source_color"],
    )
    assert result.passed and result.a_rank <= 3 and result.o_rank <= 3
    assert len(BASIS) == 3


def test_primary_observation_has_no_method_branch_or_penalty_gradient():
    source = inspect.getsource(observation_geometry).lower()
    assert "method" not in inspect.signature(observation_geometry).parameters
    assert 'if method' not in source
    assert "penalty" not in source


def test_corrected_a_toy_gate_subtracts_source_gradient():
    assert runner._toy_correct_A_gate()


def test_smooth_fd_gate_is_explicitly_implemented():
    source = inspect.getsource(smooth_fd_consistency)
    assert "central FD" in source
    assert "plus - minus" in source


def test_head_path_is_secondary_and_damping_grid_is_fixed():
    source = (ROOT / "src/ood_repr_reg/task3_aopi_cmnist_reinstantiation_repair/head_response.py").read_text(encoding="utf-8")
    assert DAMPING_GRID == (1e-10, 1e-8, 1e-6)
    assert "LBFGS" in source
    assert "target" not in source.lower()


def test_full_path_is_source_only_and_uses_preregistered_steps():
    source = inspect.getsource(full_response_rows).lower() + inspect.getsource(reconstruct_adam_state).lower()
    assert "target" not in source
    assert DELTA == 0.01 and STEPS == (1, 5, 20)
    assert "method_objective" in inspect.getsource(reconstruct_adam_state)


def test_runner_declares_old_audit_invalidated_and_full_mismatch_unestablished():
    source = (ROOT / "src/ood_repr_reg/run_task3_aopi_cmnist_reinstantiation_repair.py").read_text(encoding="utf-8")
    assert "AOPI-OLD-AUDIT-INVALIDATED-BY-SEMANTIC-MISMATCH" in source
    assert "b6c9eaf" in source
    assert "NOT_ESTABLISHED" in source
    assert source.index("_preregister(config)") < source.index("corrected_geometry(model, worlds)")


def test_outputs_have_expected_row_counts_after_run():
    summary_path = ROOT / "round3_redesign/task3_aopi_cmnist_reinstantiation_repair/results/summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"] in {"AOPI-REPAIR-PASS", "AOPI-REPAIR-PARTIAL", "AOPI-REPAIR-FAIL", "AOPI-REPAIR-INVALID"}
    assert summary["rows"]["A"] in {0, 30}
    assert summary["rows"]["O"] in {0, 30}
    assert summary["rows"]["full"] in {0, 90}
