import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from ood_repr_reg.cmnist_feature_probe import SmallCMNISTCNN, make_environment
from ood_repr_reg.run_task3_cmnist_local_response import _write_preregistration
from ood_repr_reg.task3_cmnist_local_response.curvature import (
    damped_metric,
    head_cross_entropy_hessian,
    inverse_via_eigh,
    local_response_penalty,
    matched_random_metric,
    metric_inverse_solve_agrees,
    penalty_from_centered,
    source_forward,
)
from ood_repr_reg.task3_cmnist_local_response.evaluation import SelectionScore, select_source_only
from ood_repr_reg.task3_cmnist_local_response.penalties import shuffle_environment_batches
from ood_repr_reg.task3_cmnist_local_response.trainer import Task3Config, train_candidate


ROOT = Path(__file__).resolve().parents[1]


def _toy_batches(n=16, latent_dim=4):
    del latent_dim
    torch.manual_seed(7)
    gray = torch.rand(2 * n, 1, 28, 28)
    digit = torch.arange(2 * n) % 10
    left = make_environment(gray[:n], digit[:n], correlation=0.9, env=0, seed=1)
    right = make_environment(gray[n:], digit[n:], correlation=0.8, env=1, seed=2)
    return ((left.x, left.y), (right.x, right.y))


def test_task3r_live_artifacts_are_removed_and_active_task_replaced():
    assert not (ROOT / "round3_redesign/task3r_algorithmization").exists()
    assert not (ROOT / "src/ood_repr_reg/task3r_algorithmization").exists()
    assert not (ROOT / "src/ood_repr_reg/run_task3r_algorithmization.py").exists()
    assert not (ROOT / "tests/test_task3r_algorithmization.py").exists()
    assert "TASK3-CMNIST-LOCAL-RESPONSE" in (ROOT / "active/TASK.md").read_text()
    assert (ROOT / "round3_redesign/task3_applicability").exists()


def test_damped_head_metric_is_symmetric_positive_definite_and_invertible():
    model = SmallCMNISTCNN(latent_dim=4)
    batches = _toy_batches()
    _, features, logits = source_forward(model, batches)
    hessian = head_cross_entropy_hessian(features, logits)
    metric = damped_metric(hessian, epsilon=1e-2)
    assert torch.allclose(metric.matrix, metric.matrix.T, atol=1e-7)
    assert metric.min_eigenvalue > 0
    assert metric_inverse_solve_agrees(metric.matrix)
    eye = torch.eye(metric.matrix.shape[0], dtype=metric.matrix.dtype)
    assert torch.allclose(torch.linalg.solve(metric.matrix, eye), inverse_via_eigh(metric.matrix), atol=1e-5)


def test_local_response_reduces_to_gradient_alignment_when_metric_is_identity():
    centered = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
    grad = penalty_from_centered(centered)
    local = penalty_from_centered(centered, torch.eye(2))
    assert torch.allclose(grad, local)


def test_curvature_changes_local_response_for_fixed_gradients():
    centered = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    identity = penalty_from_centered(centered, torch.eye(2))
    curved = penalty_from_centered(centered, torch.diag(torch.tensor([0.25, 1.0])))
    assert float(curved) < float(identity)


def test_random_metric_matches_real_metric_scale():
    metric = torch.diag(torch.tensor([0.5, 2.0, 4.0, 8.0]))
    random = matched_random_metric(metric, seed=11)
    assert torch.allclose(torch.linalg.eigvalsh(random), torch.linalg.eigvalsh(metric), atol=1e-6)
    assert torch.allclose(torch.trace(random), torch.trace(metric), atol=1e-6)
    assert torch.allclose(torch.linalg.norm(random), torch.linalg.norm(metric), atol=1e-6)


def test_environment_shuffle_preserves_counts():
    batches = _toy_batches(n=10)
    shuffled = shuffle_environment_batches(batches, torch.Generator().manual_seed(5))
    assert [x.shape[0] for x, _ in shuffled] == [10, 10]
    assert sum(int(y.numel()) for _, y in shuffled) == 20


def test_local_response_gradients_reach_representation_parameters():
    model = SmallCMNISTCNN(latent_dim=4)
    penalty, _ = local_response_penalty(model, _toy_batches(), epsilon=1e-2)
    penalty.backward()
    rep_params = list(model.features.parameters()) + list(model.encoder.parameters())
    assert any(param.grad is not None and float(param.grad.norm()) > 0 for param in rep_params)


def test_train_candidate_source_path_does_not_reference_target():
    source = inspect.getsource(train_candidate)
    assert "data.target" not in source


def test_source_only_selection_ignores_target_attributes():
    weak_target = SimpleNamespace(
        selection_score=SelectionScore(0.9, 0.91, 1), target_accuracy=0.0
    )
    strong_target = SimpleNamespace(
        selection_score=SelectionScore(0.8, 0.99, 2), target_accuracy=1.0
    )
    assert select_source_only([weak_target, strong_target]) is weak_target


def test_preregistration_can_be_written_without_outcome_files(tmp_path):
    config = Task3Config(profile="smoke", seeds=(0,), control_seeds=(0,))
    design = _write_preregistration(tmp_path, config)
    assert design["status"] == "PREREGISTERED_BEFORE_NEW_LOCAL_RESPONSE_TARGET_OUTCOMES"
    assert not (tmp_path / "results" / "run_table.csv").exists()
    assert "preregistered_design_hash" in design
