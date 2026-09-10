import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from ood_repr_reg.cmnist_feature_probe import SmallCMNISTCNN, make_environment
from ood_repr_reg.run_task3_cmnist_local_response import _write_preregistration
from ood_repr_reg.run_task3_cmnist_local_response import _official_verdict
from ood_repr_reg.task3_cmnist_local_response.official_irm_reference import (
    OfficialEnvironment,
    OfficialIRMMLP,
    run_official_task3_comparison,
    official_response_penalty,
    official_summary,
)
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
from ood_repr_reg.task3_cmnist_local_response.trainer import (
    Task3Config,
    default_config,
    summarize_baseline_recovery,
    train_candidate,
)


ROOT = Path(__file__).resolve().parents[1]


def _toy_batches(n=16, latent_dim=4):
    del latent_dim
    torch.manual_seed(7)
    gray = torch.rand(2 * n, 1, 28, 28)
    digit = torch.arange(2 * n) % 10
    left = make_environment(gray[:n], digit[:n], correlation=0.9, env=0, seed=1)
    right = make_environment(gray[n:], digit[n:], correlation=0.8, env=1, seed=2)
    return ((left.x, left.y), (right.x, right.y))


def _toy_official_env(role: str, offset: int = 0) -> OfficialEnvironment:
    n = 12
    generator = torch.Generator().manual_seed(100 + offset)
    labels = ((torch.arange(n) + offset) % 2).float()[:, None]
    colors = labels.reshape(-1).clone()
    x = torch.rand((n, 2, 14, 14), generator=generator)
    return OfficialEnvironment(
        images=x,
        labels=labels,
        colors=colors,
        color_flip_prob=0.1 + 0.1 * offset,
        role=role,
    )


def test_task3r_live_artifacts_are_removed_and_active_task_replaced():
    assert not (ROOT / "round3_redesign/task3r_algorithmization").exists()
    assert not (ROOT / "src/ood_repr_reg/task3r_algorithmization").exists()
    assert not (ROOT / "src/ood_repr_reg/run_task3r_algorithmization.py").exists()
    assert not (ROOT / "tests/test_task3r_algorithmization.py").exists()
    active_task = (ROOT / "active/TASK.md").read_text()
    assert (
        "TASK3-CMNIST-LOCAL-RESPONSE" in active_task
        or "TASK3-BASELINE-FIDELITY-RECOVERY" in active_task
        or "TASK3-CMNIST-CPU-MINIMAL" in active_task
        or "TASK-RERUN-1-3D-CLEANROOM" in active_task
        or "TASK-AOPI-CMNIST-REINSTANTIATION-AUDIT" in active_task
        or "TASK-AOPI-CMNIST-REINSTANTIATION-REPAIR" in active_task
    )
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


def test_main_profile_uses_legacy_strength_cmnist_protocol():
    config = default_config("main")
    assert config.train_per_environment == 2500
    assert config.source_eval_per_environment == 500
    assert config.target_eval == 1000
    assert config.epochs == 8
    assert config.label_noise == 0.25
    assert 1.0 in config.beta_grid
    assert 10.0 in config.beta_grid
    assert config.baseline_recovery_protocol == "official_irm_colored_mnist"
    assert config.official_irm_penalty_weight == 10000.0
    assert config.official_irm_penalty_anneal_iters == 100


def test_preregistration_declares_baseline_recovery_gate(tmp_path):
    design = _write_preregistration(tmp_path, default_config("main"))
    gate = design["baseline_recovery_gate"]
    assert gate["must_pass_before_primary_comparison"] is True
    assert gate["reference_protocol"] == "official_irm_colored_mnist"
    assert gate["official_irm_penalty_weight"] == 10000.0
    assert gate["official_irm_penalty_anneal_iters"] == 100
    assert gate["minimum_target_accuracy"] >= 0.55
    assert gate["maximum_erm_target_accuracy"] <= 0.45
    assert gate["minimum_irmv1_advantage_over_erm"] >= 0.05
    assert gate["label_noise"] == 0.25


def test_official_primary_preregistration_matches_official_protocol(tmp_path):
    design = _write_preregistration(
        tmp_path,
        default_config("main"),
        primary_protocol="official_colored_mnist_reversed_color_mlp",
    )
    assert design["primary_protocol"] == "official_colored_mnist_reversed_color_mlp"
    assert "2-channel 14x14 MLP" in design["architecture"]
    assert design["official_protocol_primary"]["enabled"] is True
    assert design["official_protocol_primary"]["train_color_flip_probs"] == [0.2, 0.1]
    assert design["official_protocol_primary"]["target_color_flip_prob"] == 0.9
    assert design["official_protocol_primary"]["label_noise"] == 0.25
    assert design["official_protocol_primary"]["irmv1_penalty_weight"] == 10000.0
    assert design["official_protocol_primary"]["response_beta_grid"] == list(default_config("main").beta_grid)
    assert design["official_protocol_primary"]["response_methods_use_irm_loss_rescale"] is False
    assert "V-REx" not in design["methods"]


def test_baseline_recovery_summary_requires_nontrivial_irmv1_target_accuracy():
    config = Task3Config(baseline_recovery_min_target_accuracy=0.55)
    low_irm = summarize_baseline_recovery(
        config,
        [
            {"method": "ERM", "beta": 0.0, "target_accuracy": 0.10, "prediction_color_agreement": 0.99},
            {"method": "IRMV1", "beta": 1.0, "target_accuracy": 0.11, "prediction_color_agreement": 0.99},
        ],
    )
    high_erm = summarize_baseline_recovery(
        config,
        [
            {"method": "ERM", "beta": 0.0, "target_accuracy": 0.64, "prediction_color_agreement": 0.46},
            {"method": "IRMV1", "beta": 1.0, "target_accuracy": 0.58, "prediction_color_agreement": 0.53},
        ],
    )
    ok = summarize_baseline_recovery(
        config,
        [
            {"method": "ERM", "beta": 0.0, "target_accuracy": 0.12, "prediction_color_agreement": 0.98},
            {"method": "IRMV1", "beta": 1.0, "target_accuracy": 0.58, "prediction_color_agreement": 0.53},
        ],
    )
    assert low_irm["passed"] is False
    assert high_erm["passed"] is False
    assert ok["passed"] is True


def test_official_local_response_penalty_is_finite_and_backpropagates():
    model = OfficialIRMMLP(hidden_dim=4)
    envs = (_toy_official_env("env0", 0), _toy_official_env("env1", 1))
    penalty, diag = official_response_penalty(
        "LOCAL_RESPONSE",
        model,
        envs,
        device=torch.device("cpu"),
        epsilon=1e-2,
    )
    assert torch.isfinite(penalty)
    assert diag["response_penalty_metric"] == "real_inverse_hessian"
    penalty.backward()
    encoder_has_grad = any(
        parameter.grad is not None and float(parameter.grad.norm()) > 0.0
        for parameter in model.encoder.parameters()
    )
    head_has_grad = any(
        parameter.grad is not None and float(parameter.grad.norm()) > 0.0
        for parameter in model.head.parameters()
    )
    assert encoder_has_grad
    assert head_has_grad


def test_official_protocol_summary_and_verdict_use_official_rows():
    rows = [
        {"seed": 0, "method": "ERM", "target_accuracy": 0.16, "train_accuracy": 0.72, "prediction_color_agreement": 0.98, "penalty_weight": 0.0},
        {"seed": 0, "method": "IRMV1", "target_accuracy": 0.64, "train_accuracy": 0.70, "prediction_color_agreement": 0.55, "penalty_weight": 10000.0},
        {"seed": 0, "method": "HEAD_GRADIENT_VARIANCE_SURROGATE", "target_accuracy": 0.20, "train_accuracy": 0.71, "prediction_color_agreement": 0.92, "penalty_weight": 0.1},
        {"seed": 0, "method": "LOCAL_RESPONSE", "target_accuracy": 0.21, "train_accuracy": 0.71, "prediction_color_agreement": 0.90, "penalty_weight": 0.1},
        {"seed": 0, "method": "RANDOM_METRIC", "target_accuracy": 0.19, "train_accuracy": 0.71, "prediction_color_agreement": 0.91, "penalty_weight": 0.1},
    ]
    compact = official_summary(rows)
    pairs = {(row["left_method"], row["right_method"]): row for row in compact["paired_comparisons"]}
    assert abs(float(pairs[("IRMV1", "ERM")]["mean_difference"]) - 0.48) < 1e-12
    verdict, criteria = _official_verdict(compact, {"passed": True})
    assert verdict == "TASK3-CMNIST-PARTIAL"
    assert criteria["official_reversed_color_protocol"] is True
    assert criteria["all_10_primary_seeds_completed"] is False


def test_official_task3_source_selection_does_not_use_target_accuracy():
    source = inspect.getsource(run_official_task3_comparison)
    selection_block = source[source.index("selected = max") : source.index("for row in group")]
    assert "target_accuracy" not in selection_block


def test_official_response_methods_do_not_claim_irm_loss_rescale():
    rows = [
        {"seed": 0, "method": "HEAD_GRADIENT_VARIANCE_SURROGATE", "target_accuracy": 0.20, "train_accuracy": 0.80, "prediction_color_agreement": 0.80, "penalty_weight": 0.1, "selected_by_source_rule": True},
        {"seed": 0, "method": "LOCAL_RESPONSE", "target_accuracy": 0.21, "train_accuracy": 0.80, "prediction_color_agreement": 0.80, "penalty_weight": 0.1, "selected_by_source_rule": True},
    ]
    compact = official_summary(rows)
    assert compact["selected_rows"] == 2
    assert {row["mean_selected_penalty_weight"] for row in compact["method_summary"]} == {0.1}


def test_official_summary_parses_csv_boolean_selection_flags():
    rows = [
        {"seed": 0, "method": "HEAD_GRADIENT_VARIANCE_SURROGATE", "target_accuracy": "0.10", "train_accuracy": "0.70", "prediction_color_agreement": "0.90", "penalty_weight": "0.001", "selected_by_source_rule": "False"},
        {"seed": 0, "method": "HEAD_GRADIENT_VARIANCE_SURROGATE", "target_accuracy": "0.20", "train_accuracy": "0.80", "prediction_color_agreement": "0.80", "penalty_weight": "0.01", "selected_by_source_rule": "True"},
    ]
    compact = official_summary(rows)
    assert compact["selected_rows"] == 1
    assert compact["method_summary"][0]["mean_selected_penalty_weight"] == 0.01
    assert compact["method_summary"][0]["mean_target_accuracy"] == 0.20
