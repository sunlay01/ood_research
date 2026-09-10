import csv
import inspect
import json
from pathlib import Path

import pytest
import torch

from ood_repr_reg import run_task3_ood_capability_decomposition as runner
from ood_repr_reg.task3_cmnist_counterfactual_audit.probe import build_counterfactual_probe, probe_invariant_checks
from ood_repr_reg.task3_cmnist_cpu_minimal.data import ColoredEnvironment, ColoredMNISTData, make_batch_schedule
from ood_repr_reg.task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP, parameter_hash
from ood_repr_reg.task3_ood_capability_decomposition.analysis import ALLOWED_DOMINANT_BOTTLENECKS, ALLOWED_VERDICTS, summarize_capabilities
from ood_repr_reg.task3_ood_capability_decomposition.artifacts import (
    CheckpointRecord,
    FeatureBundle,
    load_checkpoint_manifest,
    load_verified_model,
    sha256_file,
)
from ood_repr_reg.task3_ood_capability_decomposition.experiments import (
    run_feature_coverage,
    run_feature_selection,
    train_source_head,
)
from ood_repr_reg.task3_ood_capability_decomposition.linear import (
    color_response_basis,
    fit_ridge_classifier,
    projection_matrix,
    two_fold_ridge_accuracy,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results/checkpoint_manifest.csv"
CONFIG = ROOT / "configs/task3_cmnist_cpu_minimal.json"
RESULTS_DIR = ROOT / "round3_redesign/ood_capability_decomposition/results"


def _config(steps=5, batch=8):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["training"]["steps"] = steps
    config["training"]["batch_size_per_environment"] = batch
    return config


def _env(images, labels, *, role):
    n = int(images.shape[0])
    return ColoredEnvironment(
        images=images.float(),
        labels=labels.reshape(-1, 1).float(),
        digits=(torch.arange(n) % 10).long(),
        colors=(torch.arange(n) % 2).float(),
        color_flip_prob=0.2,
        role=role,
    )


def _synthetic_bundle(seed=10, method="ERM", n_source=24, n_target=30):
    torch.manual_seed(seed)
    model = CPUColoredMNISTMLP()
    schedule = make_batch_schedule(
        source_pool_sizes=(n_source, n_source),
        steps=5,
        batch_size_per_environment=8,
        seed=seed,
    )
    source0_images = torch.rand(n_source, 392)
    source1_images = torch.rand(n_source, 392)
    target_images = torch.rand(n_target, 392)
    source0_labels = (torch.arange(n_source) % 2).float()
    source1_labels = ((torch.arange(n_source) + 1) % 2).float()
    target_labels = (torch.arange(n_target) % 2).float()
    data = ColoredMNISTData(
        source_envs=(
            _env(source0_images, source0_labels, role="source0"),
            _env(source1_images, source1_labels, role="source1"),
        ),
        target_env=_env(target_images, target_labels, role="target"),
        batch_schedule=schedule,
        seed=seed,
    )
    probe = build_counterfactual_probe(data.target_env)
    with torch.no_grad():
        source0_features = model.encode(data.source_envs[0].images).double()
        source1_features = model.encode(data.source_envs[1].images).double()
        target_features = model.encode(data.target_env.images).double()
        red_features = model.encode(probe.red).double()
        green_features = model.encode(probe.green).double()
    before = parameter_hash(model)
    return FeatureBundle(
        record=CheckpointRecord(
            seed=seed,
            method=method,
            path=Path("synthetic.pt"),
            checkpoint_sha256="synthetic",
            parameter_hash=before,
            config_sha256="synthetic",
            git_commit="synthetic",
        ),
        model=model,
        data=data,
        probe=probe,
        source0_features=source0_features,
        source1_features=source1_features,
        target_features=target_features,
        red_features=red_features,
        green_features=green_features,
        parameter_hash_before=before,
    )


def test_checkpoint_manifest_loading_resolves_existing_files():
    records = load_checkpoint_manifest(MANIFEST, seeds=[10], methods=["ERM", "IRMv1"], root=ROOT)
    assert len(records) == 2
    assert all(record.path.exists() for record in records)
    assert all(sha256_file(record.path) == record.checkpoint_sha256 for record in records)


def test_config_and_parameter_hash_verification_for_checkpoint():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_sha = sha256_file(CONFIG)
    record = load_checkpoint_manifest(MANIFEST, seeds=[10], methods=["ERM"], root=ROOT)[0]
    model = load_verified_model(record, config, config_sha256=config_sha)
    assert parameter_hash(model) == record.parameter_hash


def test_frozen_encoder_immutability_under_feature_selection():
    bundle = _synthetic_bundle()
    before = parameter_hash(bundle.model)
    rows = run_feature_selection(bundle, _config())
    assert len(rows) == 3
    assert parameter_hash(bundle.model) == before
    assert all(row["frozen_encoder_parameter_hash_before"] == row["frozen_encoder_parameter_hash_after"] for row in rows)


def test_source_only_probe_does_not_use_target_metrics():
    source = inspect.getsource(run_feature_coverage)
    before_fit = source[source.index("source_probe = fit_ridge_classifier") - 200 : source.index("source_probe = fit_ridge_classifier") + 120]
    assert "target" not in before_fit.lower()
    assert "oracle" in source.lower()


def test_oracle_rows_are_explicitly_flagged():
    rows = run_feature_selection(_synthetic_bundle(), _config())
    oracle_rows = [row for row in rows if row["head_method"] == "ORACLE_CLEAN"]
    assert len(oracle_rows) == 1
    assert oracle_rows[0]["oracle"] is True
    assert oracle_rows[0]["source_only"] is False


def test_oracle_coverage_uses_red_green_counterfactual_rows_not_latent_average():
    source = inspect.getsource(run_feature_coverage)
    assert "_counterfactual_balanced_features" in source
    assert "two_fold_ridge_accuracy(balanced, _counterfactual_balanced_clean_labels(bundle)" in source


def test_counterfactual_construction_is_inherited_from_corrected_audit():
    bundle = _synthetic_bundle()
    checks = probe_invariant_checks(bundle.probe)
    assert checks == {key: True for key in checks}


def test_projection_matrix_symmetry_idempotence_and_rank_clipping():
    torch.manual_seed(7)
    red = torch.randn(18, 5)
    green = red + 0.1 * torch.randn(18, 5)
    basis = color_response_basis(red, green)
    projection = projection_matrix(basis, 99)
    assert torch.allclose(projection, projection.T, atol=1e-8)
    assert torch.allclose(projection @ projection, projection, atol=1e-8)
    assert torch.linalg.matrix_rank(projection, tol=1e-7).item() == 0


def test_ridge_classifier_is_deterministic():
    torch.manual_seed(8)
    x = torch.randn(40, 6)
    y = (torch.arange(40) % 2).double()
    left = fit_ridge_classifier(x, y, ridge=1e-3)
    right = fit_ridge_classifier(x, y, ridge=1e-3)
    assert torch.equal(left.weight, right.weight)
    assert torch.equal(left.bias, right.bias)
    assert two_fold_ridge_accuracy(x, y, ridge=1e-3)["accuracy"] == two_fold_ridge_accuracy(x, y, ridge=1e-3)["accuracy"]


def test_head_only_training_not_mutating_encoder_and_identical_initialization():
    bundle = _synthetic_bundle(seed=11)
    config = _config()
    import ood_repr_reg.task3_ood_capability_decomposition.experiments as experiments

    initial = experiments.initialized_head(64, bundle.record.seed + 12345)
    initial_state = {key: value.detach().double().clone() for key, value in initial.state_dict().items()}
    before = parameter_hash(bundle.model)
    left = train_source_head(bundle, method="HEAD_ERM", initial_state=initial_state, config=config)
    right = train_source_head(bundle, method="HEAD_IRMv1", initial_state=initial_state, config=config)
    assert parameter_hash(bundle.model) == before
    assert set(left.state_dict()) == set(right.state_dict())


def test_runner_declares_only_abc_and_no_de_execution():
    assert runner.EXECUTED_EXPERIMENTS == ["A_COVERAGE", "B_SEPARABILITY", "C_SELECTION"]
    assert set(runner.FORBIDDEN_EXPERIMENTS) == {"D_SOURCE_SIDE_IDENTIFICATION", "E_OPTIMIZATION_RESPONSE_ABILITY"}
    source = inspect.getsource(runner.run_analysis)
    assert "FORBIDDEN" not in source


def test_summary_verdict_enum_validity():
    coverage = []
    separability = []
    selection = []
    for seed in range(10, 15):
        for method, target in [("ERM", 0.1), ("IRMv1", 0.65)]:
            coverage.append({"seed": seed, "encoder_method": method, "original_target_acc": target, "oracle_clean_balanced_accuracy": 0.7})
            for rank in [0, 1, 2, 4, 8, 16, 32, 64]:
                separability.append({"seed": seed, "encoder_method": method, "rank_k": rank, "projected_target_acc": target})
            for head_method in ["HEAD_ERM", "HEAD_IRMv1", "ORACLE_CLEAN"]:
                selection.append({"seed": seed, "encoder_method": method, "head_method": head_method, "target_acc": 0.7 if head_method == "ORACLE_CLEAN" else target})
    summary = summarize_capabilities(coverage, separability, selection, valid=True)
    assert summary["verdict"] in ALLOWED_VERDICTS
    assert summary["dominant_bottleneck"] in ALLOWED_DOMINANT_BOTTLENECKS


def test_output_row_counts_and_verdict_after_runner_exists():
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("capability decomposition outputs have not been generated yet")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"] in ALLOWED_VERDICTS
    assert summary["dominant_bottleneck"] in ALLOWED_DOMINANT_BOTTLENECKS
    expected_counts = {
        "input_checkpoint_manifest.csv": 10,
        "feature_coverage.csv": 10,
        "separability_curve.csv": 80,
        "head_selection.csv": 30,
        "paired_capability_summary.csv": 5,
    }
    for name, expected in expected_counts.items():
        rows = list(csv.DictReader((RESULTS_DIR / name).open(encoding="utf-8", newline="")))
        assert len(rows) == expected
