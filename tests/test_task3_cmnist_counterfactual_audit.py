import csv
import inspect
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from ood_repr_reg import run_task3_cmnist_counterfactual_audit as runner
from ood_repr_reg.task3_cmnist_counterfactual_audit.analysis import ALLOWED_VERDICTS, build_paired_effects, summarize_audit
from ood_repr_reg.task3_cmnist_counterfactual_audit.diagnostics import (
    compute_counterfactual_diagnostics,
    covariance,
    model_counterfactual_diagnostics,
    whitener,
)
from ood_repr_reg.task3_cmnist_counterfactual_audit.probe import build_counterfactual_probe, probe_invariant_checks
from ood_repr_reg.task3_cmnist_cpu_minimal.data import ColoredEnvironment
from ood_repr_reg.task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP, linear_layer_count, parameter_hash


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "round3_redesign/task3_cmnist_counterfactual_audit/results"


def _synthetic_env(n=12):
    gray = torch.arange(n * 14 * 14, dtype=torch.float32).reshape(n, 14, 14) / 10000.0
    images = []
    colors = torch.arange(n) % 2
    for index in range(n):
        if int(colors[index]) == 0:
            images.append(torch.stack((gray[index], torch.zeros_like(gray[index])), dim=0))
        else:
            images.append(torch.stack((torch.zeros_like(gray[index]), gray[index]), dim=0))
    return ColoredEnvironment(
        images=torch.stack(images, dim=0).reshape(n, -1),
        labels=(torch.arange(n) % 2).float().reshape(-1, 1),
        digits=(torch.arange(n) % 10).long(),
        colors=colors.float(),
        color_flip_prob=0.9,
        role="target_flip_0p9",
    )


def test_model_identity_is_correct_cpu_minimal_mlp():
    model = CPUColoredMNISTMLP()
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert [tuple(layer.weight.shape) for layer in layers] == [(64, 392), (64, 64), (1, 64)]
    assert linear_layer_count(model) == 3


def test_counterfactual_probe_preserves_grayscale_and_swaps_only_color_channel():
    probe = build_counterfactual_probe(_synthetic_env())
    checks = probe_invariant_checks(probe)
    assert checks == {key: True for key in checks}
    red = probe.red.reshape(probe.n_examples, 2, 14, 14)
    green = probe.green.reshape(probe.n_examples, 2, 14, 14)
    assert torch.equal(red[:, 0], green[:, 1])
    assert torch.equal(red[:, 1], green[:, 0])


def test_clean_label_is_digit_less_than_five():
    probe = build_counterfactual_probe(_synthetic_env())
    assert torch.equal(probe.clean_labels, (probe.digits < 5).float())


def test_original_target_color_metadata_is_unused_for_counterfactual_pairs():
    env = _synthetic_env()
    probe = build_counterfactual_probe(env)
    mutated = replace(env, colors=1.0 - env.colors)
    other = build_counterfactual_probe(mutated)
    assert torch.equal(probe.red, other.red)
    assert torch.equal(probe.green, other.green)
    assert torch.equal(probe.clean_labels, other.clean_labels)


def test_diagnostics_do_not_mutate_parameters():
    torch.manual_seed(7)
    model = CPUColoredMNISTMLP()
    probe = build_counterfactual_probe(_synthetic_env(n=20))
    before = parameter_hash(model)
    result = model_counterfactual_diagnostics(model, probe, batch_size=5)
    after = parameter_hash(model)
    assert result["finite"] is True
    assert before == after


def test_scalar_logit_formula_matches_head_projection():
    torch.manual_seed(8)
    model = CPUColoredMNISTMLP()
    probe = build_counterfactual_probe(_synthetic_env(n=10))
    with torch.no_grad():
        red_z = model.encode(probe.red)
        green_z = model.encode(probe.green)
        red_logits = model.head(red_z).reshape(-1)
        green_logits = model.head(green_z).reshape(-1)
        projected = (green_z - red_z) @ model.head.weight.reshape(-1)
    assert torch.allclose(green_logits - red_logits, projected, atol=1e-7)


def test_whitening_retains_stable_directions_without_exploding_nulls():
    torch.manual_seed(9)
    z = torch.randn(80, 5)
    z[:, 4] = 1.0
    whitening = whitener(z, relative_tolerance=1e-5)
    white = (z.double() - whitening.center) @ whitening.transform
    eigenvalues = torch.linalg.eigvalsh(covariance(white))
    retained = eigenvalues[eigenvalues > 1e-4]
    assert torch.allclose(retained, torch.ones_like(retained), atol=1e-5, rtol=1e-5)
    assert float(eigenvalues.min()) >= -1e-8
    assert int(whitening.retained.sum()) == 4


def test_zero_color_response_sanity():
    torch.manual_seed(10)
    z = torch.randn(12, 6)
    logits = torch.randn(12, 1)
    labels = (torch.arange(12) % 2).float()
    result = compute_counterfactual_diagnostics(
        red_z=z,
        green_z=z,
        red_logits=logits,
        green_logits=logits,
        head_weight=torch.ones(6),
        clean_labels=labels,
    )
    assert result["latent_color_response"] == pytest.approx(0.0, abs=1e-12)
    assert result["prediction_color_response"] == pytest.approx(0.0, abs=1e-12)
    assert result["probability_color_response"] == pytest.approx(0.0, abs=1e-12)
    assert result["counterfactual_prediction_consistency"] == pytest.approx(1.0)


def test_reconstruction_training_path_does_not_pass_target_to_trainer():
    source = inspect.getsource(runner._reconstruct_and_diagnose)
    assert "train_one_method" in source
    assert "source_envs=data.source_envs" in source
    train_call = source[source.index("train_one_method") : source.index("if not result.finite")]
    assert "target_env" not in train_call
    assert "optimizer" not in inspect.getsource(runner)


def test_analysis_verdict_enum_and_pair_completeness():
    rows = []
    for seed in [10, 11, 12, 13, 14]:
        rows.append({"seed": seed, "method": "ERM", **{metric: 1.0 for metric in runner.PAIRED_METRICS}})
        rows.append({"seed": seed, "method": "IRMv1", **{metric: 1.0 for metric in runner.PAIRED_METRICS}})
    paired = build_paired_effects(rows, seeds=[10, 11, 12, 13, 14])
    summary = summarize_audit(diagnostic_rows=rows, paired_rows=paired, valid=True, checkpoint_source="reconstructed", seeds=[10, 11, 12, 13, 14])
    assert len(paired) == 5
    assert summary["verdict"] in ALLOWED_VERDICTS


def test_output_rows_and_reconciliation_after_runner_exists():
    diagnostics_path = RESULTS_DIR / "diagnostics.csv"
    summary_path = RESULTS_DIR / "summary.json"
    if not diagnostics_path.exists() or not summary_path.exists():
        pytest.skip("counterfactual audit outputs have not been generated yet")
    rows = list(csv.DictReader(diagnostics_path.open(encoding="utf-8", newline="")))
    assert len(rows) == 10
    assert {(int(row["seed"]), row["method"]) for row in rows} == {
        (seed, method) for seed in [10, 11, 12, 13, 14] for method in ["ERM", "IRMv1"]
    }
    assert all(row["reconciliation_status"] == "PASS" for row in rows)
    assert max(float(row["reconciliation_max_abs_delta"]) for row in rows) <= 1e-6
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["verdict"] in ALLOWED_VERDICTS
    assert summary["row_count"] == 10
