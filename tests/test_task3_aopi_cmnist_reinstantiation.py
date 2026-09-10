import csv
import inspect
import json
from pathlib import Path

import pytest
import torch

from ood_repr_reg import run_task3_aopi_cmnist_reinstantiation as runner
from ood_repr_reg.task3_aopi_cmnist_reinstantiation.analysis import ALLOWED_VERDICTS, discrimination_gate, finite_prediction_gate, geometry_gate, overall_verdict
from ood_repr_reg.task3_aopi_cmnist_reinstantiation.finite_validation import finite_response
from ood_repr_reg.task3_aopi_cmnist_reinstantiation.learner_response import HEAD_DIMENSION, pi_operator, refine_head, source_state
from ood_repr_reg.task3_aopi_cmnist_reinstantiation.source_geometry import (
    encoder_parameter_hash,
    freeze_encoder,
    source_risk_hessian,
    whiten_source_hessian,
)
from ood_repr_reg.task3_aopi_cmnist_reinstantiation.world_tangents import EPSILONS, REPORT_DIRECTIONS, WorldFactory, centered_pair
from ood_repr_reg.task3_cmnist_cpu_minimal.data import build_task3_data
from ood_repr_reg.task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP, linear_layer_count, parameter_hash


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/task3_cmnist_cpu_minimal.json"
OUT = ROOT / "round3_redesign/task3_aopi_cmnist_reinstantiation"


def _features(seed=3):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(40, 64, generator=generator, dtype=torch.double), torch.randn(36, 64, generator=generator, dtype=torch.double))


def _labels():
    return ((torch.arange(40) % 2).reshape(-1, 1).double(), (torch.arange(36) % 2).reshape(-1, 1).double())


def test_model_identity_and_final_head_block_dimension():
    model = CPUColoredMNISTMLP()
    assert linear_layer_count(model) == 3
    assert sum(item.numel() for item in model.head.parameters()) == HEAD_DIMENSION
    assert sum(item.numel() for item in model.encoder.parameters()) > HEAD_DIMENSION


def test_base_tangent_world_exactly_reproduces_cpu_minimal_sources():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = build_task3_data(config, 10, data_root=ROOT / "data", download=False)
    observed = WorldFactory(config, 10, data_root=ROOT / "data", download=False).build()
    for old, new in zip(expected.source_envs, observed.source_envs):
        assert torch.equal(old.images, new.images)
        assert torch.equal(old.labels, new.labels)
        assert torch.equal(old.colors, new.colors)


def test_tangent_pairs_are_symmetric_and_reproducible():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    factory = WorldFactory(config, 10, data_root=ROOT / "data", download=False)
    plus, minus = centered_pair(factory, REPORT_DIRECTIONS["source_env0_color"], EPSILONS[0])
    assert plus.parameters.label_noise == pytest.approx(minus.parameters.label_noise)
    assert plus.parameters.source_flip_probs[0] + minus.parameters.source_flip_probs[0] == pytest.approx(0.4)
    other_plus, other_minus = centered_pair(factory, REPORT_DIRECTIONS["source_env0_color"], EPSILONS[0])
    assert torch.equal(plus.source_envs[0].images, other_plus.source_envs[0].images)
    assert torch.equal(minus.source_envs[0].images, other_minus.source_envs[0].images)


def test_encoder_freeze_does_not_change_parameter_hash():
    model = CPUColoredMNISTMLP()
    complete = parameter_hash(model)
    frozen = freeze_encoder(model)
    assert encoder_parameter_hash(model) == frozen.encoder_hash
    assert parameter_hash(model) == complete
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_hessian_is_symmetric_and_whitening_is_consistent():
    features, labels = _features(), _labels()
    reference = refine_head("ERM", features, labels, torch.zeros(HEAD_DIMENSION, dtype=torch.double))
    hessian = source_risk_hessian(features, reference.weights)
    whitening = whiten_source_hessian(hessian)
    assert torch.allclose(hessian, hessian.T, atol=1e-10)
    assert whitening.identity_error <= 1e-8
    assert whitening.root.shape == (HEAD_DIMENSION, HEAD_DIMENSION)


def test_pi_is_explicit_head_only_source_state_map():
    features, labels = _features(), _labels()
    reference = refine_head("ERM", features, labels, torch.zeros(HEAD_DIMENSION, dtype=torch.double))
    whitening = whiten_source_hessian(source_risk_hessian(features, reference.weights))
    pi = pi_operator("ERM", features, labels, reference.weights, whitening.root)
    assert pi.pi.shape == (HEAD_DIMENSION, 4 * HEAD_DIMENSION)
    assert torch.isfinite(pi.pi).all()
    assert source_state(features, labels, reference.weights, create_graph=False).shape == (4 * HEAD_DIMENSION,)


def test_finite_response_is_head_only_and_finite():
    features, labels = _features(), _labels()
    reference = refine_head("ERM", features, labels, torch.zeros(HEAD_DIMENSION, dtype=torch.double))
    whitening = whiten_source_hessian(source_risk_hessian(features, reference.weights))
    perturbed = (features[0] + 0.001, features[1] - 0.001)
    response = finite_response("ERM", perturbed, labels, features, labels, reference, whitening.root, torch.ones(HEAD_DIMENSION, dtype=torch.double), 0.01)
    assert response.predicted.shape == (HEAD_DIMENSION,)
    assert response.actual.shape == (HEAD_DIMENSION,)
    assert response.finite


def test_source_only_modules_do_not_reference_target():
    for module in ("learner_response.py", "finite_validation.py"):
        source = (ROOT / "src/ood_repr_reg/task3_aopi_cmnist_reinstantiation" / module).read_text(encoding="utf-8").lower()
        assert "target" not in source


def test_runner_source_fit_precedes_evaluation_response():
    source = inspect.getsource(runner.audit_record)
    assert source.index("refine_head") < source.index("response_A")
    assert "source_features(encoder, base)" in source


def test_pre_registered_gate_enums_and_rules():
    geometry = [{"seed": seed, "method": method, "tangent": tangent, "epsilon": epsilon, "finite": True, "gram_disagreement_with_epsilon_pair": 0.0, "response_rank": 3, "reparameterization_gram_disagreement": 0.0} for seed in range(10, 15) for method in ("ERM", "IRMv1") for tangent in REPORT_DIRECTIONS for epsilon in EPSILONS]
    validation = [{"seed": seed, "method": method, "tangent": tangent, "epsilon": 0.01, "finite": True, "cosine_similarity": 1.0, "relative_vector_error": 0.0, "task_inner_product_sign_agreement": True} for seed in range(10, 15) for method in ("ERM", "IRMv1") for tangent in ("source_env0_color", "source_env1_color", "shared_label_noise")]
    mismatch = [{"seed": seed, "method": method, "tangent": tangent, "epsilon": 0.01, "mismatch_norm": 2.0 if method == "ERM" else 1.0} for seed in range(10, 15) for method in ("ERM", "IRMv1") for tangent in REPORT_DIRECTIONS]
    g1, g2 = geometry_gate(geometry), finite_prediction_gate(validation)
    g3 = discrimination_gate(mismatch, geometry_pass=g1["pass"], finite_pass=g2["pass"])
    assert g1["pass"] and g2["pass"] and g3["pass"]
    assert overall_verdict(valid=True, geometry=g1, finite=g2, discrimination=g3) in ALLOWED_VERDICTS


def test_outputs_have_complete_primary_rows_after_runner_exists():
    summary_path = OUT / "results/summary.json"
    if not summary_path.exists():
        pytest.skip("audit has not run")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["overall_verdict"] in ALLOWED_VERDICTS
    rows = list(csv.DictReader((OUT / "results/pi_finite_validation.csv").open(encoding="utf-8", newline="")))
    assert len(rows) == 5 * 2 * len(REPORT_DIRECTIONS) * len(EPSILONS)
