import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.coral import coral_penalty
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.registry import ALGORITHM_CLASSES, get_algorithm
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.vrex import vrex_penalty_from_losses
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.method_trainer import train_survey_method
from ood_repr_reg.task3_cmnist_cpu_minimal.data import make_batch_schedule, make_environment
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config


ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))


def toy_envs():
    images = torch.rand(16, 1, 28, 28)
    digits = torch.arange(16) % 10
    generator = torch.Generator().manual_seed(4)
    return (
        make_environment(images, digits, color_flip_prob=0.2, label_noise=0.25, role="source0", image_subsample=2, normalize_pixels=True, generator=generator),
        make_environment(images, digits, color_flip_prob=0.1, label_noise=0.25, role="source1", image_subsample=2, normalize_pixels=True, generator=generator),
    )


def test_config_fixes_method_panel_and_training_identity():
    cfg = config()
    assert cfg["methods"] == ["ERM", "IRMv1", "VREX", "CORAL"]
    assert cfg["seeds"] == [10, 11, 12, 13, 14]
    assert cfg["training"]["steps"] == 501
    assert cfg["training"]["batch_size_per_environment"] == 512
    assert cfg["vrex"]["lambda"] == 10000.0
    assert cfg["vrex"]["post_anneal_penalty_weight"] == 10000.0
    assert cfg["vrex"]["whole_loss_rescale_after_anneal"] is True


def test_each_algorithm_has_own_file_and_registry_is_only_dispatch():
    expected_modules = {
        "ERM": "ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.erm",
        "IRMv1": "ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.irmv1",
        "VREX": "ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.vrex",
        "CORAL": "ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.coral",
    }
    assert {name: cls.__module__ for name, cls in ALGORITHM_CLASSES.items()} == expected_modules
    full_response_source = (ROOT / "src/ood_repr_reg/task3_aopi_multimethod_mechanism_survey/full_response.py").read_text()
    assert 'method == "VREX"' not in full_response_source
    assert 'method == "CORAL"' not in full_response_source


def test_vrex_is_squared_risk_gap_and_coral_uses_n_minus_one_covariance():
    losses = torch.tensor([1.0, 3.0])
    assert torch.equal(vrex_penalty_from_losses(losses), torch.tensor(4.0))
    left = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    right = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    observed = coral_penalty((left, right))
    expected_cov = torch.tensor([[2.0, 2.0], [2.0, 2.0]])
    assert torch.allclose(observed, (torch.tensor([1.0, 1.0]).square().sum() / 2) + (expected_cov.square().sum() / 4))


def test_vrex_resets_adam_once_at_step_100_and_is_source_only():
    cfg = config()
    envs = toy_envs()
    schedule = make_batch_schedule(source_pool_sizes=(16, 16), steps=101, batch_size_per_environment=2, seed=3)
    model = build_model_from_config(cfg)
    result = train_survey_method(model=model, source_envs=envs, batch_schedule=schedule, method="VREX", config={**cfg, "training": {**cfg["training"], "steps": 101, "batch_size_per_environment": 2}}, seed=3, initial_parameter_hash="init")
    assert result.finite
    assert result.optimizer_reset_count == 1
    assert result.objective_formula_id == "CMNIST_VREX_ANNEALED_V1"


def test_vrex_post_anneal_objective_uses_rescale():
    cfg = config()
    envs = toy_envs()
    schedule = make_batch_schedule(source_pool_sizes=(16, 16), steps=101, batch_size_per_environment=2, seed=3)
    model = build_model_from_config(cfg)
    batches = tuple((env.images[schedule.indices[i][100]], env.labels[schedule.indices[i][100]]) for i, env in enumerate(envs))
    parts = get_algorithm("VREX", cfg).objective(model, batches, step=100)
    assert parts.applied_penalty_weight == 10000.0
    assert parts.rescaled_after_anneal is True
