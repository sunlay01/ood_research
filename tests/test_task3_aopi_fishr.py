import json
from pathlib import Path

import torch

from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.algorithms.fishr import (
    centered_diagonal_variance,
    classifier_gradient_matrix,
    fishr_penalty_from_variances,
)
from ood_repr_reg.task3_aopi_multimethod_mechanism_survey.config_schema import validate_config
from ood_repr_reg.task3_cmnist_cpu_minimal.model import build_model_from_config


ROOT = Path(__file__).resolve().parents[1]


def config():
    return validate_config(json.loads((ROOT / "configs/task3_aopi_multimethod_mechanism_survey.json").read_text()))


def test_fishr_classifier_gradient_matrix_is_head_only_and_finite():
    cfg = config()
    model = build_model_from_config(cfg)
    images = torch.rand(8, 392)
    labels = (torch.arange(8) % 2).float()[:, None]
    gradients = classifier_gradient_matrix(model, images, labels)
    assert gradients.shape == (8, 65)
    assert torch.isfinite(gradients).all()


def test_centered_variance_and_zero_penalty_for_identical_environments():
    gradients = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    variance = centered_diagonal_variance(gradients)
    assert torch.allclose(variance, torch.tensor([8.0 / 3.0, 8.0 / 3.0]))
    assert torch.equal(fishr_penalty_from_variances((variance, variance)), torch.tensor(0.0))


def test_fishr_penalty_is_env_swap_invariant():
    left = torch.tensor([1.0, 2.0, 4.0])
    right = torch.tensor([2.0, 1.0, 0.0])
    assert torch.equal(fishr_penalty_from_variances((left, right)), fishr_penalty_from_variances((right, left)))
