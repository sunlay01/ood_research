import torch

from ood_repr_reg.regularizers import (
    REGULARIZERS,
    LinearRepresentation,
    head_gradient,
    head_hessian,
    regularizer_value,
)
from ood_repr_reg.synthetic import Batch, make_dataset


def test_all_regularizers_are_finite_and_nonnegative() -> None:
    bundle = make_dataset(seed=0, n_train=32, n_eval=32, n_probe=16)
    torch.manual_seed(0)
    model = LinearRepresentation(input_dim=5, latent_dim=3, n_tasks=2)

    for name in REGULARIZERS:
        value = regularizer_value(name, model, bundle.train, mmd_max_samples=16)
        assert torch.isfinite(value)
        assert value >= -1e-7


def test_alignment_penalties_vanish_for_identical_environment_batches() -> None:
    torch.manual_seed(1)
    model = LinearRepresentation(input_dim=5, latent_dim=3, n_tasks=1)
    x = torch.randn(20, 5)
    y = torch.where(torch.randn(20) >= 0, 1.0, -1.0)
    batches = (Batch(x=x, y=y, task=0, env=0), Batch(x=x, y=y, task=0, env=1))

    for name in ("mmd", "coral", "grad_align", "hess_align"):
        value = regularizer_value(name, model, batches, mmd_max_samples=20)
        assert torch.allclose(value, torch.tensor(0.0), atol=1e-7)


def test_squared_risk_gradient_hessian_identity() -> None:
    bundle = make_dataset(seed=5, n_train=64, n_eval=32, n_probe=16)
    torch.manual_seed(5)
    model = LinearRepresentation(input_dim=5, latent_dim=3, n_tasks=2)
    left, right = [batch for batch in bundle.train if batch.task == 0]
    weight = model.heads[0]
    left_risk = torch.mean((model.predict(left) - left.y) ** 2)
    right_risk = torch.mean((model.predict(right) - right.y) ** 2)
    gradient_gap = head_gradient(model, left) - head_gradient(model, right)
    hessian_gap = head_hessian(model, left) - head_hessian(model, right)
    identity = weight @ gradient_gap - 0.5 * weight @ hessian_gap @ weight

    assert torch.allclose(left_risk - right_risk, identity, atol=1e-6)
