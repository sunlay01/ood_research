import json
from pathlib import Path

import numpy as np
import torch

from ood_repr_reg.cmnist_feature_probe import (
    SmallCMNISTCNN,
    colorize,
    counterfactual_diagnostics,
    make_counterfactual_probe,
    make_environment,
    regularizer_penalty,
    source_diagnostic_penalties,
)
from ood_repr_reg import run_cmnist_feature_probe as runner


def _digits(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(12)
    gray = torch.rand((n, 1, 28, 28), generator=generator)
    digit = torch.arange(n) % 10
    return gray, digit


def test_color_counterfactual_preserves_shape_and_label() -> None:
    gray, digit = _digits(20)
    red = colorize(gray, torch.zeros(20, dtype=torch.long))
    green = colorize(gray, torch.ones(20, dtype=torch.long))
    probe = make_counterfactual_probe(gray, digit)

    assert torch.allclose(red.sum(dim=1, keepdim=True), gray)
    assert torch.allclose(green.sum(dim=1, keepdim=True), gray)
    assert torch.equal(probe.y, (digit >= 5).long())
    assert torch.count_nonzero(red[:, 1:]) == 0
    assert torch.count_nonzero(green[:, 0]) == 0
    assert torch.count_nonzero(green[:, 2]) == 0


def test_environment_correlation_and_seed_are_deterministic() -> None:
    gray, digit = _digits(2000)
    first = make_environment(gray, digit, correlation=0.8, env=0, seed=5)
    second = make_environment(gray, digit, correlation=0.8, env=0, seed=5)
    agreement = (first.color == first.y).float().mean()

    assert torch.equal(first.color, second.color)
    assert abs(float(agreement) - 0.8) < 0.03


def test_all_declared_penalties_are_finite_and_differentiable() -> None:
    gray, digit = _digits(24)
    left = make_environment(gray[:12], digit[:12], correlation=0.9, env=0, seed=1)
    right = make_environment(gray[12:], digit[12:], correlation=0.8, env=1, seed=2)
    model = SmallCMNISTCNN(latent_dim=8)
    batches = ((left.x, left.y), (right.x, right.y))

    for method in ("erm", "l2", "irmv1", "coral"):
        value = regularizer_penalty(method, model, batches)
        assert torch.isfinite(value)
        assert value >= -1e-8
        if method != "erm":
            gradients = torch.autograd.grad(value, tuple(model.parameters()), allow_unused=True)
            assert any(gradient is not None for gradient in gradients)


def test_prediction_response_matches_direct_logit_margin_shift() -> None:
    gray, digit = _digits(30)
    probe = make_counterfactual_probe(gray, digit)
    torch.manual_seed(4)
    model = SmallCMNISTCNN(latent_dim=8)
    metrics, _ = counterfactual_diagnostics(model, probe, device=torch.device("cpu"))

    with torch.no_grad():
        red_margin = model(probe.red)[:, 1] - model(probe.red)[:, 0]
        green_margin = model(probe.green)[:, 1] - model(probe.green)[:, 0]
        expected = (green_margin - red_margin).square().mean()
    assert np.isclose(metrics["prediction_color_response"], float(expected), rtol=1e-5)


def test_source_diagnostics_evaluate_all_penalties() -> None:
    gray, digit = _digits(24)
    environments = (
        make_environment(gray[:12], digit[:12], correlation=0.9, env=0, seed=1),
        make_environment(gray[12:], digit[12:], correlation=0.8, env=1, seed=2),
    )
    model = SmallCMNISTCNN(latent_dim=8)
    values = source_diagnostic_penalties(
        model, environments, max_samples=8, device=torch.device("cpu")
    )

    assert set(values) == {"diagnostic_l2", "diagnostic_irmv1", "diagnostic_coral"}
    assert all(np.isfinite(value) and value >= 0.0 for value in values.values())


def test_runner_writes_numeric_and_visual_outputs(tmp_path: Path, monkeypatch) -> None:
    train_gray, train_digit = _digits(48)
    test_gray, test_digit = _digits(32)

    def fake_load(root: Path, *, train: bool, download: bool):
        del root, download
        return (train_gray, train_digit) if train else (test_gray, test_digit)

    monkeypatch.setattr(runner, "load_mnist_tensors", fake_load)
    config = {
        "experiment_id": "TEST-CMNIST-VIS",
        "data_root": str(tmp_path / "data"),
        "download": False,
        "seeds": [0],
        "methods": {"erm": [0.0]},
        "source_correlations": [0.9, 0.8],
        "target_correlation": 0.1,
        "train_per_environment": 8,
        "source_eval_per_environment": 4,
        "target_eval": 4,
        "probe_size": 4,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "latent_dim": 8,
        "device": "cpu",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "output"

    rows = runner.run(config_path, output)

    assert len(rows) == 1
    for name in (
        "metrics.csv",
        "config.json",
        "environment.json",
        "counterfactual_paths.png",
        "feature_response_summary.png",
        "counterfactual_examples.png",
    ):
        assert (output / name).stat().st_size > 0
    assert (output / "checkpoints" / "seed0-erm-l0.pt").stat().st_size > 0
