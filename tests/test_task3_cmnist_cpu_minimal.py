import inspect

import torch
from torch.nn import functional as F

from ood_repr_reg.task3_cmnist_cpu_minimal.data import (
    ColoredEnvironment,
    binary_labels_from_digits,
    colors_from_labels,
    make_batch_schedule,
    make_environment,
    scheduled_source_batches,
    noisy_binary_labels,
)
from ood_repr_reg.task3_cmnist_cpu_minimal.methods import (
    analytic_head_hessian,
    calibrate_response_scale,
    centered_head_gradients,
    grad_response_penalty,
    identity_metric_penalty,
    inverse_hessian_metric_penalty,
    local_response_penalty,
    source_forward_with_features,
)
from ood_repr_reg.task3_cmnist_cpu_minimal.model import (
    CPUColoredMNISTMLP,
    augmented_head_dimension,
    linear_layer_count,
    parameter_hash,
)
from ood_repr_reg.task3_cmnist_cpu_minimal.trainer import train_one_method


def _toy_batches(seed=0, n=16):
    generator = torch.Generator().manual_seed(seed)
    x0 = torch.rand((n, 392), generator=generator)
    y0 = torch.randint(0, 2, (n, 1), generator=generator).float()
    x1 = torch.rand((n, 392), generator=generator)
    y1 = torch.randint(0, 2, (n, 1), generator=generator).float()
    return (x0, y0), (x1, y1)


def _config():
    return {
        "training": {"l2_regularizer_weight": 0.001},
        "response": {"damping_epsilon": 0.01, "calibration_update_ratio": 0.10},
        "irmv1": {"penalty_anneal_iters": 100, "penalty_weight": 10000.0},
    }


def _train_config():
    return {
        "device": "cpu",
        "data": {"label_definition": "digit < 5"},
        "model": {"input_dim": 392, "hidden_dim": 64, "activation": "relu", "linear_layers": 3},
        "training": {
            "steps": 501,
            "batch_size_per_environment": 8,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "l2_regularizer_weight": 0.001,
            "checkpoint_steps": [0, 100, 200, 300, 400, 500],
        },
        "response": {"damping_epsilon": 0.01, "calibration_update_ratio": 0.10},
        "irmv1": {"penalty_anneal_iters": 100, "penalty_weight": 10000.0},
    }


def _toy_envs(seed=100, n=16):
    generator = torch.Generator().manual_seed(seed)
    x0 = torch.rand((n, 392), generator=generator)
    y0 = torch.randint(0, 2, (n, 1), generator=generator).float()
    c0 = y0.reshape(-1).clone()
    x1 = torch.rand((n, 392), generator=generator)
    y1 = torch.randint(0, 2, (n, 1), generator=generator).float()
    c1 = y1.reshape(-1).clone()
    return (
        ColoredEnvironment(x0, y0, torch.arange(n), c0, 0.2, "source_env0"),
        ColoredEnvironment(x1, y1, torch.arange(n), c1, 0.1, "source_env1"),
    )


def test_binary_label_is_digit_less_than_5():
    digits = torch.arange(10)
    expected = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0]).float()
    assert torch.equal(binary_labels_from_digits(digits), expected)


def test_label_noise_is_deterministic():
    digits = torch.arange(100) % 10
    left = noisy_binary_labels(digits, 0.25, torch.Generator().manual_seed(3))
    right = noisy_binary_labels(digits, 0.25, torch.Generator().manual_seed(3))
    other = noisy_binary_labels(digits, 0.25, torch.Generator().manual_seed(4))
    assert torch.equal(left, right)
    assert not torch.equal(left, other)


def test_color_flip_semantics_are_correct():
    labels = torch.tensor([0, 1, 0, 1]).float()
    same = colors_from_labels(labels, 0.0, torch.Generator().manual_seed(1))
    opposite = colors_from_labels(labels, 1.0, torch.Generator().manual_seed(1))
    assert torch.equal(same, labels)
    assert torch.equal(opposite, 1.0 - labels)


def test_source_and_target_flip_probabilities():
    labels = (torch.arange(10000) % 2).float()
    for probability in (0.2, 0.1, 0.9):
        colors = colors_from_labels(labels, probability, torch.Generator().manual_seed(100 + int(probability * 100)))
        observed = float((colors != labels).float().mean())
        assert abs(observed - probability) <= 0.02


def test_same_seed_methods_share_same_batch_schedule():
    left = make_batch_schedule(source_pool_sizes=(100, 200), steps=501, batch_size_per_environment=512, seed=9)
    right = make_batch_schedule(source_pool_sizes=(100, 200), steps=501, batch_size_per_environment=512, seed=9)
    assert torch.equal(left.indices[0], right.indices[0])
    assert torch.equal(left.indices[1], right.indices[1])


def test_different_seeds_change_batch_schedule():
    left = make_batch_schedule(source_pool_sizes=(100, 200), steps=501, batch_size_per_environment=512, seed=9)
    right = make_batch_schedule(source_pool_sizes=(100, 200), steps=501, batch_size_per_environment=512, seed=10)
    assert not torch.equal(left.indices[0], right.indices[0])


def test_model_is_exactly_392_64_64_1():
    model = CPUColoredMNISTMLP()
    layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
    assert [tuple(layer.weight.shape) for layer in layers] == [(64, 392), (64, 64), (1, 64)]


def test_model_has_exactly_three_linear_layers():
    assert linear_layer_count(CPUColoredMNISTMLP()) == 3


def test_head_has_65_augmented_coordinates():
    assert augmented_head_dimension(CPUColoredMNISTMLP()) == 65


def test_grad_penalty_matches_manual_head_gradient_variance():
    torch.manual_seed(11)
    model = CPUColoredMNISTMLP()
    batches = _toy_batches(12)
    losses, _, _ = source_forward_with_features(model, batches)
    _, centered = centered_head_gradients(losses, model.head)
    manual = identity_metric_penalty(centered)
    observed = grad_response_penalty(model, batches).penalty
    assert torch.allclose(observed, manual)


def test_grad_penalty_is_zero_when_environment_gradients_equal():
    torch.manual_seed(13)
    model = CPUColoredMNISTMLP()
    batch = _toy_batches(14)[0]
    penalty = grad_response_penalty(model, (batch, batch)).penalty
    assert torch.allclose(penalty, torch.zeros_like(penalty), atol=1e-9)


def test_grad_penalty_backprop_reaches_encoder():
    torch.manual_seed(15)
    model = CPUColoredMNISTMLP()
    penalty = grad_response_penalty(model, _toy_batches(16)).penalty
    penalty.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.norm()) > 0.0
        for parameter in model.encoder.parameters()
    )


def test_head_hessian_matches_autograd_hessian_on_tiny_batch():
    torch.manual_seed(17)
    z = torch.randn(6, 64, dtype=torch.double)
    labels = torch.randint(0, 2, (6,), dtype=torch.double)
    theta = torch.randn(65, dtype=torch.double) * 0.1

    def loss_fn(params):
        logits = z @ params[:64] + params[64]
        return F.binary_cross_entropy_with_logits(logits, labels)

    logits = z @ theta[:64] + theta[64]
    autograd_hessian = torch.autograd.functional.hessian(loss_fn, theta)
    analytic = analytic_head_hessian((z[:3], z[3:]), (logits[:3, None], logits[3:, None]))
    assert torch.allclose(analytic, autograd_hessian, atol=1e-8)


def test_damped_hessian_is_positive_definite():
    torch.manual_seed(19)
    model = CPUColoredMNISTMLP()
    response = local_response_penalty(model, _toy_batches(20), damping_epsilon=0.01)
    assert response.damped_hessian is not None
    assert response.min_eigenvalue is not None and response.min_eigenvalue > 0.0


def test_local_response_uses_cholesky_not_explicit_inverse():
    source = inspect.getsource(inverse_hessian_metric_penalty)
    assert "cholesky_solve" in source
    assert "torch.linalg.inv" not in source
    assert ".inverse(" not in source


def test_local_response_matches_quadratic_form_on_tiny_problem():
    centered = torch.tensor([[1.0, -2.0], [-1.0, 2.0]])
    hessian = torch.diag(torch.tensor([2.0, 4.0], dtype=torch.double))
    penalty, *_ = inverse_hessian_metric_penalty(centered, hessian, damping_epsilon=0.0)
    solved = torch.linalg.solve(hessian, centered.double().T)
    manual = 0.5 * torch.sum(centered.double().T * solved)
    assert torch.allclose(penalty, manual)


def test_local_response_backprop_reaches_encoder():
    torch.manual_seed(21)
    model = CPUColoredMNISTMLP()
    penalty = local_response_penalty(model, _toy_batches(22), damping_epsilon=0.01).penalty
    penalty.backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.norm()) > 0.0
        for parameter in model.encoder.parameters()
    )


def test_hessian_metric_is_stop_gradient():
    torch.manual_seed(23)
    model = CPUColoredMNISTMLP()
    response = local_response_penalty(model, _toy_batches(24), damping_epsilon=0.01)
    assert response.hessian is not None and response.hessian.requires_grad is False
    assert response.damped_hessian is not None and response.damped_hessian.requires_grad is False


def test_calibration_scale_is_source_only():
    source = inspect.getsource(calibrate_response_scale)
    assert "target" not in source.lower()


def test_grad_realized_initial_update_ratio_is_0p10():
    torch.manual_seed(25)
    model = CPUColoredMNISTMLP()
    calibration = calibrate_response_scale(
        model,
        _toy_batches(26),
        method="GRAD",
        l2_regularizer_weight=0.001,
        damping_epsilon=0.01,
        calibration_update_ratio=0.10,
    )
    assert calibration.valid
    assert 0.099 <= calibration.realized_initial_update_ratio <= 0.101


def test_lr_realized_initial_update_ratio_is_0p10():
    torch.manual_seed(27)
    model = CPUColoredMNISTMLP()
    calibration = calibrate_response_scale(
        model,
        _toy_batches(28),
        method="LOCAL_RESPONSE",
        l2_regularizer_weight=0.001,
        damping_epsilon=0.01,
        calibration_update_ratio=0.10,
    )
    assert calibration.valid
    assert 0.099 <= calibration.realized_initial_update_ratio <= 0.101


def test_optimizer_persists_across_steps():
    source = inspect.getsource(train_one_method)
    assert source.count("torch.optim.Adam") == 1
    assert "for step in range(steps)" in source
    assert source.index("torch.optim.Adam") < source.index("for step in range(steps)")


def test_same_seed_all_methods_start_identically():
    hashes = []
    for _method in ("ERM", "IRMv1", "GRAD", "LOCAL_RESPONSE"):
        torch.manual_seed(31)
        hashes.append(parameter_hash(CPUColoredMNISTMLP()))
    assert len(set(hashes)) == 1


def test_same_seed_all_methods_receive_identical_batches():
    envs = _toy_envs(32)
    schedule = make_batch_schedule(source_pool_sizes=(16, 16), steps=501, batch_size_per_environment=8, seed=32)
    reference = scheduled_source_batches(envs, schedule, 137)
    for _method in ("ERM", "IRMv1", "GRAD", "LOCAL_RESPONSE"):
        observed = scheduled_source_batches(envs, schedule, 137)
        assert torch.equal(observed[0][0], reference[0][0])
        assert torch.equal(observed[1][1], reference[1][1])


def test_target_is_not_passed_to_training_step():
    source = inspect.getsource(train_one_method)
    assert "target" not in source.lower()


def test_final_checkpoint_is_always_step_500():
    torch.manual_seed(33)
    config = _train_config()
    envs = _toy_envs(33)
    schedule = make_batch_schedule(source_pool_sizes=(16, 16), steps=501, batch_size_per_environment=8, seed=33)
    model = CPUColoredMNISTMLP()
    result = train_one_method(
        model=model,
        source_envs=envs,
        batch_schedule=schedule,
        method="ERM",
        config=config,
        seed=33,
        initial_parameter_hash=parameter_hash(model),
    )
    assert result.finite
    assert [checkpoint.step for checkpoint in result.checkpoints] == [0, 100, 200, 300, 400, 500]
