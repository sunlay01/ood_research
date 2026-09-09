"""Facebook IRM Colored MNIST protocol reimplementation for baseline recovery.

The native upstream script is GPU-only because it calls ``.cuda()`` directly.
This module keeps a CPU-compatible, line-faithful implementation of the same
data construction and objective so the gate can be run on the local machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, autograd, nn, optim
from torch.nn import functional as F

from ..cmnist_feature_probe import load_mnist_tensors


@dataclass(frozen=True)
class OfficialIRMConfig:
    data_root: str = "data"
    download: bool = True
    seeds: tuple[int, ...] = (0, 1, 2)
    hidden_dim: int = 256
    l2_regularizer_weight: float = 1e-3
    learning_rate: float = 1e-3
    penalty_anneal_iters: int = 100
    penalty_weight: float = 10000.0
    steps: int = 501
    label_noise: float = 0.25
    train_color_flip_probs: tuple[float, float] = (0.2, 0.1)
    target_color_flip_prob: float = 0.9
    device: str = "cpu"


@dataclass(frozen=True)
class OfficialEnvironment:
    images: Tensor
    labels: Tensor
    colors: Tensor
    flip_probability: float
    role: str


class OfficialIRMMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256, grayscale_model: bool = False) -> None:
        super().__init__()
        self.grayscale_model = bool(grayscale_model)
        input_dim = 14 * 14 if self.grayscale_model else 2 * 14 * 14
        lin1 = nn.Linear(input_dim, hidden_dim)
        lin2 = nn.Linear(hidden_dim, hidden_dim)
        lin3 = nn.Linear(hidden_dim, 1)
        for layer in (lin1, lin2, lin3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, x: Tensor) -> Tensor:
        if self.grayscale_model:
            out = x.view(x.shape[0], 2, 14 * 14).sum(dim=1)
        else:
            out = x.view(x.shape[0], 2 * 14 * 14)
        return self._main(out)


def _xor(a: Tensor, b: Tensor) -> Tensor:
    return (a - b).abs()


def _bernoulli(probability: float, size: int, generator: torch.Generator) -> Tensor:
    return (torch.rand(size, generator=generator) < probability).float()


def make_official_environment(
    images: Tensor,
    digits: Tensor,
    *,
    flip_probability: float,
    label_noise: float,
    role: str,
    generator: torch.Generator,
) -> OfficialEnvironment:
    downsampled = images.reshape((-1, 28, 28))[:, ::2, ::2]
    labels = (digits < 5).float()
    labels = _xor(labels, _bernoulli(label_noise, len(labels), generator))
    colors = _xor(labels, _bernoulli(flip_probability, len(labels), generator))
    colored = torch.stack([downsampled, downsampled], dim=1)
    colored[torch.arange(len(colored)), (1 - colors).long(), :, :] = 0.0
    return OfficialEnvironment(
        images=colored.float(),
        labels=labels[:, None].float(),
        colors=colors.float(),
        flip_probability=float(flip_probability),
        role=role,
    )


def build_official_environments(config: OfficialIRMConfig, seed: int) -> tuple[OfficialEnvironment, OfficialEnvironment, OfficialEnvironment]:
    gray, digits = load_mnist_tensors(Path(config.data_root), train=True, download=config.download)
    train_images = gray[:50000, 0]
    train_digits = digits[:50000]
    val_images = gray[50000:, 0]
    val_digits = digits[50000:]
    order = torch.randperm(len(train_images), generator=torch.Generator().manual_seed(seed))
    train_images = train_images[order]
    train_digits = train_digits[order]
    generator = torch.Generator().manual_seed(seed + 314159)
    env0 = make_official_environment(
        train_images[::2],
        train_digits[::2],
        flip_probability=config.train_color_flip_probs[0],
        label_noise=config.label_noise,
        role="train_env0_plus90",
        generator=generator,
    )
    env1 = make_official_environment(
        train_images[1::2],
        train_digits[1::2],
        flip_probability=config.train_color_flip_probs[1],
        label_noise=config.label_noise,
        role="train_env1_plus80",
        generator=generator,
    )
    target = make_official_environment(
        val_images,
        val_digits,
        flip_probability=config.target_color_flip_prob,
        label_noise=config.label_noise,
        role="target_minus90",
        generator=generator,
    )
    return env0, env1, target


def mean_nll(logits: Tensor, labels: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels)


def mean_accuracy(logits: Tensor, labels: Tensor) -> Tensor:
    preds = (logits > 0.0).float()
    return ((preds - labels).abs() < 1e-2).float().mean()


def irm_penalty(logits: Tensor, labels: Tensor) -> Tensor:
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = mean_nll(logits * scale, labels)
    grad = autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)


def weight_norm_squared(module: nn.Module) -> Tensor:
    total = next(module.parameters()).new_zeros(())
    for parameter in module.parameters():
        total = total + parameter.norm().pow(2)
    return total


@torch.no_grad()
def evaluate_official(module: nn.Module, env: OfficialEnvironment, device: torch.device) -> dict[str, float]:
    logits = module(env.images.to(device))
    labels = env.labels.to(device)
    preds = (logits > 0.0).float().cpu().reshape(-1)
    labels_cpu = env.labels.cpu().reshape(-1)
    colors_cpu = env.colors.cpu().reshape(-1)
    return {
        "nll": float(mean_nll(logits, labels).detach().cpu()),
        "accuracy": float(mean_accuracy(logits, labels).detach().cpu()),
        "color_label_accuracy": float((colors_cpu == labels_cpu).float().mean()),
        "prediction_color_agreement": float((preds == colors_cpu).float().mean()),
    }


def train_official_irm(
    train_envs: tuple[OfficialEnvironment, OfficialEnvironment],
    target_env: OfficialEnvironment,
    *,
    method: str,
    seed: int,
    config: OfficialIRMConfig,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config.device)
    model = OfficialIRMMLP(config.hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    method_key = method.upper()
    final_train_nll = torch.tensor(float("nan"))
    final_train_acc = torch.tensor(float("nan"))
    final_train_penalty = torch.tensor(float("nan"))

    for step in range(config.steps):
        env_metrics = []
        for env in train_envs:
            logits = model(env.images.to(device))
            labels = env.labels.to(device)
            env_metrics.append({
                "nll": mean_nll(logits, labels),
                "acc": mean_accuracy(logits, labels),
                "penalty": irm_penalty(logits, labels),
            })
        final_train_nll = torch.stack([item["nll"] for item in env_metrics]).mean()
        final_train_acc = torch.stack([item["acc"] for item in env_metrics]).mean()
        final_train_penalty = torch.stack([item["penalty"] for item in env_metrics]).mean()
        loss = final_train_nll.clone() + config.l2_regularizer_weight * weight_norm_squared(model)
        applied_penalty = 0.0
        rescaled = False
        if method_key == "IRMV1":
            applied_penalty = config.penalty_weight if step >= config.penalty_anneal_iters else 1.0
            loss = loss + applied_penalty * final_train_penalty
            if applied_penalty > 1.0:
                loss = loss / applied_penalty
                rescaled = True
        elif method_key != "ERM":
            raise ValueError(f"unknown official IRM method: {method}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_eval = [evaluate_official(model, env, device) for env in train_envs]
    target_eval = evaluate_official(model, target_env, device)
    return {
        "seed": seed,
        "method": "IRMv1" if method_key == "IRMV1" else "ERM",
        "steps": config.steps,
        "hidden_dim": config.hidden_dim,
        "learning_rate": config.learning_rate,
        "l2_regularizer_weight": config.l2_regularizer_weight,
        "penalty_anneal_iters": config.penalty_anneal_iters,
        "penalty_weight": config.penalty_weight if method_key == "IRMV1" else 0.0,
        "final_applied_penalty_weight": applied_penalty,
        "final_rescaled_after_anneal": rescaled,
        "label_noise": config.label_noise,
        "train_color_flip_probs": list(config.train_color_flip_probs),
        "target_color_flip_prob": config.target_color_flip_prob,
        "train_examples_per_env": [int(env.images.shape[0]) for env in train_envs],
        "target_examples": int(target_env.images.shape[0]),
        "train_accuracy": float(sum(row["accuracy"] for row in train_eval) / len(train_eval)),
        "target_accuracy": target_eval["accuracy"],
        "target_nll": target_eval["nll"],
        "prediction_color_agreement": target_eval["prediction_color_agreement"],
        "target_color_label_accuracy": target_eval["color_label_accuracy"],
        "final_train_nll": float(final_train_nll.detach().cpu()),
        "final_train_accuracy_step_metric": float(final_train_acc.detach().cpu()),
        "final_train_penalty": float(final_train_penalty.detach().cpu()),
        "protocol": "facebook_irm_colored_mnist_cpu_reimplementation",
        "source_environment_count": 2,
        "target_used_for_training_or_selection": False,
    }


def run_official_irm_reimplementation(config: OfficialIRMConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        env0, env1, target = build_official_environments(config, int(seed))
        for method in ("ERM", "IRMv1"):
            rows.append(train_official_irm((env0, env1), target, method=method, seed=int(seed), config=config))
    return rows
