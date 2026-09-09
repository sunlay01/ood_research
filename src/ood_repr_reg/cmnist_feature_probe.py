"""Colored-MNIST counterfactual feature-response experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ColoredEnvironment:
    x: Tensor
    y: Tensor
    digit: Tensor
    color: Tensor
    correlation: float
    env: int
    label_noise: float = 0.0


@dataclass(frozen=True)
class CounterfactualProbe:
    red: Tensor
    green: Tensor
    y: Tensor
    digit: Tensor


class SmallCMNISTCNN(nn.Module):
    def __init__(self, latent_dim: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24 * 7 * 7, latent_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(latent_dim, 2)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(self.features(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.encode(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def colorize(gray: Tensor, color: Tensor) -> Tensor:
    """Render grayscale digits in red (0) or green (1)."""
    if gray.ndim == 3:
        gray = gray[:, None]
    if gray.ndim != 4 or gray.shape[1] != 1:
        raise ValueError("gray must have shape [N, 1, H, W] or [N, H, W]")
    color = color.to(device=gray.device, dtype=torch.long)
    if color.shape != (gray.shape[0],):
        raise ValueError("color must have one value per image")
    output = torch.zeros((gray.shape[0], 3, gray.shape[2], gray.shape[3]), dtype=gray.dtype)
    output = output.to(gray.device)
    output[:, 0] = gray[:, 0] * (color == 0)[:, None, None]
    output[:, 1] = gray[:, 0] * (color == 1)[:, None, None]
    return output


def make_environment(
    gray: Tensor,
    digit: Tensor,
    *,
    correlation: float,
    env: int,
    seed: int,
    label_noise: float = 0.0,
) -> ColoredEnvironment:
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [0, 1]")
    if not 0.0 <= label_noise <= 1.0:
        raise ValueError("label_noise must be in [0, 1]")
    y = (digit >= 5).long()
    generator = torch.Generator().manual_seed(seed)
    if label_noise > 0.0:
        flips = torch.rand(y.shape[0], generator=generator) < label_noise
        y = torch.where(flips, 1 - y, y)
    agrees = torch.rand(y.shape[0], generator=generator) < correlation
    color = torch.where(agrees, y, 1 - y)
    return ColoredEnvironment(
        x=colorize(gray, color),
        y=y,
        digit=digit.clone(),
        color=color,
        correlation=correlation,
        env=env,
        label_noise=label_noise,
    )


def make_counterfactual_probe(gray: Tensor, digit: Tensor) -> CounterfactualProbe:
    n = gray.shape[0]
    return CounterfactualProbe(
        red=colorize(gray, torch.zeros(n, dtype=torch.long)),
        green=colorize(gray, torch.ones(n, dtype=torch.long)),
        y=(digit >= 5).long(),
        digit=digit.clone(),
    )


def load_mnist_tensors(root: Path, *, train: bool, download: bool) -> tuple[Tensor, Tensor]:
    from torchvision.datasets import MNIST

    dataset = MNIST(root=str(root), train=train, download=download)
    gray = dataset.data.float().div(255.0).unsqueeze(1)
    return gray, dataset.targets.long()


def deterministic_subset(
    gray: Tensor, digit: Tensor, *, n: int, seed: int, offset: int = 0
) -> tuple[Tensor, Tensor]:
    if offset + n > gray.shape[0]:
        raise ValueError("requested subset exceeds dataset size")
    order = torch.randperm(gray.shape[0], generator=torch.Generator().manual_seed(seed))
    selected = order[offset : offset + n]
    return gray[selected], digit[selected]


def covariance(z: Tensor) -> Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(z.shape[0] - 1, 1)


def regularizer_penalty(
    method: str,
    model: SmallCMNISTCNN,
    environments: tuple[tuple[Tensor, Tensor], ...],
) -> Tensor:
    if method == "erm":
        return next(model.parameters()).new_zeros(())
    if method == "l2":
        values = torch.cat([parameter.flatten() for parameter in model.parameters()])
        return values.square().mean()

    encoded = [(model.encode(x), y) for x, y in environments]
    if method == "coral":
        terms = []
        for left in range(len(encoded)):
            for right in range(left + 1, len(encoded)):
                difference = covariance(encoded[left][0]) - covariance(encoded[right][0])
                terms.append(difference.square().mean())
        return torch.stack(terms).mean()
    if method == "irmv1":
        terms = []
        for z, y in encoded:
            scale = torch.ones((), device=z.device, requires_grad=True)
            risk = F.cross_entropy(model.head(z) * scale, y)
            gradient = torch.autograd.grad(risk, scale, create_graph=True)[0]
            terms.append(gradient.square())
        return torch.stack(terms).mean()
    raise ValueError(f"unknown method: {method}")


def _batch_indices(n: int, batch_size: int, generator: torch.Generator) -> list[Tensor]:
    order = torch.randperm(n, generator=generator)
    return list(order.split(batch_size))


def train_model(
    environments: tuple[ColoredEnvironment, ...],
    *,
    method: str,
    strength: float,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[SmallCMNISTCNN, dict[str, float]]:
    seed_everything(seed)
    model = SmallCMNISTCNN(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 7919)
    final_risk = float("nan")
    final_penalty = float("nan")

    for _ in range(epochs):
        permutations = [
            _batch_indices(environment.x.shape[0], batch_size, generator)
            for environment in environments
        ]
        for batch_number in range(min(len(parts) for parts in permutations)):
            batches = tuple(
                (
                    environment.x[permutations[index][batch_number]].to(device),
                    environment.y[permutations[index][batch_number]].to(device),
                )
                for index, environment in enumerate(environments)
            )
            risks = torch.stack([F.cross_entropy(model(x), y) for x, y in batches])
            risk = risks.mean()
            penalty = regularizer_penalty(method, model, batches)
            objective = risk + strength * penalty
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            final_risk = float(risk.detach())
            final_penalty = float(penalty.detach())

    model.eval()
    return model, {"final_train_risk": final_risk, "final_train_penalty": final_penalty}


@torch.no_grad()
def accuracy(
    model: SmallCMNISTCNN,
    environment: ColoredEnvironment,
    device: torch.device,
) -> float:
    logits = model(environment.x.to(device))
    return float((logits.argmax(dim=1).cpu() == environment.y).float().mean())


def _whitener(z: Tensor, relative_tolerance: float = 1e-5) -> tuple[Tensor, Tensor]:
    center = z.mean(dim=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance(z))
    threshold = eigenvalues.max().clamp_min(1e-12) * relative_tolerance
    inverse_root = torch.where(
        eigenvalues > threshold,
        eigenvalues.clamp_min(threshold).rsqrt(),
        torch.zeros_like(eigenvalues),
    )
    return center, eigenvectors @ torch.diag(inverse_root) @ eigenvectors.T


def _ridge_coordinates(z: Tensor, targets: Tensor, ridge: float = 1e-3) -> Tensor:
    ones = torch.ones((z.shape[0], 1), dtype=z.dtype)
    design = torch.cat((z, ones), dim=1)
    identity = torch.eye(design.shape[1], dtype=z.dtype)
    identity[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + ridge * identity, design.T @ targets)
    return design @ weights


@torch.no_grad()
def counterfactual_diagnostics(
    model: SmallCMNISTCNN,
    probe: CounterfactualProbe,
    *,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    red_z = model.encode(probe.red.to(device)).cpu()
    green_z = model.encode(probe.green.to(device)).cpu()
    red_logits = model.head(red_z.to(device)).cpu()
    green_logits = model.head(green_z.to(device)).cpu()
    red_prob = red_logits.softmax(dim=1)
    green_prob = green_logits.softmax(dim=1)

    pooled = torch.cat((red_z, green_z), dim=0)
    center, whitener = _whitener(pooled)
    red_white = (red_z - center) @ whitener
    green_white = (green_z - center) @ whitener
    color_delta = green_white - red_white
    color_energy = color_delta.square().sum(dim=1).mean()

    balanced = 0.5 * (red_white + green_white)
    task_means = torch.stack([balanced[probe.y == label].mean(dim=0) for label in (0, 1)])
    task_direction = task_means[1] - task_means[0]
    task_signal = task_direction.square().sum()
    task_unit = task_direction / task_direction.norm().clamp_min(1e-12)
    overlap = (color_delta @ task_unit).square().mean() / color_energy.clamp_min(1e-12)

    head_direction = (model.head.weight[1] - model.head.weight[0]).detach().cpu()
    raw_delta = green_z - red_z
    prediction_response = (raw_delta @ head_direction).square().mean()
    probability_response = (green_prob - red_prob).square().sum(dim=1).mean()
    raw_balanced = 0.5 * (red_z + green_z)
    raw_task_means = torch.stack(
        [raw_balanced[probe.y == label].mean(dim=0) for label in (0, 1)]
    )
    task_head_margin = torch.abs((raw_task_means[1] - raw_task_means[0]) @ head_direction)

    red_correct = red_logits.argmax(dim=1) == probe.y
    green_correct = green_logits.argmax(dim=1) == probe.y
    balanced_accuracy = torch.cat((red_correct, green_correct)).float().mean()
    color_consistency = (
        red_logits.argmax(dim=1) == green_logits.argmax(dim=1)
    ).float().mean()

    task_targets = torch.cat((probe.y, probe.y)).float().mul(2).sub(1)
    color_targets = torch.cat(
        (-torch.ones(probe.y.shape[0]), torch.ones(probe.y.shape[0]))
    )
    task_coordinate = _ridge_coordinates(pooled, task_targets)
    color_coordinate = _ridge_coordinates(pooled, color_targets)
    coordinates = torch.column_stack((task_coordinate, color_coordinate))
    coordinates = (coordinates - coordinates.mean(dim=0)) / coordinates.std(
        dim=0
    ).clamp_min(1e-6)
    n = probe.y.shape[0]

    metrics = {
        "latent_color_response": float(color_energy),
        "prediction_color_response": float(prediction_response),
        "probability_color_response": float(probability_response),
        "task_signal": float(task_signal),
        "task_color_overlap": float(overlap),
        "task_head_margin": float(task_head_margin),
        "balanced_accuracy": float(balanced_accuracy),
        "counterfactual_prediction_consistency": float(color_consistency),
    }
    paths = {
        "red": coordinates[:n].numpy(),
        "green": coordinates[n:].numpy(),
        "y": probe.y.numpy(),
        "digit": probe.digit.numpy(),
        "red_probability": red_prob[:, 1].numpy(),
        "green_probability": green_prob[:, 1].numpy(),
    }
    return metrics, paths


def mean_accuracy(
    model: SmallCMNISTCNN,
    environments: Iterable[ColoredEnvironment],
    device: torch.device,
) -> float:
    values = [accuracy(model, environment, device) for environment in environments]
    return float(np.mean(values))


def source_diagnostic_penalties(
    model: SmallCMNISTCNN,
    environments: tuple[ColoredEnvironment, ...],
    *,
    max_samples: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate every candidate source penalty on one fixed source-only subset."""
    batches = tuple(
        (environment.x[:max_samples].to(device), environment.y[:max_samples].to(device))
        for environment in environments
    )
    values = {}
    with torch.enable_grad():
        for method in ("l2", "irmv1", "coral"):
            penalty = regularizer_penalty(method, model, batches)
            values[f"diagnostic_{method}"] = float(penalty.detach().cpu())
    return values
