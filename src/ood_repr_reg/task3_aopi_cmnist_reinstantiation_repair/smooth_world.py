"""Smooth empirical ColoredMNIST worlds with exact binary-outcome weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..cmnist_feature_probe import load_mnist_tensors
from ..task3_cmnist_cpu_minimal.data import colorize_downsampled


BASIS = {
    "source_env0_color": torch.tensor([1.0, 0.0, 0.0], dtype=torch.double),
    "source_env1_color": torch.tensor([0.0, 1.0, 0.0], dtype=torch.double),
    "shared_label_noise": torch.tensor([0.0, 0.0, 1.0], dtype=torch.double),
}
DERIVED_DIRECTIONS = {
    "common_source_color": torch.tensor([2.0 ** -0.5, 2.0 ** -0.5, 0.0], dtype=torch.double),
    "antisymmetric_source_color": torch.tensor([2.0 ** -0.5, -2.0 ** -0.5, 0.0], dtype=torch.double),
}
SOURCE_COLOR_BASES = (0.2, 0.1)
EVALUATION_COLOR_BASE = 0.9
LABEL_NOISE_BASE = 0.25


@dataclass(frozen=True)
class SmoothPool:
    """Raw samples; the four label/color outcomes are generated deterministically."""

    images: Tensor
    digits: Tensor
    role: str

    def outcome_batches(self, indices: Tensor | None = None) -> tuple[tuple[Tensor, Tensor, int, int], ...]:
        images = self.images if indices is None else self.images[indices]
        digits = self.digits if indices is None else self.digits[indices]
        clean = (digits < 5).double().reshape(-1, 1)
        result: list[tuple[Tensor, Tensor, int, int]] = []
        for label_flip in (0, 1):
            label = (clean - float(label_flip)).abs()
            for color_flip in (0, 1):
                color = (label.reshape(-1) - float(color_flip)).abs()
                colored = colorize_downsampled(images, color.float(), image_subsample=2, normalize_pixels=True)
                result.append((colored, label, label_flip, color_flip))
        return tuple(result)


@dataclass(frozen=True)
class SmoothWorlds:
    source: tuple[SmoothPool, SmoothPool]
    evaluation: tuple[SmoothPool, SmoothPool]
    base_theta: Tensor


def outcome_weight(p: Tensor, q: Tensor, label_flip: int, color_flip: int) -> Tensor:
    for name, probability in (("color_flip", p), ("label_noise", q)):
        value = float(probability.detach().cpu())
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} probability must be in [0, 1], got {value}")
    label_probability = q if label_flip else 1.0 - q
    color_probability = p if color_flip else 1.0 - p
    weight = label_probability * color_probability
    if not 0.0 <= float(weight.detach().cpu()) <= 1.0:
        raise ValueError("mixture weight must be a valid probability")
    return weight


def environment_parameters(delta: Tensor, *, environment: int, evaluation: bool) -> tuple[Tensor, Tensor]:
    """Map a three-dimensional displacement from the fixed CMNIST base world."""
    if delta.shape != (3,):
        raise ValueError("the repaired audit tangent space is exactly R^3")
    color_base = EVALUATION_COLOR_BASE if evaluation else SOURCE_COLOR_BASES[environment]
    return color_base + delta[environment], LABEL_NOISE_BASE + delta[2]


def base_world_identity(worlds: SmoothWorlds) -> bool:
    delta = worlds.base_theta
    source = [environment_parameters(delta, environment=index, evaluation=False) for index in range(2)]
    evaluation = [environment_parameters(delta, environment=index, evaluation=True) for index in range(2)]
    observed = [float(source[0][0]), float(source[1][0]), float(source[0][1]), float(evaluation[0][0]), float(evaluation[1][0]), float(evaluation[0][1])]
    expected = [0.2, 0.1, 0.25, 0.9, 0.9, 0.25]
    return bool(torch.allclose(torch.tensor(observed, dtype=torch.double), torch.tensor(expected, dtype=torch.double), atol=0.0, rtol=0.0))


class SmoothWorldFactory:
    def __init__(self, config: dict[str, Any], seed: int, *, data_root: Path | str, download: bool = False) -> None:
        data = config["data"]
        if tuple(float(value) for value in data["source_color_flip_probs"]) != SOURCE_COLOR_BASES:
            raise ValueError("repair requires source color bases (0.2, 0.1)")
        if float(data["target_color_flip_prob"]) != EVALUATION_COLOR_BASE or float(data["label_noise"]) != LABEL_NOISE_BASE:
            raise ValueError("repair requires target color base 0.9 and label-noise base 0.25")
        gray, digits = load_mnist_tensors(Path(data_root), train=True, download=download)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(int(seed)))
        source_images, source_digits = gray[:50000][order], digits[:50000][order]
        evaluation_images, evaluation_digits = gray[50000:], digits[50000:]
        self.worlds = SmoothWorlds(
            source=(
                SmoothPool(source_images[::2].cpu(), source_digits[::2].cpu(), "source_env0"),
                SmoothPool(source_images[1::2].cpu(), source_digits[1::2].cpu(), "source_env1"),
            ),
            evaluation=(
                SmoothPool(evaluation_images[::2].cpu(), evaluation_digits[::2].cpu(), "evaluation_env0"),
                SmoothPool(evaluation_images[1::2].cpu(), evaluation_digits[1::2].cpu(), "evaluation_env1"),
            ),
            base_theta=torch.zeros(3, dtype=torch.double),
        )

    def build(self) -> SmoothWorlds:
        return self.worlds
