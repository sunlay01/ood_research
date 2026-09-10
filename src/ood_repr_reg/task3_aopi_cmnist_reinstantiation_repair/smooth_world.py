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
    label_probability = q if label_flip else 1.0 - q
    color_probability = p if color_flip else 1.0 - p
    return label_probability * color_probability


def environment_parameters(theta: Tensor, *, environment: int, evaluation: bool) -> tuple[Tensor, Tensor]:
    """Map the declared three-dimensional tangent coordinate to one environment."""
    if theta.shape != (3,):
        raise ValueError("the repaired audit tangent space is exactly R^3")
    base = 0.9 if evaluation else (0.2 if environment == 0 else 0.1)
    return theta[environment] + base, theta[2]


class SmoothWorldFactory:
    def __init__(self, config: dict[str, Any], seed: int, *, data_root: Path | str, download: bool = False) -> None:
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
            base_theta=torch.tensor([0.2, 0.1, float(config["data"]["label_noise"])], dtype=torch.double),
        )

    def build(self) -> SmoothWorlds:
        return self.worlds
