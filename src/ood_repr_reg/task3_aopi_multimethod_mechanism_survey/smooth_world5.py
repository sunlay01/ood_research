"""Five-dimensional smooth ColoredMNIST probability-displacement worlds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..cmnist_feature_probe import load_mnist_tensors
from ..task3_cmnist_cpu_minimal.data import colorize_downsampled


BASE = torch.tensor([0.2, 0.1, 0.9, 0.25, 0.25], dtype=torch.double)
BASIS = torch.eye(5, dtype=torch.double)
DERIVED_DIRECTIONS = torch.stack((
    torch.tensor([2**-0.5, 2**-0.5, 0.0, 0.0, 0.0], dtype=torch.double),
    torch.tensor([2**-0.5, -2**-0.5, 0.0, 0.0, 0.0], dtype=torch.double),
    torch.tensor([0.5, 0.5, 2**-0.5, 0.0, 0.0], dtype=torch.double),
    torch.tensor([-0.5, -0.5, 2**-0.5, 0.0, 0.0], dtype=torch.double),
    torch.tensor([0.0, 0.0, 0.0, 2**-0.5, 2**-0.5], dtype=torch.double),
    torch.tensor([0.0, 0.0, 0.0, -2**-0.5, 2**-0.5], dtype=torch.double),
))
OPAQUE_IDS = tuple(f"u{index:03d}" for index in range(11))
SEMANTIC_NAMES = (
    "source_env0_color", "source_env1_color", "evaluation_color",
    "source_label_noise", "evaluation_label_noise",
    "source_color_common", "source_color_contrast", "color_global",
    "color_shift", "noise_global", "noise_shift",
)


@dataclass(frozen=True)
class SmoothPool:
    images: Tensor
    digits: Tensor
    role: str

    def outcome_batches(self, indices: Tensor | None = None) -> tuple[tuple[Tensor, Tensor, int, int], ...]:
        images = self.images if indices is None else self.images[indices]
        digits = self.digits if indices is None else self.digits[indices]
        clean = (digits < 5).double().reshape(-1, 1)
        rows = []
        for label_flip in (0, 1):
            label = (clean - float(label_flip)).abs()
            for color_flip in (0, 1):
                color = (label.reshape(-1) - float(color_flip)).abs()
                colored = colorize_downsampled(images, color.float(), image_subsample=2, normalize_pixels=True)
                rows.append((colored, label, label_flip, color_flip))
        return tuple(rows)


@dataclass(frozen=True)
class SmoothWorld5:
    source: tuple[SmoothPool, SmoothPool]
    evaluation: tuple[SmoothPool, SmoothPool]
    base_delta: Tensor


def environment_parameters(delta: Tensor, *, environment: int, evaluation: bool) -> tuple[Tensor, Tensor]:
    if delta.shape != (5,):
        raise ValueError("survey displacement must have shape (5,)")
    if environment not in (0, 1):
        raise ValueError("environment must be 0 or 1")
    if evaluation:
        return BASE[2] + delta[2], BASE[4] + delta[4]
    return BASE[environment] + delta[environment], BASE[3] + delta[3]


def outcome_weight(p: Tensor, q: Tensor, label_flip: int, color_flip: int) -> Tensor:
    for name, value in (("color", p), ("label", q)):
        scalar = float(value.detach().cpu())
        if not 0.0 <= scalar <= 1.0:
            raise ValueError(f"{name} probability must be in [0,1], got {scalar}")
    return (q if label_flip else 1.0 - q) * (p if color_flip else 1.0 - p)


def all_direction_vectors() -> Tensor:
    return torch.cat((BASIS, DERIVED_DIRECTIONS), dim=0)


def balanced_indices(digits: Tensor, size: int) -> Tensor:
    if size <= 0 or size % 2:
        raise ValueError("balanced sample size must be a positive even integer")
    clean = digits < 5
    negative = torch.nonzero(~clean, as_tuple=False).reshape(-1)[: size // 2]
    positive = torch.nonzero(clean, as_tuple=False).reshape(-1)[: size // 2]
    if negative.numel() != size // 2 or positive.numel() != size // 2:
        raise ValueError("pool cannot satisfy requested clean-label balance")
    return torch.stack((positive, negative), dim=1).reshape(-1)


def subset_pool(pool: SmoothPool, size: int) -> SmoothPool:
    index = balanced_indices(pool.digits, size)
    return SmoothPool(pool.images[index], pool.digits[index], pool.role)


def build_smooth_world5(config: dict[str, Any], seed: int, *, data_root: Path | str, download: bool = False) -> SmoothWorld5:
    data = config["data"]
    declared = (float(data["source_color_flip_probs"][0]), float(data["source_color_flip_probs"][1]), float(data["target_color_flip_prob"]), float(data["label_noise"]), float(data["label_noise"]))
    if declared != tuple(float(value) for value in BASE):
        raise ValueError("configuration does not recover the fixed base world")
    gray, digits = load_mnist_tensors(Path(data_root), train=True, download=download)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(int(seed)))
    source_images, source_digits = gray[:50000][order], digits[:50000][order]
    evaluation_images, evaluation_digits = gray[50000:], digits[50000:]
    return SmoothWorld5(
        source=(
            SmoothPool(source_images[::2].cpu(), source_digits[::2].cpu(), "source_env0"),
            SmoothPool(source_images[1::2].cpu(), source_digits[1::2].cpu(), "source_env1"),
        ),
        evaluation=(
            SmoothPool(evaluation_images[::2].cpu(), evaluation_digits[::2].cpu(), "evaluation_env0"),
            SmoothPool(evaluation_images[1::2].cpu(), evaluation_digits[1::2].cpu(), "evaluation_env1"),
        ),
        base_delta=torch.zeros(5, dtype=torch.double),
    )
