"""Data construction for the CPU-minimal ColoredMNIST probe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..cmnist_feature_probe import load_mnist_tensors


@dataclass(frozen=True)
class ColoredEnvironment:
    images: Tensor
    labels: Tensor
    digits: Tensor
    colors: Tensor
    color_flip_prob: float
    role: str


@dataclass(frozen=True)
class BatchSchedule:
    indices: tuple[Tensor, Tensor]
    seed: int
    steps: int
    batch_size_per_environment: int


@dataclass(frozen=True)
class ColoredMNISTData:
    source_envs: tuple[ColoredEnvironment, ColoredEnvironment]
    target_env: ColoredEnvironment
    batch_schedule: BatchSchedule
    seed: int


def binary_labels_from_digits(digits: Tensor) -> Tensor:
    """Facebook IRM binary label: one for digits below five."""
    return (digits < 5).float()


def xor_float(left: Tensor, right: Tensor) -> Tensor:
    return (left.float() - right.float()).abs()


def bernoulli_mask(probability: float, size: int, generator: torch.Generator) -> Tensor:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    return (torch.rand(size, generator=generator) < probability).float()


def noisy_binary_labels(digits: Tensor, label_noise: float, generator: torch.Generator) -> Tensor:
    base = binary_labels_from_digits(digits)
    flips = bernoulli_mask(label_noise, int(base.numel()), generator)
    return xor_float(base, flips)


def colors_from_labels(labels: Tensor, color_flip_prob: float, generator: torch.Generator) -> Tensor:
    flips = bernoulli_mask(color_flip_prob, int(labels.numel()), generator)
    return xor_float(labels.reshape(-1), flips)


def colorize_downsampled(
    images: Tensor,
    colors: Tensor,
    *,
    image_subsample: int,
    normalize_pixels: bool,
) -> Tensor:
    if image_subsample != 2:
        raise ValueError("TASK3-CMNIST-CPU-MINIMAL fixes image_subsample=2")
    gray = images.reshape((-1, 28, 28))[:, ::image_subsample, ::image_subsample].float()
    if normalize_pixels and float(gray.max()) > 1.5:
        gray = gray / 255.0
    colored = torch.stack([gray, gray], dim=1)
    colored[torch.arange(colored.shape[0]), (1 - colors.long()).reshape(-1), :, :] = 0.0
    return colored.reshape(colored.shape[0], -1).contiguous()


def make_environment(
    images: Tensor,
    digits: Tensor,
    *,
    color_flip_prob: float,
    label_noise: float,
    role: str,
    image_subsample: int,
    normalize_pixels: bool,
    generator: torch.Generator,
) -> ColoredEnvironment:
    labels = noisy_binary_labels(digits, label_noise, generator)
    colors = colors_from_labels(labels, color_flip_prob, generator)
    return ColoredEnvironment(
        images=colorize_downsampled(
            images,
            colors,
            image_subsample=image_subsample,
            normalize_pixels=normalize_pixels,
        ),
        labels=labels[:, None].float(),
        digits=digits.clone(),
        colors=colors.float(),
        color_flip_prob=float(color_flip_prob),
        role=role,
    )


def make_batch_schedule(
    *,
    source_pool_sizes: tuple[int, int],
    steps: int,
    batch_size_per_environment: int,
    seed: int,
) -> BatchSchedule:
    generator = torch.Generator().manual_seed(int(seed) + 90731)
    schedules = []
    for size in source_pool_sizes:
        schedules.append(torch.randint(size, (steps, batch_size_per_environment), generator=generator))
    return BatchSchedule(
        indices=(schedules[0].long(), schedules[1].long()),
        seed=int(seed),
        steps=int(steps),
        batch_size_per_environment=int(batch_size_per_environment),
    )


def scheduled_source_batches(
    source_envs: tuple[ColoredEnvironment, ColoredEnvironment],
    schedule: BatchSchedule,
    step: int,
    device: torch.device | str = "cpu",
) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
    batches = []
    for env, indices in zip(source_envs, schedule.indices):
        selected = indices[int(step)]
        batches.append((env.images[selected].to(device), env.labels[selected].to(device)))
    return (batches[0], batches[1])


def build_task3_data(
    config: dict[str, Any],
    seed: int,
    *,
    data_root: Path | str = "data",
    download: bool = True,
) -> ColoredMNISTData:
    data_cfg = config["data"]
    training_cfg = config["training"]
    gray, digits = load_mnist_tensors(Path(data_root), train=True, download=download)
    first = gray[:50000]
    first_digits = digits[:50000]
    target_images = gray[50000:]
    target_digits = digits[50000:]

    order = torch.randperm(first.shape[0], generator=torch.Generator().manual_seed(int(seed)))
    first = first[order]
    first_digits = first_digits[order]

    noise_generator = torch.Generator().manual_seed(int(seed) + 314159)
    source_env0 = make_environment(
        first[::2],
        first_digits[::2],
        color_flip_prob=float(data_cfg["source_color_flip_probs"][0]),
        label_noise=float(data_cfg["label_noise"]),
        role="source_env0_flip_0p2",
        image_subsample=int(data_cfg["image_subsample"]),
        normalize_pixels=bool(data_cfg["normalize_pixels"]),
        generator=noise_generator,
    )
    source_env1 = make_environment(
        first[1::2],
        first_digits[1::2],
        color_flip_prob=float(data_cfg["source_color_flip_probs"][1]),
        label_noise=float(data_cfg["label_noise"]),
        role="source_env1_flip_0p1",
        image_subsample=int(data_cfg["image_subsample"]),
        normalize_pixels=bool(data_cfg["normalize_pixels"]),
        generator=noise_generator,
    )
    target_env = make_environment(
        target_images,
        target_digits,
        color_flip_prob=float(data_cfg["target_color_flip_prob"]),
        label_noise=float(data_cfg["label_noise"]),
        role="target_flip_0p9",
        image_subsample=int(data_cfg["image_subsample"]),
        normalize_pixels=bool(data_cfg["normalize_pixels"]),
        generator=noise_generator,
    )
    schedule = make_batch_schedule(
        source_pool_sizes=(source_env0.images.shape[0], source_env1.images.shape[0]),
        steps=int(training_cfg["steps"]),
        batch_size_per_environment=int(training_cfg["batch_size_per_environment"]),
        seed=int(seed),
    )
    return ColoredMNISTData(
        source_envs=(source_env0, source_env1),
        target_env=target_env,
        batch_schedule=schedule,
        seed=int(seed),
    )
