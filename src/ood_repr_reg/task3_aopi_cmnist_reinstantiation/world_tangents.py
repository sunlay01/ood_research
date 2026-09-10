"""Deterministic source and evaluation worlds for the A/O/Pi audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..cmnist_feature_probe import load_mnist_tensors
from ..task3_cmnist_cpu_minimal.data import ColoredEnvironment, colorize_downsampled


EPSILONS = (0.01, 0.02)
BASE_DIMENSION = 3
REPORT_DIRECTIONS: dict[str, Tensor] = {
    "source_env0_color": torch.tensor([1.0, 0.0, 0.0], dtype=torch.double),
    "source_env1_color": torch.tensor([0.0, 1.0, 0.0], dtype=torch.double),
    "shared_label_noise": torch.tensor([0.0, 0.0, 1.0], dtype=torch.double),
    "common_source_color": torch.tensor([2.0 ** -0.5, 2.0 ** -0.5, 0.0], dtype=torch.double),
    "antisymmetric_source_color": torch.tensor([2.0 ** -0.5, -2.0 ** -0.5, 0.0], dtype=torch.double),
}


@dataclass(frozen=True)
class WorldParameters:
    source_flip_probs: tuple[float, float]
    label_noise: float


@dataclass(frozen=True)
class TangentWorlds:
    source_envs: tuple[ColoredEnvironment, ColoredEnvironment]
    evaluation_envs: tuple[ColoredEnvironment, ColoredEnvironment]
    parameters: WorldParameters
    seed: int


@dataclass(frozen=True)
class _RawPool:
    images: Tensor
    digits: Tensor
    label_uniform: Tensor
    color_uniform: Tensor
    role: str


def _make_pool(images: Tensor, digits: Tensor, *, role: str, generator: torch.Generator) -> _RawPool:
    size = int(len(digits))
    return _RawPool(
        images=images.detach().cpu(),
        digits=digits.detach().cpu(),
        # Keep float32 draws to reproduce CPU-minimal make_environment at zero.
        label_uniform=torch.rand(size, generator=generator),
        color_uniform=torch.rand(size, generator=generator),
        role=role,
    )


def _environment(pool: _RawPool, *, color_flip: float, label_noise: float) -> ColoredEnvironment:
    if not 0.0 <= color_flip <= 1.0 or not 0.0 <= label_noise <= 1.0:
        raise ValueError("tangent probabilities must remain in [0, 1]")
    clean = (pool.digits < 5).double()
    labels = torch.abs(clean - (pool.label_uniform < label_noise).double())
    colors = torch.abs(labels - (pool.color_uniform < color_flip).double())
    return ColoredEnvironment(
        images=colorize_downsampled(pool.images, colors.float(), image_subsample=2, normalize_pixels=True),
        labels=labels[:, None].float(),
        digits=pool.digits.clone(),
        colors=colors.float(),
        color_flip_prob=float(color_flip),
        role=pool.role,
    )


class WorldFactory:
    """Creates paired worlds with fixed uniforms, ensuring centered differences."""

    def __init__(self, config: dict[str, Any], seed: int, *, data_root: Path | str, download: bool = False) -> None:
        data = config["data"]
        gray, digits = load_mnist_tensors(Path(data_root), train=True, download=download)
        order = torch.randperm(50000, generator=torch.Generator().manual_seed(int(seed)))
        source_images, source_digits = gray[:50000][order], digits[:50000][order]
        eval_images, eval_digits = gray[50000:], digits[50000:]
        # This draw order matches build_task3_data exactly at delta=0.
        generator = torch.Generator().manual_seed(int(seed) + 314159)
        self.source_pools = (
            _make_pool(source_images[::2], source_digits[::2], role="source_env0", generator=generator),
            _make_pool(source_images[1::2], source_digits[1::2], role="source_env1", generator=generator),
        )
        target_pool = _make_pool(eval_images, eval_digits, role="evaluation", generator=generator)
        self.evaluation_pools = (
            _RawPool(target_pool.images[::2], target_pool.digits[::2], target_pool.label_uniform[::2], target_pool.color_uniform[::2], "evaluation_env0"),
            _RawPool(target_pool.images[1::2], target_pool.digits[1::2], target_pool.label_uniform[1::2], target_pool.color_uniform[1::2], "evaluation_env1"),
        )
        self.base = WorldParameters(tuple(float(x) for x in data["source_color_flip_probs"]), float(data["label_noise"]))
        self.seed = int(seed)

    def build(self, delta: Tensor | None = None) -> TangentWorlds:
        shift = torch.zeros(BASE_DIMENSION, dtype=torch.double) if delta is None else delta.detach().cpu().double()
        if tuple(shift.shape) != (BASE_DIMENSION,):
            raise ValueError("world delta must have dimension three")
        flips = (self.base.source_flip_probs[0] + float(shift[0]), self.base.source_flip_probs[1] + float(shift[1]))
        label_noise = self.base.label_noise + float(shift[2])
        source = (_environment(self.source_pools[0], color_flip=flips[0], label_noise=label_noise), _environment(self.source_pools[1], color_flip=flips[1], label_noise=label_noise))
        # Evaluation worlds mirror the source color coordinates but remain held out.
        evaluation = (_environment(self.evaluation_pools[0], color_flip=0.9 + float(shift[0]), label_noise=label_noise), _environment(self.evaluation_pools[1], color_flip=0.9 + float(shift[1]), label_noise=label_noise))
        return TangentWorlds(source_envs=source, evaluation_envs=evaluation, parameters=WorldParameters(flips, label_noise), seed=self.seed)


def centered_pair(factory: WorldFactory, direction: Tensor, epsilon: float) -> tuple[TangentWorlds, TangentWorlds]:
    vector = direction.detach().cpu().double()
    return factory.build(vector * float(epsilon)), factory.build(-vector * float(epsilon))


def tangent_manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, vector in REPORT_DIRECTIONS.items():
        rows.append({"tangent": name, "basis_coordinates": ";".join(f"{float(value):.12g}" for value in vector), "base_dimension": BASE_DIMENSION, "epsilons": ";".join(str(value) for value in EPSILONS), "implementation": "common_random_numbers_probability_threshold"})
    return rows
