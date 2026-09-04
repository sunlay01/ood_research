"""Synthetic multi-domain, multi-task data with explicit core/spurious factors."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class Batch:
    x: Tensor
    y: Tensor
    task: int
    env: int


@dataclass(frozen=True)
class DatasetBundle:
    train: tuple[Batch, ...]
    source_eval: tuple[Batch, ...]
    target_domain_eval: tuple[Batch, ...]
    target_task_support: tuple[Batch, ...]
    target_task_source_eval: tuple[Batch, ...]
    joint_eval: tuple[Batch, ...]
    input_mean: Tensor
    input_std: Tensor


SOURCE_TASK_BETAS = (
    torch.tensor([1.0, 0.35]),
    torch.tensor([0.35, 1.0]),
)
TARGET_TASK_BETA = torch.tensor([1.0, -1.0]) / (2.0**0.5)
SOURCE_ENVS = ((0.90, -0.75), (0.60, 0.75))
TARGET_ENV = (-0.90, 1.50)


def _make_raw_batch(
    *,
    beta: Tensor,
    rho: float,
    domain_mean: float,
    n: int,
    task: int,
    env: int,
    generator: torch.Generator,
) -> Batch:
    core = torch.randn((n, 2), generator=generator)
    label_score = core @ beta + 0.25 * torch.randn(n, generator=generator)
    y = torch.where(label_score >= 0, 1.0, -1.0)
    spur_noise_scale = max(1.0 - rho**2, 0.0) ** 0.5
    spurious = rho * y + spur_noise_scale * torch.randn(n, generator=generator)
    domain_feature = domain_mean + 0.50 * torch.randn(n, generator=generator)
    nuisance = torch.randn(n, generator=generator)
    x = torch.column_stack((core, spurious, domain_feature, nuisance))
    return Batch(x=x, y=y, task=task, env=env)


def _standardize(batch: Batch, mean: Tensor, std: Tensor) -> Batch:
    return replace(batch, x=(batch.x - mean) / std)


def _build_batches(
    *,
    betas: tuple[Tensor, ...],
    environments: tuple[tuple[float, float], ...],
    n: int,
    generator: torch.Generator,
    task_offset: int = 0,
    env_offset: int = 0,
) -> tuple[Batch, ...]:
    batches: list[Batch] = []
    for task_index, beta in enumerate(betas):
        for env_index, (rho, domain_mean) in enumerate(environments):
            batches.append(
                _make_raw_batch(
                    beta=beta,
                    rho=rho,
                    domain_mean=domain_mean,
                    n=n,
                    task=task_index + task_offset,
                    env=env_index + env_offset,
                    generator=generator,
                )
            )
    return tuple(batches)


def make_dataset(
    *, seed: int, n_train: int, n_eval: int, n_probe: int
) -> DatasetBundle:
    """Build fixed train/evaluation splits and standardize using source train only."""
    generator = torch.Generator().manual_seed(seed)
    train = _build_batches(
        betas=SOURCE_TASK_BETAS,
        environments=SOURCE_ENVS,
        n=n_train,
        generator=generator,
    )
    source_eval = _build_batches(
        betas=SOURCE_TASK_BETAS,
        environments=SOURCE_ENVS,
        n=n_eval,
        generator=generator,
    )
    target_domain_eval = _build_batches(
        betas=SOURCE_TASK_BETAS,
        environments=(TARGET_ENV,),
        n=n_eval,
        generator=generator,
        env_offset=len(SOURCE_ENVS),
    )
    target_task_support = _build_batches(
        betas=(TARGET_TASK_BETA,),
        environments=SOURCE_ENVS,
        n=n_probe,
        generator=generator,
        task_offset=len(SOURCE_TASK_BETAS),
    )
    target_task_source_eval = _build_batches(
        betas=(TARGET_TASK_BETA,),
        environments=SOURCE_ENVS,
        n=n_eval,
        generator=generator,
        task_offset=len(SOURCE_TASK_BETAS),
    )
    joint_eval = _build_batches(
        betas=(TARGET_TASK_BETA,),
        environments=(TARGET_ENV,),
        n=n_eval,
        generator=generator,
        task_offset=len(SOURCE_TASK_BETAS),
        env_offset=len(SOURCE_ENVS),
    )

    train_x = torch.cat([batch.x for batch in train], dim=0)
    input_mean = train_x.mean(dim=0)
    input_std = train_x.std(dim=0).clamp_min(1e-6)

    return DatasetBundle(
        train=tuple(_standardize(batch, input_mean, input_std) for batch in train),
        source_eval=tuple(
            _standardize(batch, input_mean, input_std) for batch in source_eval
        ),
        target_domain_eval=tuple(
            _standardize(batch, input_mean, input_std) for batch in target_domain_eval
        ),
        target_task_support=tuple(
            _standardize(batch, input_mean, input_std) for batch in target_task_support
        ),
        target_task_source_eval=tuple(
            _standardize(batch, input_mean, input_std)
            for batch in target_task_source_eval
        ),
        joint_eval=tuple(_standardize(batch, input_mean, input_std) for batch in joint_eval),
        input_mean=input_mean,
        input_std=input_std,
    )
