"""Fixed post-hoc functional measurement banks for the method panel."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.data import colorize_downsampled
from .smooth_world5 import SmoothPool, SmoothWorld5, balanced_indices


@dataclass(frozen=True)
class FunctionalBanks:
    source: Tensor
    counterfactual_red: Tensor
    counterfactual_green: Tensor
    clean_task: Tensor
    clean_labels: Tensor
    source_labels: Tensor
    source_env_ids: Tensor


def _colored(pool: SmoothPool, indices: Tensor, colors: Tensor) -> Tensor:
    return colorize_downsampled(pool.images[indices], colors.float(), image_subsample=2, normalize_pixels=True)


def _balanced_pool_indices(pool: SmoothPool, size: int) -> Tensor:
    return balanced_indices(pool.digits, size)


def build_functional_banks(worlds: SmoothWorld5, *, source_size_per_environment: int = 256, counterfactual_size: int = 512) -> FunctionalBanks:
    source_parts = []
    source_labels = []
    source_ids = []
    for env_id, pool in enumerate(worlds.source):
        indices = _balanced_pool_indices(pool, source_size_per_environment)
        clean = (pool.digits[indices] < 5).float()
        source_parts.append(_colored(pool, indices, clean))
        source_labels.append(clean)
        source_ids.append(torch.full_like(clean, env_id, dtype=torch.long))

    eval_pool = worlds.evaluation[0]
    indices = _balanced_pool_indices(eval_pool, counterfactual_size)
    gray = eval_pool.images[indices].reshape(-1, 28, 28)[:, ::2, ::2].float()
    if float(gray.max()) > 1.5:
        gray = gray / 255.0
    zeros = torch.zeros_like(gray)
    red = torch.stack((gray, zeros), dim=1).reshape(len(indices), -1).contiguous()
    green = torch.stack((zeros, gray), dim=1).reshape(len(indices), -1).contiguous()
    clean_task = (red + green) / 2.0
    clean_labels = (eval_pool.digits[indices] < 5).float()
    return FunctionalBanks(
        source=torch.cat(source_parts),
        counterfactual_red=red,
        counterfactual_green=green,
        clean_task=clean_task,
        clean_labels=clean_labels,
        source_labels=torch.cat(source_labels),
        source_env_ids=torch.cat(source_ids),
    )


@torch.no_grad()
def bank_logits(model: nn.Module, banks: FunctionalBanks, *, batch_size: int = 1024) -> dict[str, Tensor]:
    model.eval()
    result = {}
    for name, values in (("source", banks.source), ("counterfactual_red", banks.counterfactual_red), ("counterfactual_green", banks.counterfactual_green), ("clean_task", banks.clean_task)):
        result[name] = torch.cat([model(values[start:start + batch_size]).detach().cpu().reshape(-1) for start in range(0, values.shape[0], batch_size)])
    return result


def functional_response(model_plus: nn.Module, model_minus: nn.Module, banks: FunctionalBanks, *, delta: float) -> dict[str, float]:
    plus = bank_logits(model_plus, banks)
    minus = bank_logits(model_minus, banks)
    return {name: float(((plus[name] - minus[name]) / (2.0 * delta)).norm().item()) for name in plus}
