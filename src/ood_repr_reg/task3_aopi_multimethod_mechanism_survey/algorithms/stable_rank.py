"""Stable-rank diagnostics for CMNIST rank probes."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


def stable_rank(matrix: Tensor) -> float:
    singular_values = torch.linalg.svdvals(matrix.detach().double())
    if singular_values.numel() == 0 or float(singular_values[0]) == 0.0:
        return 0.0
    return float(matrix.detach().double().square().sum().item() / singular_values[0].square().item())


def encoder_weight_spectrum_rows(model: nn.Module, *, seed: int, method: str, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    linear_index = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        singular_values = torch.linalg.svdvals(module.weight.detach().double())
        rows.append({
            "seed": int(seed),
            "method": method,
            "stage": stage,
            "module": module_name,
            "linear_index": linear_index,
            "penalized_by_weight_nuclear": bool(module_name.startswith("encoder")),
            "singular_value_max": float(singular_values.max()) if singular_values.numel() else 0.0,
            "singular_value_min": float(singular_values.min()) if singular_values.numel() else 0.0,
            "nuclear_norm": float(singular_values.sum()),
            "frobenius_norm": float(module.weight.detach().double().norm()),
            "stable_rank": stable_rank(module.weight),
            "diagnostic_only": True,
        })
        linear_index += 1
    return rows
