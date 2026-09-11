"""Compatibility wrappers around the per-algorithm objective files."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.methods import ObjectiveParts
from .algorithms.coral import coral_penalty
from .algorithms.registry import get_algorithm
from .algorithms.vrex import vrex_penalty_from_losses


def survey_method_objective(
    method: str,
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
    *,
    step: int,
    config: dict[str, Any],
) -> ObjectiveParts:
    """Return the source-only objective from the method's dedicated algorithm file."""
    return get_algorithm(method, config).objective(model, batches, step=step)
