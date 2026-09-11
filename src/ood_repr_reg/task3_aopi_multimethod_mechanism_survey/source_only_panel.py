"""Standalone source-only panel for spectral and flatness algorithms.

This deliberately bypasses the DG survey trainer: it trains one algorithm on
fixed source batches and reports only source loss/accuracy and geometry.  It is
useful for diagnosing optimization and hyperparameters before any OOD replay.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch
from torch import Tensor, nn

from .algorithms.registry import get_algorithm


@dataclass(frozen=True)
class SourceOnlyResult:
    method: str
    steps: int
    final_loss: float
    source_accuracy: float
    finite: bool


def run_source_only(model: nn.Module, source_batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
                    method: str, config: dict[str, Any], *, steps: int = 200,
                    learning_rate: float | None = None) -> SourceOnlyResult:
    """Train a spectral/flatness method without the shared DG runner."""
    algorithm = get_algorithm(method, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate or float(config["training"]["learning_rate"]))
    state = algorithm.initial_state(seed=0)
    last = None
    for step in range(steps):
        result = algorithm.train_step(model, optimizer, source_batches, step=step,
                                     learning_rate=optimizer.param_groups[0]["lr"], algorithm_state=state)
        optimizer, state, last = result.optimizer, result.algorithm_state, result.parts
        if not bool(torch.isfinite(last.objective.detach())):
            return SourceOnlyResult(method, step + 1, float("nan"), 0.0, False)
    with torch.no_grad():
        logits = torch.cat([model(x) for x, _ in source_batches])
        labels = torch.cat([y for _, y in source_batches]).to(logits.device)
        accuracy = ((logits > 0).long() == labels.long()).float().mean().item()
    return SourceOnlyResult(method, steps, float(last.risk.detach()), accuracy, True)
