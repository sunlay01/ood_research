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
    target_accuracy: float | None
    target_color_agreement: float | None
    finite: bool


def run_source_only(model: nn.Module, source_batches: tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]],
                    method: str, config: dict[str, Any], *, steps: int = 200,
                    learning_rate: float | None = None,
                    target_batch: tuple[Tensor, Tensor] | None = None,
                    target_colors: Tensor | None = None,
                    batch_provider=None) -> SourceOnlyResult:
    """Train a spectral/flatness method without the shared DG runner."""
    algorithm = get_algorithm(method, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate or float(config["training"]["learning_rate"]))
    state = algorithm.initial_state(seed=0)
    last = None
    for step in range(steps):
        batches = batch_provider(step) if batch_provider is not None else source_batches
        result = algorithm.train_step(model, optimizer, batches, step=step,
                                     learning_rate=optimizer.param_groups[0]["lr"], algorithm_state=state)
        optimizer, state, last = result.optimizer, result.algorithm_state, result.parts
        if not bool(torch.isfinite(last.objective.detach())):
            return SourceOnlyResult(method, step + 1, float("nan"), 0.0, None, None, False)
    with torch.no_grad():
        logits = torch.cat([model(x) for x, _ in source_batches])
        labels = torch.cat([y for _, y in source_batches]).to(logits.device)
        accuracy = ((logits > 0).long() == labels.long()).float().mean().item()
        target_accuracy = target_agreement = None
        if target_batch is not None:
            tx, ty = target_batch
            pred = (model(tx) > 0).long().view(-1)
            target_accuracy = (pred == ty.long().view(-1)).float().mean().item()
            if target_colors is not None:
                target_agreement = (pred == target_colors.long().view(-1)).float().mean().item()
    return SourceOnlyResult(method, steps, float(last.risk.detach()), accuracy,
                            target_accuracy, target_agreement, True)
