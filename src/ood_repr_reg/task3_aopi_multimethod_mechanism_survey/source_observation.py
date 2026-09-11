"""Method-independent source observability geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .smooth_world5 import SmoothWorld5, subset_pool
from .task_response import augmented_head, encoded_outcomes, environment_gradient


@dataclass(frozen=True)
class ObservationGeometry:
    O: Tensor
    source_state: Tensor
    target_only_column_norms: tuple[float, float]
    rank: int
    kernel_dimension: int


def numerical_rank(matrix: Tensor, tolerance: float = 1e-8) -> int:
    values = torch.linalg.svdvals(matrix)
    if values.numel() == 0:
        return 0
    threshold = max(float(values.max()), 1e-12) * tolerance
    return int((values > threshold).sum())


def observation_geometry(model: nn.Module, worlds: SmoothWorld5, *, pool_size: int = 1024) -> ObservationGeometry:
    """Build O from only source risks; the function deliberately has no method input."""
    pools = tuple(subset_pool(pool, pool_size) for pool in worlds.source)
    outcomes = tuple(encoded_outcomes(model, pool) for pool in pools)
    weights = augmented_head(model).requires_grad_(True)
    delta = worlds.base_delta.detach().clone().requires_grad_(True)
    source_state = torch.stack([
        environment_gradient(outcomes[environment], delta, weights, environment=environment)
        for environment in range(2)
    ]).reshape(-1)
    # Each source environment has only its own color and shared source-noise coordinates.
    columns = []
    for index in range(5):
        direction = torch.zeros(5, dtype=torch.double)
        direction[index] = 1.0
        values = []
        for environment in range(2):
            values.append(
                torch.autograd.functional.jvp(
                    lambda value: environment_gradient(outcomes[environment], value, weights, environment=environment),
                    (delta,), (direction,),
                )[1]
            )
        columns.append(torch.cat(values))
    matrix = torch.stack(columns, dim=1).detach()
    rank = numerical_rank(matrix)
    if rank > 3:
        raise ValueError(f"source observability rank {rank} exceeds three source coordinates")
    return ObservationGeometry(matrix, source_state.detach(), (float(matrix[:, 2].norm()), float(matrix[:, 4].norm())), rank, 5 - rank)
