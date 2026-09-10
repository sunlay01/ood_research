"""Method-independent primary source observation geometry."""

from __future__ import annotations

import torch
from torch import Tensor

from .smooth_world import SmoothWorlds, environment_parameters
from .task_response import augmented_head, encoded_outcomes, _environment_gradient


def observation_geometry(model: torch.nn.Module, worlds: SmoothWorlds) -> Tensor:
    """Return O_S with exactly three tangent columns and no learner-method branch."""
    source = tuple(encoded_outcomes(model, pool) for pool in worlds.source)
    weights = augmented_head(model).requires_grad_(True)
    theta = worlds.base_theta.detach().clone().requires_grad_(True)

    def state(value: Tensor) -> Tensor:
        gradients = []
        for environment, outcomes in enumerate(source):
            p, q = environment_parameters(value, environment=environment, evaluation=False)
            gradients.append(_environment_gradient(outcomes, p, q, weights))
        return torch.cat(gradients)

    return torch.autograd.functional.jacobian(state, theta).detach().cpu().double()


def independent_O_direction(model: torch.nn.Module, worlds: SmoothWorlds, direction: Tensor) -> Tensor:
    source = tuple(encoded_outcomes(model, pool) for pool in worlds.source)
    weights = augmented_head(model).requires_grad_(True)
    theta = worlds.base_theta.detach().clone().requires_grad_(True)

    def state(value: Tensor) -> Tensor:
        values = []
        for environment, outcomes in enumerate(source):
            p, q = environment_parameters(value, environment=environment, evaluation=False)
            values.append(_environment_gradient(outcomes, p, q, weights))
        return torch.cat(values)

    _, result = torch.autograd.functional.jvp(state, (theta,), (direction,))
    return result.detach().cpu().double()
