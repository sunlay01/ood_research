"""Source-only end-to-end penalties for Task 3 CMNIST."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .curvature import CurvatureDiagnostics, local_response_penalty


METHODS = (
    "ERM",
    "IRMv1",
    "V-REx",
    "UNPRECONDITIONED_GRAD_ALIGN",
    "LOCAL_RESPONSE",
    "RANDOM_METRIC",
    "SHUFFLED_LOCAL_RESPONSE",
    "RESP2",
)


def shuffle_environment_batches(
    batches: tuple[tuple[Tensor, Tensor], ...],
    generator: torch.Generator,
) -> tuple[tuple[Tensor, Tensor], ...]:
    sizes = [x.shape[0] for x, _ in batches]
    all_x = torch.cat([x for x, _ in batches], dim=0)
    all_y = torch.cat([y for _, y in batches], dim=0)
    order = torch.randperm(all_x.shape[0], generator=generator, device=all_x.device)
    all_x = all_x[order]
    all_y = all_y[order]
    result = []
    start = 0
    for size in sizes:
        stop = start + size
        result.append((all_x[start:stop], all_y[start:stop]))
        start = stop
    return tuple(result)


def mean_source_loss(model: nn.Module, batches: tuple[tuple[Tensor, Tensor], ...]) -> Tensor:
    return torch.stack([F.cross_entropy(model(x), y) for x, y in batches]).mean()


def irmv1_penalty(model: nn.Module, batches: tuple[tuple[Tensor, Tensor], ...]) -> Tensor:
    terms = []
    for x, y in batches:
        scale = torch.ones((), device=x.device, requires_grad=True)
        risk = F.cross_entropy(model(x) * scale, y)
        gradient = torch.autograd.grad(risk, scale, create_graph=True)[0]
        terms.append(gradient.square())
    return torch.stack(terms).mean()


def vrex_penalty(model: nn.Module, batches: tuple[tuple[Tensor, Tensor], ...]) -> Tensor:
    risks = torch.stack([F.cross_entropy(model(x), y) for x, y in batches])
    centered = risks - risks.mean()
    return centered.square().mean()


def task3_penalty(
    method: str,
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], ...],
    *,
    epsilon: float,
    random_seed: int,
    shuffle_generator: torch.Generator | None = None,
) -> tuple[Tensor, dict[str, float | str]]:
    key = method.upper()
    if key == "ERM":
        return next(model.parameters()).new_zeros(()), {}
    if key == "IRMV1":
        return irmv1_penalty(model, batches), {}
    if key == "V-REX":
        return vrex_penalty(model, batches), {}
    if key == "UNPRECONDITIONED_GRAD_ALIGN":
        penalty, diagnostic = local_response_penalty(
            model, batches, epsilon=epsilon, metric_kind="identity"
        )
        return penalty, asdict(diagnostic)
    if key == "LOCAL_RESPONSE":
        penalty, diagnostic = local_response_penalty(model, batches, epsilon=epsilon)
        return penalty, asdict(diagnostic)
    if key == "RANDOM_METRIC":
        penalty, diagnostic = local_response_penalty(
            model, batches, epsilon=epsilon, metric_kind="random", random_seed=random_seed
        )
        return penalty, asdict(diagnostic)
    if key == "SHUFFLED_LOCAL_RESPONSE":
        if shuffle_generator is None:
            raise ValueError("shuffle_generator is required for shuffled environment control")
        shuffled = shuffle_environment_batches(batches, shuffle_generator)
        penalty, diagnostic = local_response_penalty(model, shuffled, epsilon=epsilon)
        return penalty, asdict(diagnostic)
    if key == "RESP2":
        penalty, diagnostic = local_response_penalty(
            model, batches, epsilon=epsilon, response_squared=True
        )
        return penalty, asdict(diagnostic)
    raise ValueError(f"unknown Task 3 method: {method}")


def diagnostic_penalties(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], ...],
    *,
    epsilon: float,
    random_seed: int,
) -> dict[str, float | str]:
    with torch.enable_grad():
        _, grad_diag = local_response_penalty(model, batches, epsilon=epsilon, metric_kind="identity")
        _, lr_diag = local_response_penalty(model, batches, epsilon=epsilon, metric_kind="real")
        _, random_diag = local_response_penalty(
            model, batches, epsilon=epsilon, metric_kind="random", random_seed=random_seed
        )
    result = {f"grad_{key}": value for key, value in asdict(grad_diag).items()}
    result.update({f"local_{key}": value for key, value in asdict(lr_diag).items()})
    result.update({f"random_{key}": value for key, value in asdict(random_diag).items()})
    return result
