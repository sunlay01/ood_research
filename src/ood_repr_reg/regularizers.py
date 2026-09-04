"""Regularizers compared in the exploratory sweep."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import torch
from torch import Tensor, nn

from .synthetic import Batch


class LinearRepresentation(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, n_tasks: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.heads = nn.Parameter(torch.empty(n_tasks, latent_dim))
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.normal_(self.heads, mean=0.0, std=0.25)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def predict(self, batch: Batch) -> Tensor:
        return self.encode(batch.x) @ self.heads[batch.task]


def mean_source_risk(model: LinearRepresentation, batches: Iterable[Batch]) -> Tensor:
    losses = [torch.mean((model.predict(batch) - batch.y) ** 2) for batch in batches]
    return torch.stack(losses).mean()


def _by_task(batches: Iterable[Batch]) -> dict[int, list[Batch]]:
    grouped: dict[int, list[Batch]] = defaultdict(list)
    for batch in batches:
        grouped[batch.task].append(batch)
    return grouped


def _covariance(z: Tensor) -> Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(z.shape[0] - 1, 1)


def _mmd_rbf(x: Tensor, y: Tensor) -> Tensor:
    sigmas = (0.5, 1.0, 2.0, 4.0)
    xx = torch.cdist(x, x).square()
    yy = torch.cdist(y, y).square()
    xy = torch.cdist(x, y).square()
    values = []
    for sigma in sigmas:
        denom = 2.0 * sigma**2
        values.append(
            torch.exp(-xx / denom).mean()
            + torch.exp(-yy / denom).mean()
            - 2.0 * torch.exp(-xy / denom).mean()
        )
    return torch.stack(values).mean()


def head_gradient(model: LinearRepresentation, batch: Batch) -> Tensor:
    z = model.encode(batch.x)
    residual = z @ model.heads[batch.task] - batch.y
    return 2.0 * z.T @ residual / batch.x.shape[0]


def head_hessian(model: LinearRepresentation, batch: Batch) -> Tensor:
    z = model.encode(batch.x)
    return 2.0 * z.T @ z / batch.x.shape[0]


def regularizer_value(
    name: str,
    model: LinearRepresentation,
    batches: tuple[Batch, ...],
    *,
    mmd_max_samples: int,
) -> Tensor:
    if name == "l1":
        return torch.cat((model.encoder.weight.flatten(), model.heads.flatten())).abs().mean()
    if name == "l2":
        params = torch.cat((model.encoder.weight.flatten(), model.heads.flatten()))
        return params.square().mean()

    grouped = _by_task(batches)
    task_penalties: list[Tensor] = []
    for task_batches in grouped.values():
        if len(task_batches) != 2:
            raise ValueError("The exploratory regularizers require exactly two source domains.")
        left, right = sorted(task_batches, key=lambda batch: batch.env)
        if name == "irmv1":
            scalars = []
            for batch in (left, right):
                prediction = model.predict(batch)
                scale_gradient = 2.0 * torch.mean((prediction - batch.y) * prediction)
                scalars.append(scale_gradient.square())
            task_penalties.append(torch.stack(scalars).mean())
        elif name == "mmd":
            left_z = model.encode(left.x[:mmd_max_samples])
            right_z = model.encode(right.x[:mmd_max_samples])
            task_penalties.append(_mmd_rbf(left_z, right_z))
        elif name == "coral":
            left_cov = _covariance(model.encode(left.x))
            right_cov = _covariance(model.encode(right.x))
            task_penalties.append((left_cov - right_cov).square().mean())
        elif name == "grad_align":
            task_penalties.append(
                (head_gradient(model, left) - head_gradient(model, right)).square().mean()
            )
        elif name == "hess_align":
            task_penalties.append(
                (head_hessian(model, left) - head_hessian(model, right)).square().mean()
            )
        else:
            raise ValueError(f"Unknown regularizer: {name}")
    return torch.stack(task_penalties).mean()


REGULARIZERS = ("l1", "l2", "irmv1", "mmd", "coral", "grad_align", "hess_align")
