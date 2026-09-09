"""Head-block curvature and source-only response penalties for Task 3.

The local metric is constructed on the classifier head only.  The inverse
metric is detached in the primary objective, but the source head gradients are
created with ``create_graph=True`` so the penalty updates both the encoder and
the head.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MetricBundle:
    matrix: Tensor
    inverse: Tensor
    eigenvalues: Tensor
    damping: float
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    curvature_object: str


@dataclass(frozen=True)
class CurvatureDiagnostics:
    raw_gradient_disagreement: float
    local_response_disagreement: float
    response_vector_norm: float
    hessian_min_eigenvalue: float
    hessian_max_eigenvalue: float
    damped_condition_number: float
    damping: float
    curvature_object: str
    random_trace_relative_error: float = 0.0
    random_fro_relative_error: float = 0.0


def _head_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    params = tuple(model.head.parameters())  # type: ignore[attr-defined]
    if len(params) != 2:
        raise ValueError("Task 3 expects a linear head with weight and bias parameters")
    return params


def _flatten(tensors: tuple[Tensor, ...]) -> Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def head_gradient_vector(loss: Tensor, model: nn.Module) -> Tensor:
    grads = torch.autograd.grad(
        loss,
        _head_parameters(model),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )
    return _flatten(grads)


def source_forward(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], ...],
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    features: list[Tensor] = []
    logits: list[Tensor] = []
    losses: list[Tensor] = []
    for x, y in batches:
        z = model.encode(x)  # type: ignore[attr-defined]
        out = model.head(z)  # type: ignore[attr-defined]
        features.append(z)
        logits.append(out)
        losses.append(F.cross_entropy(out, y))
    return tuple(losses), tuple(features), tuple(logits)


def head_cross_entropy_hessian(features: tuple[Tensor, ...], logits: tuple[Tensor, ...]) -> Tensor:
    """Exact empirical Hessian/Gauss-Newton matrix for a linear softmax head."""
    with torch.no_grad():
        z = torch.cat([item.detach() for item in features], dim=0)
        out = torch.cat([item.detach() for item in logits], dim=0)
        probs = out.softmax(dim=1)
        n, latent_dim = z.shape
        classes = probs.shape[1]
        total = classes * latent_dim + classes
        hessian = torch.zeros((total, total), dtype=z.dtype, device=z.device)
        cov = torch.diag_embed(probs) - probs[:, :, None] * probs[:, None, :]
        for left in range(classes):
            for right in range(classes):
                coeff = cov[:, left, right] / max(n, 1)
                ww = z.T @ (z * coeff[:, None])
                wb = (z * coeff[:, None]).sum(dim=0)
                bb = coeff.sum()
                l0 = left * latent_dim
                r0 = right * latent_dim
                hessian[l0 : l0 + latent_dim, r0 : r0 + latent_dim] = ww
                hessian[l0 : l0 + latent_dim, classes * latent_dim + right] = wb
                hessian[classes * latent_dim + left, r0 : r0 + latent_dim] = wb
                hessian[classes * latent_dim + left, classes * latent_dim + right] = bb
        return (hessian + hessian.T) / 2.0


def damped_metric(hessian: Tensor, epsilon: float) -> MetricBundle:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    sym = (hessian.detach().double() + hessian.detach().double().T) / 2.0
    dim = sym.shape[0]
    trace_scale = float(torch.trace(sym).abs().item() / max(dim, 1))
    damping = epsilon * max(trace_scale, 1e-8)
    matrix = sym + damping * torch.eye(dim, dtype=sym.dtype, device=sym.device)
    eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) / 2.0)
    min_eig = float(eigenvalues.min().item())
    if min_eig <= 0.0:
        jitter = -min_eig + max(damping, 1e-8)
        matrix = matrix + jitter * torch.eye(dim, dtype=sym.dtype, device=sym.device)
        damping += float(jitter)
        eigenvalues, eigenvectors = torch.linalg.eigh((matrix + matrix.T) / 2.0)
    inverse = (eigenvectors * eigenvalues.clamp_min(1e-12).reciprocal()) @ eigenvectors.T
    min_final = float(eigenvalues.min().item())
    max_final = float(eigenvalues.max().item())
    condition = max_final / max(min_final, 1e-12)
    return MetricBundle(
        matrix=matrix.detach(),
        inverse=inverse.detach(),
        eigenvalues=eigenvalues.detach(),
        damping=float(damping),
        min_eigenvalue=min_final,
        max_eigenvalue=max_final,
        condition_number=float(condition),
        curvature_object="linear-head cross-entropy Gauss-Newton/Hessian + trace damping",
    )


def inverse_via_eigh(matrix: Tensor) -> Tensor:
    values, vectors = torch.linalg.eigh((matrix + matrix.T) / 2.0)
    if bool((values <= 0).any()):
        raise ValueError("matrix must be positive definite")
    return (vectors * values.reciprocal()) @ vectors.T


def matched_random_metric(metric: Tensor, seed: int) -> Tensor:
    """Random SPD metric with the same eigenvalue multiset as ``metric``."""
    values = torch.linalg.eigvalsh((metric.detach() + metric.detach().T) / 2.0)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(metric.shape, dtype=metric.dtype, generator=generator).to(metric.device)
    q, r = torch.linalg.qr(random)
    signs = torch.sign(torch.diag(r)).clamp(min=0).mul(2).sub(1)
    q = q * signs.to(metric.device)
    return (q * values.to(metric.device)) @ q.T


def _disagreement_vectors(losses: tuple[Tensor, ...], model: nn.Module) -> Tensor:
    grads = torch.stack([head_gradient_vector(loss, model) for loss in losses], dim=0)
    return grads - grads.mean(dim=0, keepdim=True)


def penalty_from_centered(centered: Tensor, metric_inverse: Tensor | None = None) -> Tensor:
    if metric_inverse is None:
        return centered.square().sum(dim=1).mean()
    inverse = metric_inverse.detach().to(dtype=centered.dtype, device=centered.device)
    return torch.einsum("bi,ij,bj->", centered, inverse, centered) / centered.shape[0]


def gradient_disagreement_penalty(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], ...],
) -> tuple[Tensor, CurvatureDiagnostics]:
    losses, features, logits = source_forward(model, batches)
    centered = _disagreement_vectors(losses, model)
    raw = penalty_from_centered(centered)
    metric = damped_metric(head_cross_entropy_hessian(features, logits), epsilon=1e-2)
    local = penalty_from_centered(centered, metric.inverse)
    response = (centered @ metric.inverse.T).square().sum(dim=1).mean().sqrt()
    return raw, CurvatureDiagnostics(
        raw_gradient_disagreement=float(raw.detach().cpu()),
        local_response_disagreement=float(local.detach().cpu()),
        response_vector_norm=float(response.detach().cpu()),
        hessian_min_eigenvalue=metric.min_eigenvalue,
        hessian_max_eigenvalue=metric.max_eigenvalue,
        damped_condition_number=metric.condition_number,
        damping=metric.damping,
        curvature_object=metric.curvature_object,
    )


def local_response_penalty(
    model: nn.Module,
    batches: tuple[tuple[Tensor, Tensor], ...],
    *,
    epsilon: float = 1e-2,
    metric_kind: str = "real",
    random_seed: int = 0,
    response_squared: bool = False,
) -> tuple[Tensor, CurvatureDiagnostics]:
    losses, features, logits = source_forward(model, batches)
    centered = _disagreement_vectors(losses, model)
    raw = penalty_from_centered(centered)
    metric = damped_metric(head_cross_entropy_hessian(features, logits), epsilon=epsilon)
    matrix = metric.matrix
    if metric_kind == "identity":
        inverse = torch.eye(centered.shape[1], dtype=centered.dtype, device=centered.device)
        random_trace_error = 0.0
        random_fro_error = 0.0
    elif metric_kind == "random":
        matrix = matched_random_metric(metric.matrix, random_seed)
        inverse = inverse_via_eigh(matrix).detach()
        random_trace_error = float(
            abs(torch.trace(matrix).item() - torch.trace(metric.matrix).item())
            / max(abs(torch.trace(metric.matrix).item()), 1e-12)
        )
        random_fro_error = float(
            abs(torch.linalg.norm(matrix).item() - torch.linalg.norm(metric.matrix).item())
            / max(torch.linalg.norm(metric.matrix).item(), 1e-12)
        )
    elif metric_kind == "real":
        inverse = metric.inverse
        random_trace_error = 0.0
        random_fro_error = 0.0
    else:
        raise ValueError(f"unknown metric_kind: {metric_kind}")

    inverse_for_grad = inverse.to(dtype=centered.dtype, device=centered.device)
    response_vectors = centered @ inverse_for_grad.T
    if response_squared:
        penalty = response_vectors.square().sum(dim=1).mean()
    else:
        penalty = penalty_from_centered(centered, inverse)
    local = penalty_from_centered(centered, metric.inverse)
    response = response_vectors.square().sum(dim=1).mean().sqrt()
    return penalty, CurvatureDiagnostics(
        raw_gradient_disagreement=float(raw.detach().cpu()),
        local_response_disagreement=float(local.detach().cpu()),
        response_vector_norm=float(response.detach().cpu()),
        hessian_min_eigenvalue=metric.min_eigenvalue,
        hessian_max_eigenvalue=metric.max_eigenvalue,
        damped_condition_number=metric.condition_number,
        damping=metric.damping,
        curvature_object=metric.curvature_object,
        random_trace_relative_error=random_trace_error,
        random_fro_relative_error=random_fro_error,
    )


def metric_inverse_solve_agrees(metric: Tensor, tolerance: float = 1e-6) -> bool:
    rhs = torch.eye(metric.shape[0], dtype=metric.dtype, device=metric.device)
    direct = torch.linalg.solve(metric, rhs)
    spectral = inverse_via_eigh(metric)
    error = torch.linalg.norm(direct - spectral) / max(torch.linalg.norm(direct).item(), 1e-12)
    return bool(error.item() < tolerance)


def finite_value(value: Tensor) -> bool:
    return bool(torch.isfinite(value.detach()).all().item() and not math.isnan(float(value.detach())))
