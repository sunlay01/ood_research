"""Spectral and flatness diagnostics for the common-harness panel."""

from __future__ import annotations

import json
import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .algorithms.fishr import classifier_gradient_matrix
from .smooth_world5 import balanced_indices
from ..task3_cmnist_cpu_minimal.data import ColoredEnvironment


def source_bank(source_envs: tuple[ColoredEnvironment, ColoredEnvironment], *, size_per_environment: int) -> tuple[Tensor, Tensor]:
    images, labels = [], []
    for env in source_envs:
        index = balanced_indices(env.digits, size_per_environment)
        images.append(env.images[index])
        labels.append(env.labels[index])
    return torch.cat(images, dim=0), torch.cat(labels, dim=0)


def _singular_summary(values: Tensor) -> dict[str, Any]:
    s = values.detach().double().cpu()
    total = float(s.sum())
    if s.numel() == 0 or total <= 0.0:
        return {"singular_values": "[]", "spectral_norm": 0.0, "frobenius_norm": 0.0, "nuclear_norm": 0.0, "stable_rank": 0.0, "effective_rank": 0.0, "numerical_rank_1e2": 0, "numerical_rank_1e3": 0, "numerical_rank_1e4": 0, "top1_spectral_mass": 0.0, "top5_spectral_mass": 0.0, "cumulative_explained_spectral_mass": "[]", "condition_number_eps1e12": 0.0}
    probabilities = s / total
    entropy = -float((probabilities * torch.log(probabilities.clamp_min(1e-300))).sum())
    max_s = float(s.max())
    return {
        "singular_values": json.dumps([float(x) for x in s]),
        "spectral_norm": max_s,
        "frobenius_norm": float(torch.sqrt(s.square().sum())),
        "nuclear_norm": total,
        "stable_rank": float(s.square().sum() / max(s[0].square(), torch.tensor(1e-24, dtype=s.dtype))),
        "effective_rank": float(math.exp(entropy)),
        "numerical_rank_1e2": int((s >= 1e-2 * max_s).sum()),
        "numerical_rank_1e3": int((s >= 1e-3 * max_s).sum()),
        "numerical_rank_1e4": int((s >= 1e-4 * max_s).sum()),
        "top1_spectral_mass": float(s[:1].sum() / total),
        "top5_spectral_mass": float(s[:5].sum() / total),
        "cumulative_explained_spectral_mass": json.dumps([float(x) for x in torch.cumsum(probabilities, dim=0)]),
        "condition_number_eps1e12": float(s.max() / s.min().clamp_min(1e-12)),
    }


def weight_spectrum_long_rows(model: nn.Module, *, seed: int, method: str, variant: str, checkpoint: int | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (module_name, module) in enumerate((item for item in model.named_modules() if isinstance(item[1], nn.Linear))):
        rows.append({"seed": int(seed), "method": method, "variant": variant, "checkpoint": checkpoint, "module": module_name, "linear_index": index, **_singular_summary(torch.linalg.svdvals(module.weight.detach().double()))})
    return rows


@torch.no_grad()
def representation_spectrum_rows(model: nn.Module, bank: tuple[Tensor, Tensor], *, seed: int, method: str, variant: str, checkpoint: int | str) -> list[dict[str, Any]]:
    model.eval()
    images, _ = bank
    features = model.encode(images).detach().double()
    return [{"seed": int(seed), "method": method, "variant": variant, "checkpoint": checkpoint, "bank": "source_balanced", **_singular_summary(torch.linalg.svdvals(features))}]


@torch.no_grad()
def gradient_spectrum_rows(model: nn.Module, bank: tuple[Tensor, Tensor], *, seed: int, method: str, variant: str, checkpoint: int | str) -> list[dict[str, Any]]:
    model.eval()
    images, labels = bank
    gradients = classifier_gradient_matrix(model, images, labels).detach().double()
    return [{"seed": int(seed), "method": method, "variant": variant, "checkpoint": checkpoint, "bank": "source_balanced_classifier_grad", **_singular_summary(torch.linalg.svdvals(gradients))}]


def source_loss(model: nn.Module, bank: tuple[Tensor, Tensor]) -> Tensor:
    images, labels = bank
    return F.binary_cross_entropy_with_logits(model(images), labels.float())


def _flatten_tensors(tensors: list[Tensor]) -> Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors]) if tensors else torch.zeros(0)


def _assign_vector_like(parameters: list[nn.Parameter], vector: Tensor) -> list[Tensor]:
    chunks = []
    offset = 0
    for parameter in parameters:
        size = parameter.numel()
        chunks.append(vector[offset: offset + size].reshape_as(parameter).to(parameter))
        offset += size
    return chunks


def _hvp(model: nn.Module, bank: tuple[Tensor, Tensor], vectors: list[Tensor]) -> Tensor:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss = source_loss(model, bank)
    grads = torch.autograd.grad(loss, parameters, create_graph=True)
    dot = sum((grad * vector).sum() for grad, vector in zip(grads, vectors))
    hvp = torch.autograd.grad(dot, parameters, retain_graph=False)
    return _flatten_tensors([value.detach().double() for value in hvp])


def _loss_with_perturbation(model: nn.Module, bank: tuple[Tensor, Tensor], vector: Tensor, *, rho: float) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    chunks = _assign_vector_like(parameters, vector)
    with torch.no_grad():
        for parameter, chunk in zip(parameters, chunks):
            parameter.add_(chunk, alpha=rho)
    value = float(source_loss(model, bank).detach())
    with torch.no_grad():
        for parameter, chunk in zip(parameters, chunks):
            parameter.add_(chunk, alpha=-rho)
    return value


def flatness_diagnostic_rows(model: nn.Module, bank: tuple[Tensor, Tensor], *, seed: int, method: str, variant: str, checkpoint: int | str, config: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = config["diagnostics"]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    loss = source_loss(model, bank)
    grads = torch.autograd.grad(loss, parameters, create_graph=False)
    grad_flat = _flatten_tensors([grad.detach().double() for grad in grads])
    grad_norm = float(grad_flat.norm())
    generator = torch.Generator().manual_seed(int(diagnostics["random_direction_seed"]) + int(seed))
    vector = torch.randn(grad_flat.numel(), generator=generator, dtype=torch.double)
    vector = vector / vector.norm().clamp_min(1e-12)
    top = 0.0
    for _ in range(int(diagnostics["hessian_power_iterations"])):
        hv = _hvp(model, bank, _assign_vector_like(parameters, vector))
        top = float(vector @ hv)
        vector = hv / hv.norm().clamp_min(1e-12)
    trace_values = []
    for probe in range(int(diagnostics["hutchinson_probes"])):
        probe_vec = torch.randint(0, 2, (grad_flat.numel(),), generator=torch.Generator().manual_seed(int(seed) * 1000 + probe), dtype=torch.int64).double() * 2.0 - 1.0
        hv = _hvp(model, bank, _assign_vector_like(parameters, probe_vec))
        trace_values.append(float(probe_vec @ hv))
    base_loss = float(loss.detach())
    random_deltas = []
    for probe in range(int(diagnostics["random_direction_probes"])):
        direction = torch.randn(grad_flat.numel(), generator=torch.Generator().manual_seed(int(seed) * 2000 + probe), dtype=torch.double)
        direction = direction / direction.norm().clamp_min(1e-12)
        random_deltas.append(_loss_with_perturbation(model, bank, direction, rho=float(diagnostics["sharpness_radii"][-1])) - base_loss)
    rows = []
    direction = grad_flat / grad_flat.norm().clamp_min(1e-12)
    for rho in diagnostics["sharpness_radii"]:
        sharp = _loss_with_perturbation(model, bank, direction, rho=float(rho)) - base_loss
        rows.append({
            "method": method,
            "variant": variant,
            "seed": int(seed),
            "checkpoint": checkpoint,
            "source_loss": base_loss,
            "gradient_norm": grad_norm,
            "hessian_top_eigenvalue": top,
            "hessian_trace_estimate": float(sum(trace_values) / len(trace_values)),
            "hutchinson_probes": int(diagnostics["hutchinson_probes"]),
            "sharpness_rho": float(rho),
            "sam_sharpness_delta": sharp,
            "random_direction_mean_delta": float(sum(random_deltas) / len(random_deltas)),
            "random_direction_max_delta": float(max(random_deltas)),
        })
    return rows
