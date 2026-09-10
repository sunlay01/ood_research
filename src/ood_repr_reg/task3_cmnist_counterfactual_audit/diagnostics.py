"""Counterfactual diagnostics for scalar-logit CPU-minimal CMNIST models."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from ..task3_cmnist_cpu_minimal.model import CPUColoredMNISTMLP
from .probe import ColorCounterfactualProbe


@dataclass(frozen=True)
class WhiteningResult:
    center: Tensor
    transform: Tensor
    eigenvalues: Tensor
    retained: Tensor
    threshold: float


def covariance(z: Tensor) -> Tensor:
    centered = z - z.mean(dim=0, keepdim=True)
    return centered.T @ centered / max(int(z.shape[0]) - 1, 1)


def whitener(z: Tensor, *, relative_tolerance: float = 1e-5) -> WhiteningResult:
    z64 = z.detach().cpu().double()
    center = z64.mean(dim=0)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance(z64))
    threshold = float(eigenvalues.max().clamp_min(1e-12).item() * relative_tolerance)
    retained = eigenvalues > threshold
    inverse_root = torch.where(
        retained,
        eigenvalues.clamp_min(threshold).rsqrt(),
        torch.zeros_like(eigenvalues),
    )
    transform = eigenvectors @ torch.diag(inverse_root) @ eigenvectors.T
    return WhiteningResult(center=center, transform=transform, eigenvalues=eigenvalues, retained=retained, threshold=threshold)


def _as_vector(values: Tensor, name: str) -> Tensor:
    flattened = values.detach().cpu().double().reshape(-1)
    if flattened.ndim != 1:
        raise ValueError(f"{name} must be vector-like")
    return flattened


def _class_mean(values: Tensor, labels: Tensor, label: int) -> Tensor:
    mask = labels == float(label)
    if int(mask.sum().item()) == 0:
        raise ValueError(f"clean label {label} is absent from counterfactual probe")
    return values[mask].mean(dim=0)


def compute_counterfactual_diagnostics(
    *,
    red_z: Tensor,
    green_z: Tensor,
    red_logits: Tensor,
    green_logits: Tensor,
    head_weight: Tensor,
    clean_labels: Tensor,
    relative_tolerance: float = 1e-5,
) -> dict[str, float | int | bool]:
    """Compute representation and scalar-head color diagnostics from latent pairs."""
    red_z64 = red_z.detach().cpu().double()
    green_z64 = green_z.detach().cpu().double()
    if red_z64.shape != green_z64.shape or red_z64.ndim != 2:
        raise ValueError("red_z and green_z must have matching shape [N, D]")
    labels = _as_vector(clean_labels, "clean_labels")
    if labels.numel() != red_z64.shape[0]:
        raise ValueError("clean_labels must have one entry per probe example")
    red_l = _as_vector(red_logits, "red_logits")
    green_l = _as_vector(green_logits, "green_logits")
    if red_l.numel() != labels.numel() or green_l.numel() != labels.numel():
        raise ValueError("logits must have one entry per probe example")
    w = head_weight.detach().cpu().double().reshape(-1)
    if w.numel() != red_z64.shape[1]:
        raise ValueError("head_weight dimension must match latent dimension")

    pooled = torch.cat((red_z64, green_z64), dim=0)
    whitening = whitener(pooled, relative_tolerance=relative_tolerance)
    red_white = (red_z64 - whitening.center) @ whitening.transform
    green_white = (green_z64 - whitening.center) @ whitening.transform
    color_delta = green_white - red_white
    color_energy = color_delta.square().sum(dim=1).mean()

    balanced_white = 0.5 * (red_white + green_white)
    task_direction = _class_mean(balanced_white, labels, 1) - _class_mean(balanced_white, labels, 0)
    task_signal = task_direction.square().sum()
    if float(color_energy.item()) == 0.0 or float(task_direction.norm().item()) == 0.0:
        overlap = torch.tensor(float("nan"), dtype=torch.double)
        overlap_degenerate = True
    else:
        task_unit = task_direction / task_direction.norm()
        overlap = (color_delta @ task_unit).square().mean() / color_energy
        overlap_degenerate = False

    logit_delta = green_l - red_l
    probability_delta = torch.sigmoid(green_l) - torch.sigmoid(red_l)
    raw_balanced = 0.5 * (red_z64 + green_z64)
    raw_task_direction = _class_mean(raw_balanced, labels, 1) - _class_mean(raw_balanced, labels, 0)
    task_head_margin = torch.abs(raw_task_direction @ w)
    red_pred = (red_l >= 0.0).double()
    green_pred = (green_l >= 0.0).double()
    balanced_clean_accuracy = 0.5 * ((red_pred == labels).double().mean() + (green_pred == labels).double().mean())
    consistency = (red_pred == green_pred).double().mean()
    flip_rate = 1.0 - consistency

    values = {
        "latent_color_response": float(color_energy.item()),
        "task_signal": float(task_signal.item()),
        "task_color_overlap": float(overlap.item()),
        "task_color_overlap_degenerate": bool(overlap_degenerate),
        "prediction_color_response": float(logit_delta.square().mean().item()),
        "probability_color_response": float(probability_delta.square().mean().item()),
        "task_head_margin": float(task_head_margin.item()),
        "balanced_clean_accuracy": float(balanced_clean_accuracy.item()),
        "counterfactual_prediction_consistency": float(consistency.item()),
        "counterfactual_prediction_flip_rate": float(flip_rate.item()),
        "n_probe_examples": int(labels.numel()),
        "whitener_retained_rank": int(whitening.retained.sum().item()),
        "whitener_threshold": float(whitening.threshold),
    }
    finite_values = [value for key, value in values.items() if key != "task_color_overlap" and isinstance(value, float)]
    values["finite"] = bool(all(math.isfinite(value) for value in finite_values))
    return values


@torch.no_grad()
def _encode_and_logit(
    model: CPUColoredMNISTMLP,
    x: Tensor,
    *,
    device: torch.device | str,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    z_parts: list[Tensor] = []
    logit_parts: list[Tensor] = []
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        batch = x[start:stop].to(device)
        z = model.encode(batch)
        logits = model.head(z)
        z_parts.append(z.detach().cpu())
        logit_parts.append(logits.detach().cpu())
    return torch.cat(z_parts, dim=0), torch.cat(logit_parts, dim=0)


def model_counterfactual_diagnostics(
    model: CPUColoredMNISTMLP,
    probe: ColorCounterfactualProbe,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 4096,
    relative_tolerance: float = 1e-5,
) -> dict[str, float | int | bool]:
    """Evaluate a trained model on red/green counterfactual target pairs."""
    was_training = model.training
    model.eval()
    try:
        red_z, red_logits = _encode_and_logit(model, probe.red, device=device, batch_size=batch_size)
        green_z, green_logits = _encode_and_logit(model, probe.green, device=device, batch_size=batch_size)
        head_weight = model.head.weight.detach().cpu().reshape(-1)
        return compute_counterfactual_diagnostics(
            red_z=red_z,
            green_z=green_z,
            red_logits=red_logits,
            green_logits=green_logits,
            head_weight=head_weight,
            clean_labels=probe.clean_labels,
            relative_tolerance=relative_tolerance,
        )
    finally:
        model.train(was_training)
